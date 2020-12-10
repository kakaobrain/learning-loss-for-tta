import json
import math
import os
import logging
import sys

import numpy as np
import torch
from sklearn.model_selection._split import StratifiedShuffleSplit
from theconf.argument_parser import ConfigArgumentParser
from torch.nn.functional import binary_cross_entropy
from torch.utils.data.dataset import Subset
from tqdm._tqdm import tqdm
import colored
from colored import stylize

from commons import tta_actions, tta_num, get_logger, decode_desc, encoded_tta_default, add_filehandler
from dataloader import AugmentedDataset, get_dataset, CorruptedDataset
from dataloader_cache import CachedDataLoader
from architectures.efficientnet_pytorch.ema import EMA
from architectures.efficientnet_pytorch.rmsproptf import RMSpropTF
from metrics import Accumulator, prediction_correlation
from networks import get_model
from profile import Profiler
from remote_dataloader.loader import RemoteDataLoader
from theconf import Config as C
from imagenet_c import corruption_dict
from conf import sodeep_model
from sodeep.sodeep import SpearmanLoss, load_sorter


if __name__ == '__main__':
    logger = get_logger('learn2test-train')
    logger.setLevel(logging.DEBUG)

    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--tag', default='dev', type=str)
    parser.add_argument('--port', default=1958, type=int)
    parser.add_argument('--cv', default=0, type=int)

    parser.add_argument('--regularization', default=2, type=int)
    parser.add_argument('--ema-momentum', default=0.999, type=float)

    parser.add_argument('--data-cache', default=0, type=int)

    parser.add_argument('--cutout', default=-1, type=int)
    parser.add_argument('--aug-corrupt', default=1, type=int)
    parser.add_argument('--aug-p', default=0.4, type=float)
    args = parser.parse_args()

    tv = 'v8'  # 12 tta, bugfix

    logpath = f'models/{tv}_{args.dataset}_{args.target_network}_{args.target_aug}_{args.tag}'
    os.system(f'mkdir -p {logpath}')
    add_filehandler(logger, os.path.join(logpath, 'log.txt'))

    logger.info('------- Start New Experiment ------')
    logger.info(f'logpath = {logpath}')
    logger.info(json.dumps(C.get().conf, indent=4))
    logger.info(str(tta_actions))

    """
    dataset protocol.
    Split 'train' set -> train / valid for loss predictor.
    Use 'test' set -> test for loss predictor & TTA performance
    """
    set_tr, set_aug, set_def = get_dataset(args.dataset)
    num_class = set_aug.num_class
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for _ in range(args.cv + 1):
        sss = sss.split(list(range(len(set_tr))), set_tr.targets)
    train_idx, valid_idx = next(sss)
    logger.info('dataset size=%d %d' % (len(train_idx), len(valid_idx)))

    set_aug_tr = Subset(set_tr, train_idx)
    set_aug_vl = Subset(set_tr, valid_idx)
    set_aug_ts = set_aug
    set_def_ts = set_def

    set_aug_tr.num_class = set_aug_vl.num_class = set_aug_ts.num_class = set_def_ts.num_class = num_class

    """
    tar_model : target model
    l2t_model : model which predict relative losses from tar_model
    ema_model : EMA-ed l2t_model
    """
    tar_model = get_model(args.target_network, gpus=[0], num_classes=num_class, train_aug=args.target_aug).eval()
    l2t_model = get_model(args.network, gpus=[0], num_classes=tta_num).train()
    ema_model = get_model(args.network, gpus=[0], num_classes=tta_num).train()
    ema = EMA(args.ema_momentum)

    profiler = Profiler(tar_model)
    logger.info(f'target network, FLOPs={profiler.flops(torch.zeros((1, 3, C.get()["target_size"], C.get()["target_size"])).cuda(), )}')

    profiler = Profiler(l2t_model)
    logger.info(f'L2T network, FLOPs={profiler.flops(torch.zeros((32, 3, C.get()["size"], C.get()["size"])).cuda(), ) / 32.}')

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(l2t_model.parameters(), args.lr, momentum=args.momentum, weight_decay=0.0, nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(l2t_model.parameters(), args.lr, weight_decay=0.0, amsgrad=True)
    elif args.optimizer == 'rmsproptf':
        optimizer = RMSpropTF(l2t_model.parameters(), args.lr, weight_decay=0.0, alpha=0.9, momentum=0.9, eps=0.001)
    else:
        raise ValueError(args.optimizer)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=0.)

    scaled_size = int(math.floor(args.target_size / 0.875))
    padding, cutout = (scaled_size - args.target_size, args.target_size // 3) if args.dataset == 'imagenet' else (0, 12)
    if args.cutout >= 0:
        cutout = args.cutout

    norm = args.dataset
    logger.info(f'norm={norm} target_aug={args.target_aug} padding={padding} cutout={cutout}')
    MAX_CACHE_EPOCH = 90

    cache_prefix = f'{tv}_{args.dataset}_{args.target_network}{args.target_aug}_{args.network}_augp={args.aug_p}'
    logger.info(cache_prefix)

    augmented_dataset = CorruptedDataset(set_aug_tr, args.target_network, args.target_size,
                                         l2t_size=args.size, padding=padding, norm=norm,
                                         num_sample=args.num_duplication, cutout=cutout, target_aug=args.target_aug, aug_p=args.aug_p,
                                         do_random_corrupt=args.aug_corrupt)
    augmented_dataset_v = CorruptedDataset(set_aug_vl, args.target_network, args.target_size,
                                         l2t_size=args.size, padding=padding, norm=norm,
                                         num_sample=1, cutout=0, target_aug=args.target_aug, aug_p=args.aug_p,
                                         do_random_corrupt=args.aug_corrupt, do_transform=False)
    augmented_dataset_t_aug = CorruptedDataset(set_aug_ts, args.target_network, args.target_size,
                                         l2t_size=args.size, padding=padding, norm=norm,
                                         num_sample=1, cutout=0, target_aug=args.target_aug, aug_p=1.0,
                                         do_random_corrupt=args.aug_corrupt, do_transform=False)

    # clean test-set
    augmented_dataset_t = AugmentedDataset(set_def_ts, args.target_network, args.target_size, [0], [1.0], [1.0], [False], args.size, padding,
                                           norm=norm, target_aug=args.target_aug, num_sample=1, cutout=0, d_tta_prob=0.0)

    trainloader = RemoteDataLoader(augmented_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=True, listen='*:%d' % (args.port + 0), timeout=600)
    validloader = RemoteDataLoader(augmented_dataset_v, batch_size=1, shuffle=False, pin_memory=True, drop_last=False, listen='*:%d' % (args.port + 1), timeout=600)
    testloadera = RemoteDataLoader(augmented_dataset_t_aug, batch_size=1, shuffle=False, pin_memory=True, drop_last=False, listen='*:%d' % (args.port + 2), timeout=600)
    testloader2 = RemoteDataLoader(augmented_dataset_t, batch_size=1, shuffle=False, pin_memory=True, drop_last=False, listen='*:%d' % (args.port + 3), timeout=600)

    if args.data_cache:
        logger.info(f'---- use data cache @ {cache_prefix} ---- ')
        trainloader = CachedDataLoader(trainloader, tag=f'{cache_prefix}_tr', cache_max=MAX_CACHE_EPOCH)
        validloader = CachedDataLoader(validloader, tag=f'{cache_prefix}_ts2', cache_max=CachedDataLoader.NO_SEPARATE_EPOCH)
        testloadera = CachedDataLoader(testloadera, tag=f'{cache_prefix}_tsa', cache_max=CachedDataLoader.NO_SEPARATE_EPOCH)
        testloader2 = CachedDataLoader(testloader2, tag=f'{cache_prefix}_ts3', cache_max=CachedDataLoader.NO_SEPARATE_EPOCH)

    start_epoch = 0
    spearman_loss = SpearmanLoss(*load_sorter(sodeep_model)).cuda()

    def run_epoch(_loader, _model, _optimizer, _tag, _ema, _epoch, _scheduler=None, max_step=100000):
        params_without_bn = [params for name, params in _model.named_parameters() if not ('_bn' in name or '.bn' in name)]

        tta_cnt = [0] * tta_num
        metric = Accumulator()
        batch = []
        total_steps = len(_loader)
        tqdm_loader = tqdm(_loader, desc=f'[{_tag} epoch={_epoch+1:03}/{args.epoch:03}]', total=min(max_step, total_steps))
        try:
            for example_id, (img_orig, lb, losses, corrects) in enumerate(tqdm_loader):
                batch.append((img_orig, lb, losses, corrects))
                if (example_id + 1) % args.batch != 0:
                    continue

                if max_step < example_id:
                    break

                imgs = torch.cat([x[0] for x in batch]).cuda()
                lbs = torch.cat([x[1] for x in batch]).long().cuda()
                losses = torch.cat([x[2] for x in batch]).cuda()
                corrects = torch.cat([x[3] for x in batch]).cuda()
                assert len(imgs) > 0

                imgs = imgs.view(imgs.size(0) * imgs.size(1), imgs.size(2), imgs.size(3), imgs.size(4))
                lbs = lbs.view(lbs.size(0) * lbs.size(1))
                losses = losses.view(losses.size(0) * losses.size(1), -1)
                corrects = corrects.view(corrects.size(0) * corrects.size(1), -1)
                assert losses.shape[1] == tta_num, losses.shape
                assert corrects.shape[1] == tta_num, corrects.shape
                assert torch.isnan(losses).sum() == 0

                softmin_target = torch.nn.functional.softmin(losses / args.tau, dim=1).detach()
                pred = _model(imgs)
                pred_softmax = torch.nn.functional.softmax(pred, dim=1)
                assert torch.isnan(pred).sum() == 0, pred
                assert torch.isnan(pred_softmax).sum() == 0, pred_softmax
                assert torch.isnan(softmin_target).sum() == 0
                assert softmin_target.shape[0] == pred_softmax.shape[0], (softmin_target.shape, pred_softmax.shape)
                assert softmin_target.shape[1] == pred_softmax.shape[1], (softmin_target.shape, pred_softmax.shape)

                pred_final = pred_softmax
                loss = spearman_loss(pred_softmax, softmin_target)

                if _optimizer is not None:
                    loss_total = loss + args.decay * sum([torch.norm(p, p=args.regularization) for p in params_without_bn])
                    loss_total.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                if _ema is not None:
                    _ema(_model, _epoch * total_steps + example_id)

                for idx in torch.argmax(pred_softmax, dim=1):
                    tta_cnt[idx] += 1

                pred_correct = torch.Tensor([x[y] for x, y in zip(corrects, torch.argmax(pred_final, dim=1))])
                orac_correct = torch.Tensor([x[y] for x, y in zip(corrects, torch.argmax(softmin_target, dim=1))])
                defa_correct = corrects[:, encoded_tta_default()]

                pred_loss = torch.Tensor([x[y] for x, y in zip(losses, torch.argmax(pred_final, dim=1))])
                defa_loss = losses[:, encoded_tta_default()]
                corr_p = prediction_correlation(pred_final, softmin_target)

                metric.add('loss', loss.item())
                metric.add('l_l2t', torch.mean(pred_loss).item())
                metric.add('l_org', torch.mean(defa_loss).item())
                metric.add('top1_l2t', torch.mean(pred_correct).item())
                metric.add('top1_oracle', torch.mean(orac_correct).item())
                metric.add('top1_org', torch.mean(defa_correct).item())
                metric.add('corr_p', corr_p)
                metric.add('cnt', 1)
                tqdm_loader.set_postfix(
                    lr=_optimizer.param_groups[0]['lr'] if _optimizer is not None else 0,
                    l=metric['loss'] / metric['cnt'],
                    l_l2t=metric['l_l2t'] / metric['cnt'],
                    l_org=metric['l_org'] / metric['cnt'],
                    # l_curr=loss.item(),
                    corr_p=metric['corr_p'] / metric['cnt'],
                    acc_l2t=metric['top1_l2t'] / metric['cnt'],
                    acc_org=metric['top1_org'] / metric['cnt'],
                    acc_d=(metric['top1_l2t'] - metric['top1_org']) / metric['cnt'],
                    acc_O=metric['top1_oracle'] / metric['cnt'],
                    # tta_top=decode_desc(np.argmax(tta_cnt)),
                    # tta_max='%.2f(%d)' % (max(tta_cnt) / float(sum(tta_cnt)), np.argmax(tta_cnt)),
                    ttas=f'{tta_cnt[0]/sum(tta_cnt):.2f},{tta_cnt[-3]/sum(tta_cnt):.2f},{tta_cnt[-2]/sum(tta_cnt):.2f},{tta_cnt[-1]/sum(tta_cnt):.2f}'
                    # tta_min='%.2f' % (min(tta_cnt) / float(sum(tta_cnt))),
                    # grad_l2=metric['grad_l2'] / metric['cnt'],
                )

                batch = []
                if _scheduler is not None:
                    _scheduler.step(_epoch + (float(example_id) / total_steps))
                del pred, loss
        except KeyboardInterrupt as e:
            if 'test' not in _tag:
                raise e
            pass
        finally:
            tqdm_loader.close()

        del tqdm_loader, batch

        if 'test' in _tag:
            if metric['top1_l2t'] >= metric['top1_org']:
                c = 107     # green
            else:
                c = 124     # red

        else:
            if metric['top1_l2t'] >= metric['top1_org']:
                c = 149
            else:
                c = 14      # light_cyan
        logger.info(f'[{_tag} epoch={_epoch + 1}] ' + stylize(
            'loss=%.4f l(l2t=%.4f org=%.4f) top1_O=%.4f top1_org=%.4f << corr_p=%.4f delta=%.4f %s(%s)>>' %
            (metric['loss'] / metric['cnt'],
             metric['l_l2t'] / metric['cnt'], metric['l_org'] / metric['cnt'],
             metric['top1_oracle'] / metric['cnt'],
             metric['top1_l2t'] / metric['cnt'],
             metric['top1_org'] / metric['cnt'],
             metric['corr_p'] / metric['cnt'],
             (metric['top1_l2t'] / metric['cnt']) - (metric['top1_org'] / metric['cnt']),
             decode_desc(np.argmax(tta_cnt)), '%.2f(%d)' % (max(tta_cnt) / float(sum(tta_cnt)), np.argmax(tta_cnt)),
             )
        , colored.fg(c)))
        return metric

    logger.info('start to train a learn2test model.')
    for epoch in range(start_epoch, args.epoch):
        if epoch > 0:
            scheduler.step(epoch)
        if 0 < C.get()['early_stop'] <= epoch:
            break
        l2t_model.train()
        spearman_loss.train()

        metric = run_epoch(trainloader, l2t_model, optimizer, 'train', ema, epoch, _scheduler=scheduler)
        save_path = 'epoch%05d.pth' % (epoch + 1)
        save_path = os.path.join(logpath, save_path)
        torch.save({
            'model': l2t_model.state_dict(),
            'model_ema': ema_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
        }, save_path)

    logger.info('------------------------------------- EVAL -------------------------------------')
    spearman_loss.eval()
    l2t_model.eval()
    ema_model.eval()
    ema_model.load_state_dict(ema.state_dict())

    val_top = 0.0
    with torch.no_grad():
        metric_v_l2t = run_epoch(validloader, l2t_model, None, 'test(valid, Model)', None, epoch, max_step=10000)
        metric_v_ema = run_epoch(validloader, ema_model, None, 'test(valid, EMA)', None, epoch, max_step=10000)

        _ = run_epoch(testloadera, l2t_model, None, 'test(aug-test, Model)', None, epoch)
        _ = run_epoch(testloadera, ema_model, None, 'test(aug-test, EMA)', None, epoch)

        metric_t_l2t = run_epoch(testloader2, l2t_model, None, 'test(test, Model)', None, epoch)
        metric_t_ema = run_epoch(testloader2, ema_model, None, 'test(test, EMA)', None, epoch)
        metric_t = metric_t_ema

    save_path = os.path.join(logpath, 'conf.json')
    with open(save_path, 'w') as f:
        json.dump(C.get().conf, f)
