import json
import logging
import itertools
import math
import os
import pickle
import random
import sys
from collections import defaultdict

import PIL
import numpy as np
import torch
import torchvision
from PIL import Image
from PIL import ImageEnhance
from sklearn.model_selection._split import StratifiedShuffleSplit
from theconf.argument_parser import ConfigArgumentParser
from torch.utils.data.dataset import Dataset, Subset
from torchvision.datasets.folder import ImageFolder
from torchvision.transforms import transforms
from torchvision.transforms.transforms import Compose, Resize, CenterCrop
from tqdm._tqdm import tqdm

from commons import tta_num, get_logger, decode, print_log, encoded_tta_default, mirror_expansion, decode_desc, \
    pil_center_crop, tta_actions
from conf import dataroot
from dataloader_cache import CachedDataLoader
from imagenet import ImageNet
from metrics import accuracy, Accumulator
from networks import get_model
from profile import Profiler
from theconf import Config as C
from imagenet_c import corrupt

logger = get_logger('learn2test')
logger.setLevel(logging.DEBUG)


class LearnedAugmentedDataset(Dataset):
    def __init__(self, dataset, augmentor, target_size, padding, deform=None, multicrop=1, num_chain=1):
        if num_chain > 1:
            assert isinstance(augmentor, AugmentorLearned)

        self.dataset = dataset
        self.augment_rule = augmentor
        self.target_size = target_size
        self.deform = deform
        self.padding = padding
        self.last_img = None
        self.num_chain = num_chain
        # logger.info(f'LearnedAugmentedDataset target_size={target_size} padding={padding} deform={deform}')

        self.multicrop = multicrop
        self.t_policy = (('cc', False), ('tl', False), ('tr', False), ('bl', False), ('br', False),
                         ('cc', True), ('tl', True), ('tr', True), ('bl', True), ('br', True))
        assert self.multicrop in [1, 5, 10]

    def __getitem__(self, index):
        img_orig, lb = self.dataset[index]
        # img_orig.save('img_orig.jpg', 'JPEG')

        mirror_expansion_factor = 3
        img_orig = mirror_expansion(img_orig)
        if self.deform != '':
            deform_type, deform_level = self.deform.split(' ')
            if deform_type in ['rotate', 'rotation']:
                img_orig = torchvision.transforms.functional.rotate(img_orig, int(deform_level), resample=PIL.Image.BICUBIC)
            elif deform_type in ['bright', 'brightness']:
                img_orig = torchvision.transforms.functional.adjust_brightness(img_orig, float(deform_level))
            elif deform_type == 'zoom':
                resize = int((self.target_size + self.padding) * mirror_expansion_factor * float(deform_level))
                img_orig = torchvision.transforms.functional.resize(img_orig, resize, interpolation=PIL.Image.BICUBIC)
            elif deform_type == 'hflip':
                img_orig = torchvision.transforms.functional.hflip(img_orig)
            elif deform_type:
                raise ValueError('Invalid Deformation=%s' % deform_type)

        new_resize = int((self.target_size + self.padding) * mirror_expansion_factor)
        img_orig = torchvision.transforms.functional.resize(img_orig, new_resize, interpolation=PIL.Image.BICUBIC)

        assert self.num_chain >= 1
        for _ in range(self.num_chain):
            img_orig_center = pil_center_crop(img_orig, self.target_size)
            tta_rules, tta_p = self.augment_rule(img_orig_center)
            tta_id = tta_rules[0]       # TODO : select first tta

            if tta_id == encoded_tta_default():
                break
            if _ == self.num_chain - 1:
                break

            tta_rotate, tta_brightness, tta_zoom, tta_contrast, tta_color, tta_sharpen, tta_flip = decode(tta_id)
            img = img_orig.copy()
            if tta_rotate != 0:
                img = torchvision.transforms.functional.rotate(img, tta_rotate, resample=PIL.Image.BICUBIC)
            if tta_brightness != 1.0:
                img = torchvision.transforms.functional.adjust_brightness(img, tta_brightness)
            if tta_zoom != 1.0:
                img = torchvision.transforms.functional.resize(img, int(new_resize * tta_zoom), interpolation=PIL.Image.BICUBIC)
            if tta_contrast > 0.0:
                img = PIL.ImageOps.autocontrast(img)
            if tta_color != 1.0:
                img = PIL.ImageEnhance.Color(img).enhance(tta_color)
            if tta_sharpen != 1.0:
                img = ImageEnhance.Sharpness(img).enhance(tta_sharpen)
            if tta_flip == 1:
                img = torchvision.transforms.functional.hflip(img)
            img_orig = img

        imgs = []
        for tta_id in tta_rules:
            tta_rotate, tta_brightness, tta_zoom, tta_contrast, tta_color, tta_sharpen, tta_att, tta_flip = decode(tta_id)
            img = img_orig.copy()
            if tta_rotate != 0:
                img = torchvision.transforms.functional.rotate(img, tta_rotate, resample=PIL.Image.BICUBIC)
            if tta_brightness != 1.0:
                img = torchvision.transforms.functional.adjust_brightness(img, tta_brightness)
            if tta_zoom != 1.0:
                img = torchvision.transforms.functional.resize(img, int(new_resize * tta_zoom), interpolation=PIL.Image.BICUBIC)
            if tta_contrast > 0.0:
                img = PIL.ImageOps.autocontrast(img)
            if tta_color != 1.0:
                img = PIL.ImageEnhance.Color(img).enhance(tta_color)
            if tta_sharpen != 1.0:
                img = ImageEnhance.Sharpness(img).enhance(tta_sharpen)
            if tta_flip == 1:
                img = torchvision.transforms.functional.hflip(img)

            crop_width = crop_height = self.target_size
            # print(tta_action, 'orig.size=', img_orig.size, 'zoom.size=', img_zoom.size, 'img_pad.size', img_pad.size, 'target_size=', self.target_size, 'padding=', self.padding)
            tta_att_pad = self.padding if self.padding else 4
            if tta_att == 0:
                # img = pil_center_crop(img, self.target_size)
                pass
            elif tta_att == 1:
                img = pil_center_crop(img, int(self.target_size + tta_att_pad))
                img = img.crop((0, 0, crop_width, crop_height))
            elif tta_att == 2:
                img = pil_center_crop(img, int(self.target_size + tta_att_pad))
                image_width, image_height = img.size
                img = img.crop((image_width - crop_width, 0, image_width, crop_height))
            elif tta_att == 3:
                img = pil_center_crop(img, int(self.target_size + tta_att_pad))
                image_width, image_height = img.size
                img = img.crop((0, image_height - crop_height, crop_width, image_height))
            elif tta_att == 4:
                img = pil_center_crop(img, int(self.target_size + tta_att_pad))
                image_width, image_height = img.size
                img = img.crop((image_width - crop_width, image_height - crop_height, image_width, image_height))
            else:
                raise Exception

            # img = torchvision.transforms.functional.center_crop(img, self.target_size)
            img_before_crop = img.copy()
            for multicrop_idx in range(self.multicrop):
                img = img_before_crop.copy()
                policy_crop, policy_flip = self.t_policy[multicrop_idx % self.multicrop]

                crop_width = crop_height = self.target_size
                crop_pad = self.padding if self.padding else 4
                if policy_crop == 'cc':
                    img = pil_center_crop(img, self.target_size)
                elif policy_crop == 'tl':
                    img = pil_center_crop(img, int(self.target_size + crop_pad))
                    img = img.crop((0, 0, crop_width, crop_height))
                elif policy_crop == 'tr':
                    img = pil_center_crop(img, int(self.target_size + crop_pad))
                    image_width, image_height = img.size
                    img = img.crop((image_width - crop_width, 0, image_width, crop_height))
                elif policy_crop == 'bl':
                    img = pil_center_crop(img, int(self.target_size + crop_pad))
                    image_width, image_height = img.size
                    img = img.crop((0, image_height - crop_height, crop_width, image_height))
                elif policy_crop == 'br':
                    img = pil_center_crop(img, int(self.target_size + crop_pad))
                    image_width, image_height = img.size
                    img = img.crop((image_width - crop_width, image_height - crop_height, image_width, image_height))
                else:
                    raise Exception

                if policy_flip:
                    img = torchvision.transforms.functional.hflip(img)

                self.last_img = img
                img = torchvision.transforms.functional.to_tensor(img)
                img = torchvision.transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                imgs.append(img)
        imgs = torch.stack(imgs)
        return imgs, lb, tta_rules[0], tta_p

    def __len__(self):
        return len(self.dataset)


class AugmentorDefault:
    def __call__(self, img_orig):
        return [encoded_tta_default()], [1.0]


class AugmentorHFlip:
    def __init__(self, top_k=1):
        self.top_k = top_k

    def __call__(self, img_orig):
        if self.top_k == 1:
            return [-1 * encoded_tta_default() - 0.1], [1.0]  # TODO : -1 for horizontal-flip
        return [encoded_tta_default(), -1 * encoded_tta_default() - 0.1], [0.5, 0.5]       # TODO : -1 for horizontal-flip


class AugmentorRandom:
    def __init__(self, top_k=1):
        self.top_k = top_k

    def __call__(self, img_orig):
        return [random.randint(0, tta_num-1) for _ in range(self.top_k)], [1.0] * self.top_k


class AugmentorLearned:
    def __init__(self, model_l2t, resize=224, top_k=1, th=0.0, force_hflip=False):
        self.model_l2t = model_l2t
        self.resize = resize
        self.top_k = top_k
        self.th = th
        self.force_hflip = force_hflip
        # logger.info(f'AugmentorLearned resize={resize} top_k={top_k}')

    def __call__(self, img_orig):
        img = img_orig.copy()
        if C.get()['dataset'] == 'imagenet':
            img = torchvision.transforms.functional.resize(img, self.resize, interpolation=PIL.Image.BICUBIC)
            # img = torchvision.transforms.functional.center_crop(img, (self.resize, self.resize))
        assert img.size == (self.resize, self.resize)
        img = torchvision.transforms.functional.to_tensor(img)
        img = torchvision.transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        with torch.no_grad():
            pred = self.model_l2t(img.unsqueeze(0))
            pred_normalized = torch.softmax(pred, dim=1)
            pred_normalized = pred_normalized[0]

        flip_ensemble = os.environ.get('flip_ensemble', '1')
        if flip_ensemble:
            img_flip = torch.flip(img, (1, ))
            with torch.no_grad():
                pred_flip = self.model_l2t(img_flip.unsqueeze(0))
                pred_normalized_flip = torch.softmax(pred_flip, dim=1)
                pred_normalized_flip = pred_normalized_flip[0]

                tmp = pred_normalized_flip[tta_actions.index('r-20')].item()
                pred_normalized_flip[tta_actions.index('r-20')] = pred_normalized_flip[tta_actions.index('r20')]
                pred_normalized_flip[tta_actions.index('r20')] = tmp
            pred_normalized = (pred_normalized + pred_normalized_flip) / 2.

        tta_rules = []
        tta_p = []
        include_default = False
        for k in range(self.top_k):
            tta_id = torch.argmax(pred_normalized).item()
            p = pred_normalized[tta_id].item()

            if p < self.th * pred_normalized[encoded_tta_default()].item():
                tta_id = encoded_tta_default()
                if k > 0 and include_default:
                    p = 0.      # not included in ensemble
            if tta_id != encoded_tta_default():
                pred_normalized[tta_id] = 0
            else:
                include_default = True
            tta_p.append(p)
            tta_rules.append(tta_id)

        if self.force_hflip:
            tta_rules.extend([-x-0.1 for x in tta_rules])
            tta_p.extend([x for x in tta_p])

        assert len(tta_rules) > 0
        assert len(tta_rules) == len(tta_p)
        return tta_rules, tta_p


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--test-batch', type=int, default=32)
    parser.add_argument('--augmentor', type=str, default='default')
    parser.add_argument('--force-hflip', type=int, default=0)
    parser.add_argument('--num-aug', type=int, default=1)
    parser.add_argument('--num-chain', type=int, default=1)
    parser.add_argument('--multicrop', type=int, default=1)
    parser.add_argument('--ckpt-key', type=str, default='model_ema')
    parser.add_argument('--th', type=float, default=0.0)
    parser.add_argument('--deform', type=str, default='')
    parser.add_argument('--corrupt', type=str, default='')
    args = parser.parse_args()

    print(json.dumps(C.get().conf, indent=4))

    model_target = get_model(args.target_network, gpus=[0], num_classes=args.num_classes, train_aug=args.target_aug).eval()
    profiler = Profiler(model_target)
    print('target network, FLOPs=', profiler.flops(torch.zeros((1, 3, C.get()['target_size'], C.get()['target_size'])).cuda(), ))

    if '.pth' in args.augmentor:
        if args.augmentor.startswith('models'):
            args.augmentor = args.augmentor.replace('models', '/data/private/learn2test/models')
        torch.set_num_threads(2)
        model_l2t = get_model(args.network, gpus='cpu', num_classes=tta_num).eval()     # use cpu for data-prepare parallelization
        state = torch.load(args.augmentor)
        model_l2t.load_state_dict({k.replace('module.', ''): v for k, v in state[args.ckpt_key].items()})

        profiler = Profiler(model_l2t)
        print('l2t network,    FLOPs=', profiler.flops(torch.zeros((1, 3, C.get()['size'], C.get()['size'])), ))

        augmentor = AugmentorLearned(model_l2t, resize=args.size, top_k=args.num_aug, th=args.th, force_hflip=args.force_hflip)
        if args.force_hflip:
            args.num_aug *= 2
    elif args.augmentor == 'default':
        assert args.num_aug == 1
        augmentor = AugmentorDefault()
    elif args.augmentor == 'hflip':
        args.num_aug = 2
        augmentor = AugmentorHFlip(top_k=args.num_aug)
    elif args.augmentor == 'random':
        augmentor = AugmentorRandom(top_k=args.num_aug)
    else:
        raise ValueError(args.augmentor)

    if args.dataset == 'imagenet':
        scaled_size = int(math.floor(args.target_size / 0.875))
        self_padding = scaled_size - C.get()['target_size']
        # testset = torchvision.datasets.ImageFolder(root=os.path.join(dataroot, 'imagenet/val'))
        ts = [
            Resize(scaled_size, interpolation=PIL.Image.BICUBIC),
            CenterCrop(scaled_size)
        ]

        transform_test = Compose(ts)
        if args.corrupt != '':
            corrupt_type, corrupt_level = args.corrupt.split(':')
            corrupt_level = int(corrupt_level)
            print(f'corruption {corrupt_type} : {corrupt_level}')
        else:
            corrupt_type, corrupt_level = 'clean', 1

        if corrupt_type != 'clean':
            root = f'/data/public/rw/datasets/imagenet-c/{corrupt_type}/{corrupt_level}'
            testset = ImageFolder(root, transform=transform_test)
        else:
            testset = ImageNet(root='/data/public/rw/datasets/imagenet-pytorch', split='val', transform=transform_test)

        # cache_path = f'/data/private/learn2test/_cachedl/imagenet_{args.corrupt}_0000000000' if args.corrupt != '' else ''
        # testset = ImageNet(root='/data/public/rw/datasets/imagenet-pytorch', split='val', transform=transform_test, cache_path=cache_path)

    else:
        raise ValueError(args.dataset)

    testset_tta = LearnedAugmentedDataset(testset, augmentor, args.target_size, padding=self_padding, deform=args.deform, multicrop=args.multicrop, num_chain=args.num_chain)
    testloader = torch.utils.data.DataLoader(testset_tta, batch_size=args.test_batch, shuffle=False, num_workers=24, pin_memory=False, drop_last=False)

    odaps = []
    metric = Accumulator()
    dl_test = tqdm(testloader, disable=os.environ.get('tqdm_disabled', ''))
    data_id = 0
    tmp_correct_cnt = 0
    tmp_correct_bucket = defaultdict(lambda: 0)
    tta_rule_cnt = [0] * tta_num
    for data, label, tta_rule, tta_p in dl_test:
        tta_p = torch.stack(tta_p, dim=1).unsqueeze(-1).float().cuda()
        data = data.view(data.shape[0] * data.shape[1], data.shape[2], data.shape[3], data.shape[4])
        data = data.cuda()

        with torch.no_grad():
            preds = model_target(data)
            preds = torch.softmax(preds, dim=1)

        preds = preds.view(len(label), -1, preds.shape[-1])
        assert preds.shape[1] == args.num_aug * args.multicrop, preds.shape

        # preds_merged = torch.mean(preds * tta_p, dim=1)     # simple averaging
        preds_merged = torch.mean(preds, dim=1)  # simple averaging

        (top1, top5), _ = accuracy(preds_merged.cuda(), label.cuda(), (1, 5))
        metric.add_dict({
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
            'cnt': len(data)
        })

        metric_dict = (metric / metric['cnt']).metrics
        for tta_id in tta_rule:
            tta_rule_cnt[int(tta_id)] += 1
        metric_dict['tta_max'] = max(tta_rule_cnt) / sum(tta_rule_cnt)
        metric_dict['tta_favor'] = decode_desc(np.argmax(tta_rule_cnt))
        dl_test.set_postfix(metric_dict)

        # odaps
        for pred_merged, pred, lb in zip(preds_merged, preds, label):
            if torch.argmax(pred_merged).item() != lb.item():
                odaps.append(data_id)
            pred_lbs = [torch.argmax(pred[x]).item() for x in range(args.num_aug)]
            if lb.item() in pred_lbs:
                tmp_correct_cnt += 1
            tmp_correct_bucket[sum([x == lb.item() for x in pred_lbs])] += 1
            data_id += 1

    with open('odaps.pkl', 'wb') as f:
        pickle.dump(odaps, f, protocol=pickle.HIGHEST_PROTOCOL)
    # print('----')
    # print(odaps)
    print(f'---- corrupt={args.corrupt}')
    for tta_id, tta_rule in enumerate(tta_rule_cnt):
        if tta_rule == 0:
            continue
        print(decode_desc(tta_id), tta_rule)
    if data_id > 0:
        print(tmp_correct_cnt, data_id, float(tmp_correct_cnt) / data_id)
    print(tmp_correct_bucket)
    print('evaluation done.')
    print(metric / metric['cnt'])
    print('top1 acc=')
    print('%.4f' % (metric / metric['cnt'])['top1'])
