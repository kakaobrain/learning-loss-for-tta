import itertools
import logging
import math
import os
import pickle
import random
from collections import defaultdict

import PIL
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.model_selection._split import StratifiedShuffleSplit
from theconf.argument_parser import ConfigArgumentParser
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms
from torchvision.transforms.transforms import Compose, Resize
from tqdm._tqdm import tqdm
from theconf import Config as C

from commons import tta_num, get_logger, decode, \
    print_log, encoded_tta_default, mirror_expansion
from conf import dataroot
from imagenet import ImageNet
from metrics import accuracy, Accumulator
from networks import get_model
from profile import Profiler

logger = get_logger('learn2test')
logger.setLevel(logging.DEBUG)


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--test-batch', type=int, default=32)
    parser.add_argument('--tta', type=str, default='center')
    parser.add_argument('--deform', type=str, default='')
    parser.add_argument('--corrupt', type=str, default='')
    args = parser.parse_args()

    assert args.dataset == 'imagenet'

    model_target = get_model(args.target_network, gpus=[0], num_classes=args.num_classes, train_aug=args.target_aug).eval()
    profiler = Profiler(model_target)
    print('target network, FLOPs=', profiler.flops(torch.zeros((1, 3, C.get()['target_size'], C.get()['target_size'])).cuda(), ))

    scaled_size = int(math.floor(args.target_size / 0.875))

    if args.deform != '':
        deform_type, deform_level = args.deform.split(' ')
        if deform_type in ['rotate', 'rotation']:
            t = torchvision.transforms.Lambda(lambda img_orig: torchvision.transforms.functional.rotate(img_orig, int(deform_level), resample=PIL.Image.BICUBIC))
        elif deform_type == 'bright':
            t = torchvision.transforms.Lambda(lambda img_orig: torchvision.transforms.functional.adjust_brightness(img_orig, float(deform_level)))
        elif deform_type == 'zoom':
            resize = int(scaled_size * float(deform_level))
            t = torchvision.transforms.Lambda(lambda img_orig: torchvision.transforms.functional.resize(img_orig, resize, interpolation=PIL.Image.BICUBIC))
        elif deform_type:
            raise ValueError('Invalid Deformation=%s' % deform_type)
    else:
        t = None

    if args.tta == 'center':
        ts = [
            transforms.Resize(scaled_size, interpolation=Image.BICUBIC),
            transforms.CenterCrop(args.target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    elif args.tta == '5crop':
        ts = [
            transforms.Resize((scaled_size, scaled_size), interpolation=Image.BICUBIC),
            transforms.FiveCrop(args.target_size),
            transforms.Lambda(lambda crops: torch.stack([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
                    transforms.ToTensor()(crop)
                ) for crop in crops
            ]))
            # transforms.Lambda(lambda crops: torch.stack([
            #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(
            #         transforms.ToTensor()(crop) for crop in crops
            #     )
            # ])),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    else:
        raise

    corrupt_idx = 1
    if t is not None:
        ts.insert(1, t)
        corrupt_idx = 2

    if args.corrupt != '':
        corrupt_type, corrupt_level = args.corrupt.split(':')
        corrupt_level = int(corrupt_level)
        print(f'corruption {corrupt_type} : {corrupt_level}')

        from imagenet_c import corrupt
        if not corrupt_type.isdigit():
            ts.insert(corrupt_idx, lambda img: PIL.Image.fromarray(corrupt(np.array(img), corrupt_level, corrupt_type)))
        else:
            ts.insert(corrupt_idx, lambda img: PIL.Image.fromarray(corrupt(np.array(img), corrupt_level, None, int(corrupt_type))))

    transform_test = transforms.Compose(ts)

    testset = ImageNet(root='/data/public/rw/datasets/imagenet-pytorch', split='val', transform=transform_test)
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for _ in range(1):
        sss = sss.split(list(range(len(testset))), testset.targets)
    train_idx, valid_idx = next(sss)
    testset = Subset(testset, valid_idx)

    testloader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch, shuffle=False, num_workers=32, pin_memory=True, drop_last=False)

    metric = Accumulator()
    dl_test = tqdm(testloader)
    data_id = 0
    tta_rule_cnt = [0] * tta_num
    for data, label in dl_test:
        data = data.view(-1, data.shape[-3], data.shape[-2], data.shape[-1])
        data = data.cuda()

        with torch.no_grad():
            preds = model_target(data)
            preds = torch.softmax(preds, dim=1)

        preds = preds.view(len(label), -1, preds.shape[-1])

        preds_merged = torch.mean(preds, dim=1)     # simple averaging
        # TODO : weighted average mean?
        # preds_merged = torch.max(preds, dim=1)[0]       # simple maximum peak
        # preds_merged = torch.mean(torch.softmax(preds / 0.1, dim=2), dim=1)       # hard voting?

        (top1, top5), _ = accuracy(preds_merged.cuda(), label.cuda(), (1, 5))
        metric.add_dict({
            'top1': top1.item() * len(data),
            'top5': top5.item() * len(data),
            'cnt': len(data)
        })

        dl_test.set_postfix(metric / metric['cnt'])

    # print('----')
    # print(odaps)
    print('----')
    print('evaluation done.')
    print(metric / metric['cnt'])
