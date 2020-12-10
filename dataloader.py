import logging
import math
import os
import pickle
import random
import sys
import time

import PIL, PIL.ImageDraw
from PIL import ImageEnhance
import numpy as np
import torch
import torchvision
from PIL import Image
from sklearn.model_selection._split import StratifiedShuffleSplit
from theconf.argument_parser import ConfigArgumentParser
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data.dataset import Dataset, Subset
from torchvision.transforms.transforms import RandomRotation, RandomResizedCrop, Compose, RandomHorizontalFlip, \
    ColorJitter, Resize, RandomCrop
from tqdm._tqdm import tqdm

from commons import tta_actions, tta_num, decode_desc, get_logger, \
    mirror_expansion, pil_center_crop, decode
from conf import dataroot
from dataloader_cache import CachedDataLoader
from imagenet import ImageNet
from imagenet_c import corruptions
from imagenet_c.corruptions import motion_blur
from imagenet_c import corruption_tuple_tr, corruption_dict, corruption_tuple_vl
from networks import get_model
from remote_dataloader.loader import RemoteDataLoader
from theconf import Config as C
from conf import imagenet_path

logger = get_logger('learn2test-data', logging.DEBUG)

corruption_tuple = corruption_tuple_tr
logger.info(corruption_tuple)


def cutout(img, cutsize=16):
    if cutsize <= 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x1 = int(max(0, x0 - cutsize / 2.))
    y1 = int(max(0, y0 - cutsize / 2.))
    x2 = min(w, x0 + cutsize / 2.)
    y2 = min(h, y0 + cutsize / 2.)

    xy = (x1, y1, x2, y2)
    # color = (125, 123, 114)
    c = random.randint(0, 255)
    color = (c, c, c)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img


class AugmentedDataset(Dataset):
    def __init__(self, dataset, target_network, target_size, transform_r, transform_zs, transform_b, transform_flip, l2t_size, padding, norm, target_aug='default', num_sample=2, cutout=0, d_tta_prob=0.3,
                 do_random_corrupt=False, corrupt_list=None):
        self.dataset = dataset
        self.target_size = target_size
        self.transform_r, self.transform_zs, self.transform_b = transform_r, transform_zs, transform_b
        self.transform_flip = transform_flip
        self.do_random_corrupt = do_random_corrupt
        self.corrupt_list = corrupt_list
        assert type(self.transform_flip) == list
        logger.info('AugmentedDataset r=%s zs=%s b=%s flip=%s norm=%s' % (transform_r, transform_zs, transform_b, transform_flip, norm))

        self.l2t_size = l2t_size
        self.padding = padding
        self.target_network = None
        self.target_network_name = target_network
        self.target_aug = target_aug
        self.norm = norm
        self.is_test = False
        self.num_sample = num_sample
        assert num_sample > 0
        self.cutout = cutout
        self.d_tta_prob = d_tta_prob

        self.orig_img_pil = None
        self.img_pils = []
        assert 1.0 >= d_tta_prob >= 0.0

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self.target_network is None:
            self.target_network = get_model(self.target_network_name, gpus=[0], num_classes=self.dataset.num_class, train_aug=self.target_aug).eval()

        img_orig, lb = self.dataset[idx]
        n_img, n_lb, n_losses, n_corrects = [], [], [], []
        for _ in range(self.num_sample):
            tta_rotate_default = random.choice(self.transform_r) if random.random() < self.d_tta_prob else 0.0
            tta_zoom_default = random.choice(self.transform_zs) if random.random() < self.d_tta_prob else 1.0
            tta_bright_default = random.choice(self.transform_b) if random.random() < self.d_tta_prob else 1.0

            for t_flip in self.transform_flip:
                img_new = img_orig.copy()

                corrupt_name = corrupt_op = corrupt_lv = None
                if self.do_random_corrupt and random.random() < self.d_tta_prob:        # TODO
                    corrupt_op = random.choice(corruption_tuple)
                    corrupt_lv = random.choice([1, 2, 3, 4, 5])
                elif isinstance(self.corrupt_list, list) and random.random() < self.d_tta_prob:           # TODO : Partial Corruptions
                    corrupt_name = random.choice(self.corrupt_list)
                    corrupt_op = corruption_dict[corrupt_name]
                    corrupt_lv = random.choice([1, 2, 3, 4, 5])
                    # corrupt_lv = random.choice([3, 4, 5])

                if corrupt_op is not None:
                    img_np = corrupt_op(img_new, severity=corrupt_lv)
                    if isinstance(img_np, np.ndarray):
                        img_np = img_np.astype(np.uint8)
                        img_new = Image.fromarray(img_np)
                    elif isinstance(img_np, PIL.Image.Image):
                        img_new = img_np
                    else:
                        raise Exception(type(img_np))

                if t_flip:
                    img_new = torchvision.transforms.functional.hflip(img_new)
                mirror_expansion_factor = 3
                try:
                    img_mirror = mirror_expansion(img_new)
                except Exception as e:
                    print(corrupt_op, corrupt_lv)
                    print(e)
                    print(type(img_new))
                    print(img_new.size)
                    raise e
                img_new = img_mirror.copy()

                if tta_rotate_default != 0:
                    img_new = torchvision.transforms.functional.rotate(img_new, tta_rotate_default, expand=False, resample=PIL.Image.BICUBIC)
                assert tta_bright_default > 0
                if tta_bright_default != 1.0:
                    img_new = torchvision.transforms.functional.adjust_brightness(img_new, tta_bright_default)
                new_resize = int((self.target_size + self.padding) * mirror_expansion_factor * tta_zoom_default)
                assert 0.5 < tta_zoom_default < 1.5
                if tta_zoom_default != 1.0:
                    img_new = torchvision.transforms.functional.resize(img_new, new_resize, interpolation=PIL.Image.BICUBIC)

                imgs_pil = []
                for tta_action in tta_actions:
                    tta_rotate, tta_brightness, tta_zoom, tta_contrast, tta_color, tta_blur, tta_att, _ = decode(tta_action)
                    if tta_rotate != 0:
                        img_rotate = torchvision.transforms.functional.rotate(img_new, tta_rotate, expand=False, resample=PIL.Image.BICUBIC)
                    else:
                        img_rotate = img_new.copy()
                    if tta_brightness != 1.0:
                        img_bright = torchvision.transforms.functional.adjust_brightness(img_rotate, tta_brightness)
                    else:
                        img_bright = img_rotate.copy()
                    if tta_zoom != 1.0:
                        resize = int(new_resize * tta_zoom)
                        img_zoom = torchvision.transforms.functional.resize(img_bright, resize, interpolation=PIL.Image.BICUBIC)
                        assert img_zoom.width > 32, (img_zoom.size, img_bright.size)
                    else:
                        img_zoom = img_bright.copy()

                    if tta_contrast > 0.0:
                        img_zoom = PIL.ImageOps.autocontrast(img_zoom)
                    assert img_zoom.width > 32, ('autocont', img_zoom.size, img_bright.size, img_new.size)
                    if tta_color != 1.0:
                        img_zoom = PIL.ImageEnhance.Color(img_zoom).enhance(tta_color)
                    assert img_zoom.width > 32, ('color', img_zoom.size, img_bright.size, img_new.size)
                    if tta_blur != 1.0:
                        img_zoom = ImageEnhance.Sharpness(img_zoom).enhance(tta_blur)
                    assert img_zoom.width > 32, ('blur', img_zoom.size, img_bright.size, img_new.size)

                    w, h = img_zoom.size
                    att_padding = self.padding if self.padding else 0
                    pw, ph = max(0, self.target_size + att_padding - w), max(0, self.target_size + att_padding - h)
                    pw1, ph1 = pw // 2, ph // 2
                    pw2, ph2 = pw - pw1, ph - ph1
                    if pw1 > 0 or ph1 > 0 or pw2 > 0 or ph2 > 0:
                        img_pad = torchvision.transforms.functional.pad(img_zoom, (pw1, ph1, pw2, ph2), random.randint(0, 255), 'reflect')
                    else:
                        img_pad = img_zoom
                    # img = torchvision.transforms.functional.center_crop(img_zoom, (self.target_size, self.target_size))

                    crop_width = crop_height = self.target_size
                    # print(tta_action, 'orig.size=', img_orig.size, 'zoom.size=', img_zoom.size, 'img_pad.size', img_pad.size, 'target_size=', self.target_size, 'padding=', self.padding)
                    if tta_att == 0:
                        img = pil_center_crop(img_pad, self.target_size)
                    elif tta_att == 1:
                        img_pad = pil_center_crop(img_pad, int(self.target_size + att_padding))
                        img = img_pad.crop((0, 0, crop_width, crop_height))
                    elif tta_att == 2:
                        img_pad = pil_center_crop(img_pad, int(self.target_size + att_padding))
                        image_width, image_height = img_pad.size
                        img = img_pad.crop((image_width - crop_width, 0, image_width, crop_height))
                    elif tta_att == 3:
                        img_pad = pil_center_crop(img_pad, int(self.target_size + att_padding))
                        image_width, image_height = img_pad.size
                        img = img_pad.crop((0, image_height - crop_height, crop_width, image_height))
                    elif tta_att == 4:
                        img_pad = pil_center_crop(img_pad, int(self.target_size + att_padding))
                        image_width, image_height = img_pad.size
                        img = img_pad.crop((image_width - crop_width, image_height - crop_height, image_width, image_height))
                    else:
                        raise Exception

                    imgs_pil.append(img)
                    self.img_pils = imgs_pil

                imgs = []
                for img in imgs_pil:
                    img = torchvision.transforms.functional.to_tensor(img)
                    img = torchvision.transforms.functional.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    imgs.append(img)
                imgs = torch.stack(imgs).cuda()
                assert len(imgs) == tta_num

                with torch.no_grad():
                    preds = self.target_network(imgs)
                corrects = (torch.argmax(preds, dim=1).squeeze() == lb).detach().cpu().int().float()
                lbs = torch.tensor([lb] * tta_num).squeeze().cuda()
                # taus = torch.FloatTensor(tta_taus).detach()
                losses = torch.nn.functional.cross_entropy(preds, lbs, reduction='none').detach().cpu()
                del preds
                if self.target_size > 32:   # TODO
                    torch.cuda.empty_cache()

                w, h = img_new.size
                pw, ph = max(0, self.target_size + self.padding - w), max(0, self.target_size + self.padding - h)
                pw1, ph1 = pw // 2, ph // 2
                pw2, ph2 = pw - pw1, ph - ph1
                if pw1 > 0 or ph1 > 0 or pw2 > 0 or ph2 > 0:
                    img_new = torchvision.transforms.functional.pad(img_new, (pw1, ph1, pw2, ph2), random.randint(0, 255), 'reflect')
                if img_new.size[0] >= self.target_size or img_new.size[1] >= self.target_size:
                    # img_new = torchvision.transforms.functional.center_crop(img_new, self.target_size)
                    img_new = pil_center_crop(img_new, self.target_size)
                self.orig_img_pil = img_new

                img_new = cutout(img_new, cutsize=self.cutout * mirror_expansion_factor)

                if self.is_test:
                    return img_mirror, imgs_pil, img_new, losses, corrects

                img_new = torchvision.transforms.functional.resize(img_new, self.l2t_size, interpolation=PIL.Image.BICUBIC)     # TODO
                img_new = torchvision.transforms.functional.to_tensor(img_new)
                img_new = torchvision.transforms.functional.normalize(img_new, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).cpu()
                
                n_img.append(img_new)
                n_lb.append(lb)
                n_losses.append(losses)
                n_corrects.append(corrects)
        return torch.stack(n_img), torch.Tensor(n_lb), torch.stack(n_losses), torch.stack(n_corrects)


class CorruptedDataset(AugmentedDataset):
    def __init__(self, dataset, target_network, target_size, l2t_size, padding, norm, num_sample, cutout,
                 target_aug, aug_p, do_random_corrupt, corrupt_list=None, do_transform=True):
        if do_transform:
            if norm == 'intensive':
                transform_r = [-30, -20, -10, 0, 10, 20, 30]
                transform_zs = [0.8, 0.9, 1.0, 1.1, 1.2]
                transform_b = [0.8, 0.9, 1.0, 1.1, 1.2]
                transform_flip = [False, True]
            else:
                transform_r = [-20, 0, 20]
                transform_zs = [0.9, 1.0, 1.1]
                transform_b = [0.9, 1.0, 1.1]
                transform_flip = [False, True]
        else:
            transform_r = [0]
            transform_zs = [1.0]
            transform_b = [1.0]
            transform_flip = [False]

        super().__init__(dataset, target_network, target_size, transform_r, transform_zs, transform_b,
                         transform_flip, l2t_size, padding, norm,
                         target_aug=target_aug, num_sample=num_sample, cutout=cutout, d_tta_prob=aug_p,
                         do_random_corrupt=do_random_corrupt, corrupt_list=corrupt_list)


def get_dataset(dataset):
    if dataset == 'imagenet':
        transform_train = Compose([
            RandomResizedCrop(C.get()['target_size'] + 32, scale=(0.9, 1.0), interpolation=PIL.Image.BICUBIC),
        ])
        transform_test = Compose([
            Resize(C.get()['target_size'] + 32, interpolation=PIL.Image.BICUBIC)
        ])
        trainset = ImageNet(root=imagenet_path, split='train', transform=transform_train)
        testset1 = ImageNet(root=imagenet_path, split='val', transform=transform_train)
        testset2 = ImageNet(root=imagenet_path, split='val', transform=transform_test)

        trainset.num_class = testset1.num_class = testset2.num_class = 1000
        trainset.targets = [lb for _, lb in trainset.samples]
    else:
        raise ValueError(dataset)
    return trainset, testset1, testset2


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    parser.add_argument('--aug-p', default=0.2, type=float)
    parser.add_argument('--port', default=1958, type=int)
    args = parser.parse_args()

    total_trainset, testset_t, testset = get_dataset(args.dataset)
    num_class = total_trainset.num_class
    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    sss = sss.split(list(range(len(total_trainset))), total_trainset.targets)
    train_idx, valid_idx = next(sss)

    valid_size = 25600
    small_train_size = 2048
    test_size = len(testset)
    validset = Subset(testset_t, valid_idx[:valid_size])  # TODO : max images for validation
    total_trainset = Subset(total_trainset, train_idx)
    small_trainset = Subset(total_trainset, list(range(small_train_size)))
    small_trainset.num_class = validset.num_class = total_trainset.num_class = num_class

    scaled_size = int(math.floor(args.target_size / 0.875))
    padding, cutout = (scaled_size - args.target_size, args.target_size // 3) if args.dataset == 'imagenet' else (0, 12)

    norm = args.dataset
    logger.info(f'norm={norm} target_aug={args.target_aug} padding={padding} cutout={cutout}')
    MAX_CACHE_EPOCH = 90

    tv = os.environ.get('tv', 'v2-tta13l')
    cache_prefix = f'{tv}_{args.dataset}_{args.target_network}{args.target_aug}_{args.network}_augp={args.aug_p}'
    logger.info(cache_prefix)

    augmented_dataset = AugmentedDataset(total_trainset, args.target_network, args.target_size,
                                         [-20, 10, 0, 10, 20], [0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
                                         [0.8, 0.9, 1.0, 1.1, 1.2], [False, True],  # TODO *********
                                         args.size, padding, norm=norm, target_aug=args.target_aug,
                                         num_sample=args.num_duplication, cutout=cutout, d_tta_prob=args.aug_p)
    trainloader = RemoteDataLoader(augmented_dataset, batch_size=1, shuffle=True, pin_memory=True, drop_last=True,
                                   listen='*:%d' % (args.port + 0), timeout=600)

    augmented_dataset_s = AugmentedDataset(small_trainset, args.target_network, args.target_size, [0], [1.0], [1.0], [False], args.size, padding,
                                           norm=norm, target_aug=args.target_aug, num_sample=1, cutout=0,
                                           d_tta_prob=0.0)
    augmented_dataset_v = AugmentedDataset(validset, args.target_network, args.target_size, [0], [1.0], [1.0], [False], args.size, padding,
                                           norm=norm, target_aug=args.target_aug, num_sample=1, cutout=0,
                                           d_tta_prob=0.0)
    augmented_dataset_t_aug = AugmentedDataset(testset, args.target_network, args.target_size, [-20, 10, 0, 10, 20],
                                               [0.8, 0.9, 1.0, 1.1, 1.2, 1.3], [0.8, 0.9, 1.0, 1.1, 1.2], [False, True],
                                               args.size, padding, norm=norm, target_aug=args.target_aug, num_sample=1,
                                               cutout=0, d_tta_prob=args.aug_p)
    augmented_dataset_t = AugmentedDataset(testset, args.target_network, args.target_size, [0], [1.0], [1.0], [False], args.size, padding,
                                           norm=norm, target_aug=args.target_aug, num_sample=1, cutout=0,
                                           d_tta_prob=0.0)

    testloader1 = RemoteDataLoader(augmented_dataset_s, batch_size=1, shuffle=False, pin_memory=True, drop_last=False,
                                   listen='*:%d' % (args.port + 1), timeout=600)
    validloader = RemoteDataLoader(augmented_dataset_v, batch_size=1, shuffle=False, pin_memory=True, drop_last=False,
                                   listen='*:%d' % (args.port + 2), timeout=600)
    testloadera = RemoteDataLoader(augmented_dataset_t_aug, batch_size=1, shuffle=False, pin_memory=True,
                                   drop_last=False, listen='*:%d' % (args.port + 3), timeout=600)
    testloader2 = RemoteDataLoader(augmented_dataset_t, batch_size=1, shuffle=False, pin_memory=True, drop_last=False,
                                   listen='*:%d' % (args.port + 4), timeout=600)

    logger.info(f'---- use data cache @ {cache_prefix} ---- ')
    trainloader = CachedDataLoader(trainloader, tag=f'{cache_prefix}_tr', cache_max=MAX_CACHE_EPOCH)
    # testloader1 = CachedDataLoader(testloader1, tag=f'{cache_prefix}_ts1', cache_max=CachedDataLoader.NO_SEPARATE_EPOCH)
    # validloader = CachedDataLoader(validloader, tag=f'{cache_prefix}_ts2', cache_max=CachedDataLoader.NO_SEPARATE_EPOCH)
    # testloadera = CachedDataLoader(testloadera, tag=f'{cache_prefix}_tsa', cache_max=CachedDataLoader.NO_SEPARATE_EPOCH)
    # testloader2 = CachedDataLoader(testloader2, tag=f'{cache_prefix}_ts3', cache_max=CachedDataLoader.NO_SEPARATE_EPOCH)

    trainloader = tqdm(trainloader)

    for epoch in range(args.epoch):
        cnt = 0
        start_t = time.time()
        for _ in tqdm(trainloader, desc="%04d" % epoch):
            cnt += 1
            if cnt > 38400:
                break
