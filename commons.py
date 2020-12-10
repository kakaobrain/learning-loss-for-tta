import itertools
import logging
import os

import colored
import numpy as np
from PIL import Image
from colored.colored import stylize


formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')


def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = 0
        logger.handlers.clear()
        logger.setLevel(level)
        ch = logging.StreamHandler()
        ch.setLevel(level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    print('logger(%s) handler #=%d' % (name, len(logger.handlers)))
    return logger


def add_filehandler(logger, filepath):
    fh = logging.FileHandler(filepath)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


tta_actions = ['n', 'r-20', 'r20', 'z0.8', 'z1.2', 'c', 's0.2', 's0.5', 's3', 's4', 'l0.5', 'l2.0']
tta_num = len(tta_actions)
print('tta #=', tta_num)
print(tta_actions)


def encoded_tta_default():
    return tta_actions.index('n')


def decode(idx):
    if isinstance(idx, (int, float)):
        action = tta_actions[int(abs(idx))]
    elif isinstance(idx, str):
        action = idx
    else:
        raise
    rotation = 0
    bright = 1.0
    zoom = 1.0
    contrast = 0
    color = 1
    sharpen = 1.0
    attention_region = 0
    if action[0] == 'n':
        pass
    elif action[0] == 'r':
        rotation = int(action[1:])
        assert rotation != 0
    elif action[0] == 'b':
        bright = float(action[1:])
        assert bright != 1.0
    elif action[0] == 'z':
        zoom = float(action[1:])
        assert zoom != 1.0
    elif action[0] == 'c':
        contrast = 1
    elif action[0] == 'l':      # T.T
        color = float(action[1:])
    elif action[0] == 's':
        sharpen = float(action[1:])
    elif action[0] == 'a':
        attention_region = int(action[1:])
        assert 1 <= attention_region <= 4, attention_region
    else:
        raise
    if isinstance(idx, (int, float)) and idx < 0:        # TODO: T.T
        tta_flip = 1
    else:
        tta_flip = 0

    return rotation, bright, zoom, contrast, color, sharpen, attention_region, tta_flip


def decode_desc(idx):
    return tta_actions[idx]


def print_log(predictions, corrects=None):
    max_val = max(predictions)
    colors = [237, 237, 237, 237, 240, 243, 246, 249, 252, 255]
    log = ''
    for idx, p in enumerate(predictions):
        color_id = int((p / max_val) * 10 + 0.5)
        color_id = min(9, color_id)
        color_id = max(0, color_id)
        if corrects is not None:
            c = corrects[idx]
        else:
            c = ''
        log += stylize('%.3f%s ' % (p, c), colored.fg(colors[color_id]))
    argmax_id = np.argmax(predictions.detach().cpu().numpy())
    log += str(decode_desc(argmax_id))
    print(log)


def to_color_str(value, max_val):
    colors = [237, 237, 237, 237, 240, 243, 246, 249, 252, 255]
    color_id = int((value / max_val) * 10 + 0.5)
    color_id = min(9, color_id)
    color_id = max(0, color_id)
    return stylize('%.3f' % value, colored.fg(colors[color_id]))


def mirror_expansion(img):
    w, h = img.size
    img_np = np.array(img)
    img_mirror = np.pad(img_np, pad_width=((h, h), (w, w), (0, 0)), mode='symmetric')

    # img_center = img_np[h: 2*h, w: 2*w, :]
    # img.save('debug_orig.png', 'PNG')
    # Image.fromarray(img_center).save('debug_center.png', 'PNG')
    # Image.fromarray(img_np).save('debug_mirror.png', 'PNG')
    # assert np.array_equal(img_center, img_np)

    return Image.fromarray(img_mirror)


def pil_center_crop(img, size):
    # for mirror-padded image
    w, h = img.size
    # assert w % 3 == 0 and h % 3 == 0, (w, h)

    o_w, o_h = int(round(w / 3.)), int(round(h // 3.))

    left = int(round((o_w - size) / 2.))
    top = int(round((o_h - size) / 2.))
    left += o_w
    top += o_h

    right = left + size
    bottom = top + size

    crop_rectangle = (left, top, right, bottom)
    cropped_im = img.crop(crop_rectangle)
    return cropped_im
