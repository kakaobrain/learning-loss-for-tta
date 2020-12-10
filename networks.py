import os
from collections import defaultdict

import torch
from torch.nn.parallel.data_parallel import DataParallel
from torchvision.models.resnet import resnet50
from architectures.efficientnet_pytorch.model import EfficientNet

from commons import tta_num


_pretrain_folder = '/data/private/fast-autoaugment-public/pretrains/'
_pretrain_folder2 = '/data/private/faa-learn4tta/'
_checkpoints = {
    # imagenet ----------------------------------------------------------------
    'resnet50': {
        'default': 'modelzoo',
        'augmix': _pretrain_folder + 'imagenet_resnet50_augmix.pth',
    },
    
    # l2t
    'efficientnet-l2t': defaultdict(lambda: ''),
}


def get_model(model, gpus=None, num_classes=1000, train_aug="default"):
    checkpoint = _checkpoints[model][train_aug]
    print(model, train_aug, checkpoint)
    if 'efficientnet-l2t' in model:
        norm_type = os.environ.get('norm', 'batch')
        model = EfficientNet.from_name(model, num_classes=num_classes, multiple_feat=True, norm_type=norm_type)   # instance norm
    
    # IMAGENET -----------------------------------------------------
    elif model == 'resnet50':
        model = resnet50(pretrained=True)
    else:
        raise ValueError(model)

    if checkpoint and checkpoint != 'modelzoo':
        state = torch.load(checkpoint)
        key = 'model' if 'model' in state else 'state_dict'
        if key in state and not isinstance(state[key], dict):
            key = 'state_dict'
        if 'omem' in train_aug:
            key = 'ema'
        print('model epoch=', state.get('epoch', -1))
        if key in state:
            model.load_state_dict({k.replace('module.', ''): v for k, v in state[key].items()})
        else:
            model.load_state_dict({k.replace('module.', ''): v for k, v in state.items()})  # without key

    if gpus not in ['cpu', None]:
        if len(gpus) > 1:
            model = DataParallel(model, device_ids=gpus)
        model = model.cuda()

    return model
