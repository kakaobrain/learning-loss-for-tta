import base64
from io import BytesIO

import numpy as np
from PIL import Image
from theconf.argument_parser import ConfigArgumentParser
from torchvision import transforms
from tqdm._tqdm import tqdm

from commons import decode_desc, encoded_tta_default
from dataloader import AugmentedDataset, get_dataset
from imagenet import ImageNet

html_body = """
<table>
    {content}
</table>
"""

html_td = """
<td bgcolor="{bgcolor}"><img src="data:image/jpeg;base64,{base64img}"/></br>T={t}</br>loss={loss}</br>soft={softmin}</td>
"""

def pil_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


if __name__ == '__main__':
    parser = ConfigArgumentParser(conflict_handler='resolve')
    args = parser.parse_args()

    # L2T Dataset
    trainset, testset = get_dataset(args.dataset)

    # Legacy Dataset
    transform_test = transforms.Compose([
        transforms.Resize(args.target_size + 32, interpolation=Image.BICUBIC),
        transforms.CenterCrop(args.target_size),
    ])

    testset_legacy = ImageNet(root='/data/public/rw/datasets/imagenet-pytorch', split='val', transform=transform_test)

    # L2T Augmented Dataset
    padding = 32 if args.dataset == 'imagenet' else 0
    augmented_dataset = AugmentedDataset(testset, args.target_network, args.target_size,
                                         [-20, 10, 0, 10, 20], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], [0.8, 0.9, 1.0, 1.1, 1.2], [False, True],     # TODO *********
                                         args.size, padding, norm=args.dataset,
                                         target_aug='default',
                                         num_sample=1, cutout=0)
    augmented_dataset.is_test = True
    img_o, imgs_t, img_c, _, _ = augmented_dataset[0]
    img_l, _ = testset_legacy[0]
    imgs_t[encoded_tta_default()].save('111img_ours.png', 'PNG')
    img_o.save('111img_orig.png', 'PNG')
    img_c.save('111img_center.png', 'PNG')
    img_l.save('111img_legacy.png', 'PNG')

    print('-------')

    augmented_dataset = AugmentedDataset(testset, args.target_network, args.target_size,
                                         [-20, 10, 0, 10, 20], [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3], [0.8, 0.9, 1.0, 1.1, 1.2], [False, True],
                                         args.size, padding, norm=args.dataset,
                                         target_aug='default',
                                         num_sample=4, cutout=16)
    augmented_dataset.is_test = True

    content = []
    idx_offset = 12000
    for idx in tqdm(range(10)):
        for _ in range(4):
            _, imgs_t, img_orig, losses, corrects = augmented_dataset[idx_offset+idx]
            softmins = softmax(-1 * losses / args.tau)

            row = [
                html_td.format(
                    bgcolor='#ffffff',
                    base64img=pil_to_base64(img_orig),
                    loss='-', t='-', softmin='-'
                )
            ]
            t_idx = 0
            for img_t, loss, softmin, correct in zip(imgs_t, losses, softmins, corrects):
                td = html_td.format(
                    bgcolor='#FF0000' if correct == 0 else '#ffffff',
                    base64img=pil_to_base64(img_t),
                    loss='%.3f' % loss, t=decode_desc(t_idx),
                    softmin='%.3f' % softmin
                )
                row.append(td)
                t_idx += 1
            row = '<tr>%s</tr>' % (''.join(row))
            content.append(row)
    html = html_body.format(content=''.join(content))
    with open('dump.html', 'w') as f:
        f.write(html)
