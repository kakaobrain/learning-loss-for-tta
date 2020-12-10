# Learning Loss for Test-Time Augmentation

We release most of the code for researchers interested in our work. Reuse of this code is permitted for non-commercial research.

Some of our internal code has been removed, but it contains the main algorithms proposed in the paper. We believe it will be helpful for your reference.

This is not an official Kakao(Brain) product.

## Dependencies

- requirements.txt
- opencv-python >= 4.2.0.34
- apt-get install libmagickwand-dev

## Run

### Train

To train the loss predictor,

```bash
$ python train.py -c confs/imagenet.yaml
```

For the data processing remote-workers,

```bash
$ python remote_dataloader/worker.py --server task1:{1958 - 1962}
```

### Evaluation

```bash
$ python eval_l2t.py --augmentor ...pth --multicrop {1, 5, 10} --corrupt {corrupt_type}:{corrupt_level} ...
```

{corrupt_type} can be any corruption name proposed in ImageNet-c. It defined five levels of corruption, so {corrupt_level} should be a value between 1 and 5.