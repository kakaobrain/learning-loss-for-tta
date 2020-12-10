import torch
from torch import nn
from torch.nn import functional as F

from .utils import (
    relu_fn,
    round_filters,
    round_repeats,
    drop_connect,
    get_same_padding_conv2d,
    get_model_params,
    efficientnet_params,
    load_pretrained_weights,
    drop_connect2)
from ..shakedrop import ShakeDropFunction


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            if global_params.norm_type == 'batch':
                self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            elif global_params.norm_type == 'instance':
                self._bn0 = nn.InstanceNorm2d(num_features=oup, momentum=self._bn_mom, affine=True)
            else:
                raise

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        if global_params.norm_type == 'batch':
            self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        else:
            self._bn1 = nn.InstanceNorm2d(num_features=oup, momentum=self._bn_mom, affine=True)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        if global_params.norm_type == 'batch':
            self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        else:
            self._bn2 = nn.InstanceNorm2d(num_features=final_oup, momentum=self._bn_mom, affine=True)

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = relu_fn(self._bn0(self._expand_conv(inputs)))
        x = relu_fn(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(relu_fn(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None):
        super().__init__()
        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        if global_params.norm_type == 'batch':
            self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        elif global_params.norm_type == 'instance':
            self._bn0 = nn.InstanceNorm2d(num_features=out_channels, momentum=bn_mom, affine=True)
        else:
            raise

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            # print(len(self._blocks), block_args.stride)
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        if global_params.norm_type == 'batch':
            self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        else:
            self._bn1 = nn.InstanceNorm2d(num_features=out_channels, momentum=bn_mom, affine=True)

        # print('EfficientNet Final ConvHead', in_channels, out_channels)

        # Final linear layer
        self._dropout = self._global_params.dropout_rate
        if not global_params.multiple_feat:
            self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        else:
            self._fc = nn.Linear(1616, self._global_params.num_classes)     # TODO
            # self._fc = nn.Linear(1400, self._global_params.num_classes)

        self.multiple_feat = global_params.multiple_feat
        if self.multiple_feat:
            self.branch_fc = nn.ModuleList([nn.Linear(_ch, _ch) for _ch in [24, 40, 80, 192]])

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = relu_fn(self._bn0(self._conv_stem(inputs)))

        # Blocks
        feats = []
        extract = [1, 3, 5, 11]
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if self.multiple_feat and idx in extract:
                x_flatten = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
                x_flatten = relu_fn(self.branch_fc[extract.index(idx)](x_flatten))
                x_flatten = drop_connect2(x_flatten, self._global_params.drop_connect_rate, self.training)
                # x_flatten = ShakeDropFunction.apply(x_flatten, self.training, self._global_params.drop_connect_rate, [-1, 1])
                feats.append(x_flatten)

        # Head
        x = relu_fn(self._bn1(self._conv_head(x)))
        x = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)

        if self.multiple_feat:
            feats.append(x)
            x = torch.cat(feats, dim=1)

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """

        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        if self._dropout:
            x = F.dropout(x, p=self._dropout, training=self.training)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, num_classes=1000, multiple_feat=False, norm_type='batch'):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params={'num_classes': num_classes, 'multiple_feat': multiple_feat, 'norm_type': norm_type})
        model = cls(blocks_args, global_params)

        import os
        if os.environ.get('pretrain', '1') and (multiple_feat is True or 'b0' in model_name):
            print('-------------------------------- use pretrained weights --------------------------------')
            load_pretrained_weights(model, 'efficientnet-b0', load_fc=False)
        return model

    @classmethod
    def from_pretrained(cls, model_name, num_classes=1000, multiple_feat=False):
        model = cls.from_name(model_name, num_classes=num_classes, multiple_feat=multiple_feat)
        if not multiple_feat:       # TODO
            load_pretrained_weights(model, model_name, load_fc=(num_classes == 1000))
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet-b'+str(i) for i in range(num_models)] + ['efficientnet-l2t']
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))
