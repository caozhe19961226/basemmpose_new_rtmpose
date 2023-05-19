import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, caffe2_xavier_init
from torch.utils.checkpoint import checkpoint
from mmdet.models.backbones.ditehrnet import DynamicSplitConvolution,DiteHRModule
from ..builder import NECKS


@NECKS.register_module()
class HRFPN(nn.Module):
    """HRFPN (High Resolution Feature Pyrmamids)

    arXiv: https://arxiv.org/abs/1904.04514

    Args:
        in_channels (list): number of channels for each branch.
        out_channels (int): output channels of feature pyramids.
        num_outs (int): number of output stages.
        pooling_type (str): pooling for generating feature pyramids
            from {MAX, AVG}.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp  (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        stride (int): stride of 3x3 convolutional layers
    """
    # neck=dict(
    #     _delete_=True,
    #     type='HRFPN',
    #     in_channels=[48, 96, 192, 384],
    #     out_channels=channels,  输出通道是256
    #     stride=2,
    #     num_outs=3, #输出3级
    #     norm_cfg=norm_cfg),
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs=5,
                 pooling_type='AVG',
                 conv_cfg=None,
                 norm_cfg=None,
                 with_cp=False,
                 stride=1):
        super(HRFPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.reduction_conv = ConvModule(
            sum(in_channels),
            out_channels,
            kernel_size=1,
            conv_cfg=self.conv_cfg,
            act_cfg=None)

        self.fpn_convs = nn.ModuleList()
        for i in range(self.num_outs):
            print(self.num_outs)
            self.fpn_convs.append(
                #(self, channels, stride, num_branch, num_groups, num_kernels, with_cp=False)
                # DynamicSplitConvolution(
                #     channels=out_channels,
                #     stride=stride,
                #     num_branch=2,
                #     num_groups=[1, 1, 2, 4],
                #     num_kernels=[4, 4, 2, 1],
                #     with_cp=with_cp
                # ),
                # DiteHRModule(
                #     num_branches=4,
                #     num_blocks=2,
                #     with_fuse=True,
                #     in_channels=[48, 96, 192, 384],
                #     multiscale_output=out_channels,
                #     conv_cfg=self.conv_cfg,
                #     norm_cfg=self.norm_cfg,
                #     with_cp=self.with_cp)
                ConvModule(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    stride=stride,
                    conv_cfg=self.conv_cfg,
                    act_cfg=None)
            )

        if pooling_type == 'MAX':
            self.pooling = F.max_pool2d
        else:
            self.pooling = F.avg_pool2d

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                caffe2_xavier_init(m)

    def forward(self, inputs):
        # 检查输入和实例数是否一一致
        assert len(inputs) == self.num_ins

        outs = [inputs[0]]
        #print('1',len(outs))
        for i in range(1, self.num_ins):
            outs.append(
                #功能：利用插值方法，对输入的张量数组进行上\下采样操作，换句话说就是科学合理地改变数组的尺寸大小，尽量保持数据完整。
                F.interpolate(inputs[i], scale_factor=2**i, mode='bilinear'))
        out = torch.cat(outs, dim=1)
       # print('2',out.size())
        if out.requires_grad and self.with_cp:
            out = checkpoint(self.reduction_conv, out)
        else:
            out = self.reduction_conv(out)
        outs = [out]
       # print('3',outs[0].size())
        for i in range(1, self.num_outs):
            outs.append(self.pooling(out, kernel_size=2**i, stride=2**i))
        outputs = []
       # print('4',outs[0].size(),outs[1].size(),outs[2].size(),len(outs))
        for i in range(self.num_outs):
            if outs[i].requires_grad and self.with_cp:
                tmp_out = checkpoint(self.fpn_convs[i], outs[i])
            else:
                tmp_out = self.fpn_convs[i](outs[i])
            outputs.append(tmp_out)
       # print("fanhui",outputs[0].size(),outputs[1].size(),outputs[2].size(),len(outputs))
        return tuple(outputs)
