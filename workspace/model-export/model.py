# This file is not used by the tracking application and currently outdated
import torch
import torch.nn as nn
import geffnet.mobilenetv3
from geffnet.config import set_layer_config
from geffnet.efficientnet_builder import round_channels, decode_arch_def, resolve_act_layer, resolve_bn_args
from geffnet.activations import get_act_fn, get_act_layer

class DSConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernels_per_layer=4, groups=1, old=0):
        super(DSConv2d, self).__init__()
        if old == 2:
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * kernels_per_layer, kernel_size=3, padding=1, groups=in_planes),
                nn.Conv2d(in_planes * kernels_per_layer, out_planes, kernel_size=1, groups=groups)
            )
        elif old == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * kernels_per_layer, kernel_size=3, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes * kernels_per_layer),
                nn.Conv2d(in_planes * kernels_per_layer, out_planes, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_planes, in_planes * kernels_per_layer, kernel_size=3, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes * kernels_per_layer),
                nn.ReLU6(inplace=True),
                nn.Conv2d(in_planes * kernels_per_layer, out_planes, kernel_size=1, groups=groups, bias=False),
                nn.BatchNorm2d(out_planes),
                nn.ReLU6(inplace=True)
            )
    def forward(self, x):
        x = self.conv(x)
        return x

class UNetUp(nn.Module):
    def __init__(self, in_channels, residual_in_channels, out_channels, size, old=0):
        super(UNetUp, self).__init__()
        self.up = nn.Upsample(size=size, mode='bilinear', align_corners=True)
        self.conv = DSConv2d(in_channels + residual_in_channels, out_channels, 1, 1, old=old)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# This is the gaze tracking model
class OpenSeeFaceGaze(geffnet.mobilenetv3.MobileNetV3):
    def __init__(self):
        kwargs = get_mobilenetv3_kwargs('small')
        super(OpenSeeFaceGaze, self).__init__(**kwargs)
        self.up1 = UNetUp(576, 48, 64, (2,2), old=2)
        self.up2 = UNetUp(64, 24, 32, (4,4), old=2)
        self.up3 = UNetUp(32, 16, 15, (8,8), old=2)
        self.group = DSConv2d(15, 3, kernels_per_layer=4, groups=3, old=2)
    def _forward_impl(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        r1 = None
        r2 = None
        r3 = None
        for i, feature in enumerate(self.blocks):
            x = feature(x)
            if i == 3:
                r3 = x
            if i == 1:
                r2 = x
            if i == 0:
                r1 = x
        x = self.up1(x, r3)
        x = self.up2(x, r2)
        x = self.up3(x, r1)
        x = self.group(x)
        return x
    def forward(self, x):
        return self._forward_impl(x)

# This is the face detection model. Because the landmark model is very robust, it gets away with predicting very rough bounding boxes. It is fully convolutional and can be made to run on different resolutions. It was trained on 224x224 crops and the most reasonable results can be found in the range of 224x224 to 640x640.
class OpenSeeFaceDetect(geffnet.mobilenetv3.MobileNetV3):
    def __init__(self, size="large", channel_multiplier=0.1):
        kwargs = get_mobilenetv3_kwargs(size, channel_multiplier)
        super(OpenSeeFaceDetect, self).__init__(**kwargs)
        if size == "large":
            self.up1 = UNetUp(round_channels(960, channel_multiplier), round_channels(112, channel_multiplier), 256, (14,14), old=1)
            self.up2 = UNetUp(256, round_channels(40, channel_multiplier), 128, (28,28), old=1)
            self.up3 = UNetUp(128, round_channels(24, channel_multiplier), 64, (56,56), old=1)
            self.group = DSConv2d(64, 2, kernels_per_layer=4, groups=2, old=1)
            self.r1_i = 1
            self.r2_i = 2
            self.r3_i = 4
        elif size == "small":
            self.up1 = UNetUp(round_channels(576, channel_multiplier), round_channels(40, channel_multiplier), 256, (14,14), old=1)
            self.up2 = UNetUp(256, round_channels(24, channel_multiplier), 128, (28,28), old=1)
            self.up3 = UNetUp(128, round_channels(16, channel_multiplier), 64, (56,56), old=1)
            self.group = DSConv2d(64, 2, kernels_per_layer=4, groups=2, old=1)
            self.r1_i = 0
            self.r2_i = 1
            self.r3_i = 2
        self.maxpool = nn.MaxPool2d(kernel_size=3, dilation=1, stride=1, padding=1)
    def _forward_impl(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        r2 = None
        r3 = None
        for i, feature in enumerate(self.blocks):
            x = feature(x)
            if i == self.r3_i:
                r3 = x
            if i == self.r2_i:
                r2 = x
            if i == self.r1_i:
                r1 = x
        x = self.up1(x, r3)
        x = self.up2(x, r2)
        x = self.up3(x, r1)
        x = self.group(x)
        x2 = self.maxpool(x)
        return x, x2
    def forward(self, x):
        return self._forward_impl(x)

def logit_arr(p, factor=16.0):
    p = p.clamp(0.0000001, 0.9999999)
    return torch.log(p / (1 - p)) / factor

# Landmark detection model
# Models:
# 0: "small", 0.5
# 1: "small", 1.0
# 2: "large", 0.75
# 3: "large", 1.0
class OpenSeeFaceLandmarks(geffnet.mobilenetv3.MobileNetV3):
    def __init__(self, size="large", channel_multiplier=1.0, inference=False):
        kwargs = get_mobilenetv3_kwargs(size, channel_multiplier)
        super(OpenSeeFaceLandmarks, self).__init__(**kwargs)
        if size == "large":
            self.up1 = UNetUp(round_channels(960, channel_multiplier), round_channels(112, channel_multiplier), 256, (14,14))
            self.up2 = UNetUp(256, round_channels(40, channel_multiplier), 198 * 1, (28,28))
            self.group = DSConv2d(198 * 1, 198, kernels_per_layer=4, groups=3)
            self.r2_i = 2
            self.r3_i = 4
        elif size == "small":
            self.up1 = UNetUp(round_channels(576, channel_multiplier), round_channels(40, channel_multiplier), 256, (14,14))
            self.up2 = UNetUp(256, round_channels(24, channel_multiplier), 198 * 1, (28,28))
            self.group = DSConv2d(198 * 1, 198, kernels_per_layer=4, groups=3)
            self.r2_i = 1
            self.r3_i = 2
        self.inference = inference
    def _forward_impl(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        r2 = None
        r3 = None
        for i, feature in enumerate(self.blocks):
            x = feature(x)
            if i == self.r3_i:
                r3 = x
            if i == self.r2_i:
                r2 = x
        x = self.up1(x, r3)
        x = self.up2(x, r2)
        x = self.group(x)

        if self.inference:
            t_main = x[:, 0:66].reshape((-1, 66, 28*28))
            t_m = t_main.argmax(dim=2)
            indices = t_m.unsqueeze(2)
            t_conf = t_main.gather(2, indices).squeeze(2)
            t_off_x = x[:, 66:132].reshape((-1, 66, 28*28)).gather(2, indices).squeeze(2)
            t_off_y = x[:, 132:198].reshape((-1, 66, 28*28)).gather(2, indices).squeeze(2)
            t_off_x = (223. * logit_arr(t_off_x) + 0.5).floor()
            t_off_y = (223. * logit_arr(t_off_y) + 0.5).floor()
            t_x = 223. * (t_m / 28.).floor() / 27. + t_off_x
            t_y = 223. * t_m.remainder(28.).float() / 27. + t_off_y
            x = (t_conf.mean(1), torch.stack([t_x, t_y, t_conf], 2))

        return x
    def forward(self, x):
        return self._forward_impl(x)

def get_mobilenetv3_kwargs(variant, channel_multiplier = 1.0):
    if 'small' in variant:
        num_features = 1024
        if 'minimal' in variant:
            act_layer = 'relu'
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16'],
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24', 'ir_r1_k3_s1_e3.67_c24'],
                # stage 2, 28x28 in
                ['ir_r1_k3_s2_e4_c40', 'ir_r2_k3_s1_e6_c40'],
                # stage 3, 14x14 in
                ['ir_r2_k3_s1_e3_c48'],
                # stage 4, 14x14in
                ['ir_r3_k3_s2_e6_c96'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],
            ]
        else:
            act_layer = 'hard_swish'
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s2_e1_c16_se0.25_nre'],  # relu
                # stage 1, 56x56 in
                ['ir_r1_k3_s2_e4.5_c24_nre', 'ir_r1_k3_s1_e3.67_c24_nre'],  # relu
                # stage 2, 28x28 in
                ['ir_r1_k5_s2_e4_c40_se0.25', 'ir_r2_k5_s1_e6_c40_se0.25'],  # hard-swish
                # stage 3, 14x14 in
                ['ir_r2_k5_s1_e3_c48_se0.25'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r3_k5_s2_e6_c96_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c576'],  # hard-swish
            ]
    else:
        num_features = 1280
        if 'minimal' in variant:
            act_layer = 'relu'
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16'],
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24', 'ir_r1_k3_s1_e3_c24'],
                # stage 2, 56x56 in
                ['ir_r3_k3_s2_e3_c40'],
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112'],
                # stage 5, 14x14in
                ['ir_r3_k3_s2_e6_c160'],
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],
            ]
        else:
            act_layer = 'hard_swish'
            arch_def = [
                # stage 0, 112x112 in
                ['ds_r1_k3_s1_e1_c16_nre'],  # relu
                # stage 1, 112x112 in
                ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
                # stage 2, 56x56 in
                ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
                # stage 3, 28x28 in
                ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
                # stage 4, 14x14in
                ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
                # stage 5, 14x14in
                ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
                # stage 6, 7x7 in
                ['cn_r1_k1_s1_c960'],  # hard-swish
            ]

    return dict(
        block_args=decode_arch_def(arch_def),
        num_features=num_features,
        stem_size=16,
        channel_multiplier=channel_multiplier,
        act_layer=resolve_act_layer({}, act_layer),
        se_kwargs=dict(
            act_layer=get_act_layer('relu'), gate_fn=get_act_fn('hard_sigmoid'), reduce_mid=True, divisor=8),
        norm_kwargs={}
    )

# Checkpoint test
if __name__== "__main__":
    '''
    set_layer_config(
        scriptable=False,
        exportable=False,
        no_jit=False
    )
    '''
    
    print("Checking gaze model")
    m=OpenSeeFaceGaze()
    ckpt = torch.load("weights/gaze.pth")
    m.load_state_dict(ckpt)
    print("Checking detection model")
    m=OpenSeeFaceDetect()
    ckpt = torch.load("weights/detection.pth", map_location=torch.device('cpu'))
    m.load_state_dict(ckpt)
    print("Checking lm_model0 model")
    m=OpenSeeFaceLandmarks("small", 0.5)
    ckpt = torch.load("weights/lm_model0.pth", map_location=torch.device('cpu'))
    m.load_state_dict(ckpt)
    print("Checking lm_model1 model")
    m=OpenSeeFaceLandmarks("small", 1.0)
    ckpt = torch.load("weights/lm_model1.pth", map_location=torch.device('cpu'))
    m.load_state_dict(ckpt)
    print("Checking lm_model2 model")
    m=OpenSeeFaceLandmarks("large", 0.75)
    ckpt = torch.load("weights/lm_model2.pth", map_location=torch.device('cpu'))
    m.load_state_dict(ckpt)
    print("Checking lm_model3 model")
    m=OpenSeeFaceLandmarks("large", 1.0)
    ckpt = torch.load("weights/lm_model3.pth", map_location=torch.device('cpu'))
    m.load_state_dict(ckpt)
