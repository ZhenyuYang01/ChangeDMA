import math
from functools import partial
from timm.layers import DropPath, trunc_normal_
from timm.layers.activations import *
from timm.layers import LayerNorm2d, RmsNorm2d


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_value: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return x * self.gamma.reshape(1, -1, 1, 1)


class Conv(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels)
        )


class ConvRelu(nn.Sequential):
    def __init__(self, *conv_args):
        super().__init__()
        self.add_module('conv', nn.Conv2d(*conv_args))
        self.add_module('relu', nn.ReLU())
        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ConvNormAct(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False,
                 skip=False, norm_layer='bn_2d', act_layer='relu', inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = get_norm(norm_layer)(dim_out)
        self.act = get_act(act_layer)(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


def get_norm(norm_layer='in_1d'):
    eps = 1e-6
    norm_dict = {
        'none': nn.Identity,
        'in_1d': partial(nn.InstanceNorm1d, eps=eps),
        'in_2d': partial(nn.InstanceNorm2d, eps=eps),
        'in_3d': partial(nn.InstanceNorm3d, eps=eps),
        'bn_1d': partial(nn.BatchNorm1d, eps=eps),
        'bn_2d': partial(nn.BatchNorm2d, eps=eps),
        'bn_3d': partial(nn.BatchNorm3d, eps=eps),
        'gn': partial(nn.GroupNorm, eps=eps),
        'ln_1d': partial(nn.LayerNorm, eps=eps),
        'ln_2d': partial(LayerNorm2d, eps=eps),
    }
    return norm_dict[norm_layer]


def get_act(act_layer='relu'):
    act_dict = {
        'none': nn.Identity,
        'sigmoid': Sigmoid,
        'swish': Swish,
        'mish': Mish,
        'hsigmoid': HardSigmoid,
        'hswish': HardSwish,
        'hmish': HardMish,
        'tanh': Tanh,
        'relu': nn.ReLU,
        'relu6': nn.ReLU6,
        'prelu': PReLU,
        'gelu': GELU,
        'silu': nn.SiLU
    }
    return act_dict[act_layer]


def adaptive_avg_pool1d(x, out_len: int):
    B, C, L = x.shape
    device = x.device

    i = torch.arange(out_len, device=device)
    start = torch.floor(i * L / out_len).to(torch.long)
    end = torch.ceil((i + 1) * L / out_len).to(torch.long)
    end = torch.clamp(end, max=L)

    prefix = torch.zeros((B, C, L + 1), device=device, dtype=x.dtype)
    prefix[..., 1:] = x.cumsum(dim=-1)

    seg_sum = prefix[..., end] - prefix[..., start]
    seg_len = (end - start).clamp(min=1).to(x.dtype)
    return seg_sum / seg_len.view(1, 1, -1)


class GLU(nn.Module):
    def __init__(self, in_features, hidden_features, drop=0., norm=True):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, padding=1, groups=hidden_features)
        self.act = nn.GELU()
        if norm:
            self.norm_layer = RmsNorm2d(hidden_features, eps=1e-6)
        else:
            self.norm_layer = nn.Identity()
        self.fc2 = nn.Conv2d(hidden_features, in_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x)) * v
        x = self.norm_layer(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, dim_in, dim_out, drop=0.):
        super().__init__()
        self.conv = ConvNormAct(dim_in, dim_in, kernel_size=1)
        self.conv_local = ConvNormAct(dim_in, dim_in, kernel_size=9, groups=dim_in, act_layer='silu')
        self.proj = ConvNormAct(dim_in, dim_out, kernel_size=1)
        self.proj_drop = nn.Dropout(drop)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_local(x)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = self.upsample(x)
        return x