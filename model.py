import math
from os.path import exists
from inspect import isfunction
import torch
from functools import partial
from einops import rearrange, reduce
from einops.layers.torch import Rearrange

import utils
import settings


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=2, mode="nearest"),
        torch.nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return torch.nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        torch.nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv2d(torch.nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, "o ... -> o 1 1 1", "mean")
        var = reduce(weight, "o ... -> o 1 1 1", partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return torch.nn.functional.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = torch.nn.GroupNorm(groups, dim_out)
        self.act = torch.nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(torch.nn.Module):
    """https://arxiv.org/abs/1512.03385"""
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = torch.nn.Conv2d(dim, dim_out, 1) if dim != dim_out else torch.nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class Attention(torch.nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(torch.nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = torch.nn.Sequential(torch.nn.Conv2d(hidden_dim, dim, 1), torch.nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)
        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)


class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet(torch.nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        mask_classes=settings.MASK_CLASSES,
        self_condition=True,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.mask_classes = mask_classes
        self.self_condition = self_condition
        # input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = torch.nn.Conv2d(channels, init_dim, 1, padding=0)  # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        #
        # dims = [i * 2 for i in dims]
        #
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        # Self condition blocks
        # Initial
        self.self_condition_conv_init = torch.nn.Conv2d(self.self_condition, init_dim, 3, padding=1)
        # Middle
        self.self_condition_conv_mid = torch.nn.Conv2d(init_dim, init_dim * 8, 3, stride=4, padding=1)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            torch.nn.Linear(dim, time_dim),
            torch.nn.GELU(),
            torch.nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        self.ups_y = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                torch.nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else torch.nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        # in_out = [(128, 128), (128, 256), (256, 1024)]
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            if ind == 0:
                dim_out *= 2
            is_last = ind == (len(in_out) - 1)
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        block_klass(dim_in + dim_out, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_in + dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else torch.nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups_y.append(
                torch.nn.ModuleList(
                    [
                        block_klass(dim_in + dim_out, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_in + dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else torch.nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = torch.nn.Conv2d(dim * 2, self.out_dim, 1)

    def forward(self, x, time, y):
        x = self.init_conv(x)
        r_x = x.clone()
        y = self.self_condition_conv_init(y)
        r_y = y.clone()

        # x = torch.cat((x, y), dim=1)
        t = self.time_mlp(time)
        h = []
        h_y = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            y = block1(y, t)
            h.append(x)
            h_y.append(y)
            x = block2(x, t)
            x = attn(x)
            y = block2(y, t)
            y = attn(y)
            h.append(x)
            h_y.append(y)
            x = downsample(x)
            y = downsample(y)

        # y = self.self_condition_conv_mid(y)
        y = self.mid_block1(y, t)
        y = self.mid_attn(y)
        y = self.mid_block2(y, t)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        x = torch.cat((x, y), dim=1)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h_y.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h_y.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)

        for block1, block2, attn, upsample in self.ups_y:
            y = torch.cat((y, h.pop()), dim=1)
            y = block1(y, t)
            y = torch.cat((y, h.pop()), dim=1)
            y = block2(y, t)
            y = attn(y)
            y = upsample(y)

        y = torch.cat((y, r_y), dim=1)
        y = self.final_res_block(y, t)
        x = torch.cat((x, r_x), dim=1)
        x = self.final_res_block(x, t)

        x = torch.cat((x, y), dim=1)
        return self.final_conv(x)


if __name__ == '__main__':
    model = Unet(
        dim=settings.IMAGE_SIZE,
        channels=settings.CHANNELS,
        dim_mults=(1, 2, 4)
    )
    image_test = torch.rand(1, settings.CHANNELS, settings.IMAGE_SIZE, settings.IMAGE_SIZE)
    mask_test = torch.rand(1, settings.MASK_CLASSES, settings.IMAGE_SIZE, settings.IMAGE_SIZE)
    t_test = torch.randint(settings.TIME_STEPS, size=(1, ))
    test_predict = model(image_test, t_test, mask_test)
    print(f'Predict Shape: {list(test_predict.shape)}')
    print('Num params: ', sum(p.numel() for p in model.parameters()))
