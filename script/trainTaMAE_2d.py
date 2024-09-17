


import sys
import os
from functools import partial
import numpy as np
import pandas as pd

from enum import Enum
from itertools import repeat
from typing import Callable, Optional

import torch
from torch import nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm.models.vision_transformer import Block


from dataLoader import single_IFP


def get_1d_sincos_pos_embed_from_pos(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.array(pos)
    pos = pos.reshape(-1)  # (M,)

    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_talairach_pos_embed(pos_array, embed_dim, patch_size):
    ta_embed_dim = (embed_dim // 4) // patch_size
    N = pos_array.shape[0]
    emb_x = get_1d_sincos_pos_embed_from_pos(ta_embed_dim // 4, pos_array['x'])
    emb_y = get_1d_sincos_pos_embed_from_pos(ta_embed_dim // 4, pos_array['y'])
    emb_z = get_1d_sincos_pos_embed_from_pos(ta_embed_dim // 4, pos_array['z'])
    emb_region = get_1d_sincos_pos_embed_from_pos(ta_embed_dim // 4, pos_array['region_code'])

    emb_data = np.concatenate([emb_x, emb_y, emb_z, emb_region], axis=1)  # (N, D)

    emb_data = emb_data.reshape(N//patch_size,-1)

    return emb_data


def get_2d_sincos_pos_embed(embed_dim, grid_size_h,grid_size_w,
                            talairach_pos,patch_size,cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """

    ta_embed_data = get_talairach_pos_embed(talairach_pos, embed_dim, patch_size)
    ta_embed_data = ta_embed_data.repeat(grid_size_h, axis=0)

    grid_h = np.arange(grid_size_h, dtype=np.float32)
    grid_w = np.arange(grid_size_w, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size_h, grid_size_w])

    print('ta_embed_data',ta_embed_data.shape)
    print('grid',grid.shape)
    print('embed_dim',embed_dim)

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid, ta_embed_data)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid,ta_embed_data):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 4, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, ta_embed_data, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Format(str, Enum):
    NCHW = 'NCHW'
    NHWC = 'NHWC'
    NCL = 'NCL'
    NLC = 'NLC'

def nchw_to(x: torch.Tensor, fmt: Format):
    if fmt == Format.NHWC:
        x = x.permute(0, 2, 3, 1)
    elif fmt == Format.NLC:
        x = x.flatten(2).transpose(1, 2)
    elif fmt == Format.NCL:
        x = x.flatten(2)
    return x
class PatchEmbed_ECoG(nn.Module):
    output_fmt: Format
    dynamic_img_pad: torch.jit.Final[bool]

    def __init__(
            self,
            time_len: Optional[int] = 100,chans_len: Optional[int] = 88,
            patch_size: int = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True,
            dynamic_img_pad: bool = False,
    ):
        super().__init__()
        if type(patch_size) is int:
            self.patch_size = tuple(repeat(patch_size, 2))
        if time_len is not None:
            self.img_size = tuple([time_len,chans_len])
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
            self.output_fmt = Format(output_fmt)
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
            self.output_fmt = Format.NCHW
        self.strict_img_size = strict_img_size
        self.dynamic_img_pad = dynamic_img_pad

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.dynamic_img_pad:
            pad_h = (self.patch_size[0] - H % self.patch_size[0]) % self.patch_size[0]
            pad_w = (self.patch_size[1] - W % self.patch_size[1]) % self.patch_size[1]
            x = F.pad(x, (0, pad_w, 0, pad_h))
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        elif self.output_fmt != Format.NCHW:
            x = nchw_to(x, self.output_fmt)
        x = self.norm(x)
        return x


class MaskedAutoencoderViT_ECoG(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, time_len=224, chans_len=88, patch_size=16, in_chans=1,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,pos_array=None):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics

        self.pos_array = pos_array
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed_ECoG(time_len,chans_len, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,  norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.grid_size[0]),
                                            int(self.patch_embed.grid_size[1]),
                                            self.pos_array,
                                            self.patch_size,
                                            cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.grid_size[0]),
                                                    int(self.patch_embed.grid_size[1]),
                                                    self.pos_array,
                                                    self.patch_size,
                                                    cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0

        h = imgs.shape[2] // p
        w = imgs.shape[3] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        # h = w = int(x.shape[1] ** .5)
        h = int(self.patch_embed.grid_size[0])
        w = int(self.patch_embed.grid_size[1])
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, w * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        raw_x = x.clone()
        # print('x.shape', x.shape)
        # print('self.pos_embed.shape', self.pos_embed.shape)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # print('x.shape before mask', x.shape)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # print('x.shape', x.shape)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore,raw_x

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        # mse = nn.MSELoss(reduction='none')

        loss = (pred - target) ** 2

        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

        return loss

    def forward(self, imgs, mask_ratio=0.75):
        print('imgs.shape', imgs.shape)
        latent, mask, ids_restore,raw_x = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, raw_x

    def ecog_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 150

    ## load GT data
    subj_id = sys.argv[1]
    main_out_dir = 'out_results/mae_2d_pos_subj/'

    ifp_dir = '../../data/ifp_raw/'
    img_dir = '../../data/images/'
    talairach_dir = '../../data/loc_data/'
    meta_path = '../../data/meta_info.csv'

    out_dir = os.path.join(main_out_dir, str(subj_id))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    meta_info = pd.read_csv(meta_path)

    T = 120
    C = int(np.mean(meta_info[meta_info['subj_id'] == subj_id]['num_channels'].values))
    S1 = C // 8

    talairach_pos = pd.read_csv(os.path.join(talairach_dir,'talairach_%s.csv' % subj_id), index_col=0)

    print('talairach_pos', talairach_pos.shape)
    print('num_channels', C)

    train_meta_info = meta_info

    dataset = single_IFP(subj_id=subj_id, ifp_dir=ifp_dir, img_dir=img_dir,
                         meta_info=train_meta_info, time_len=120)
    dataloader = DataLoader(dataset, batch_size=6, shuffle=True)


    ## load model
    model = MaskedAutoencoderViT_ECoG(time_len=T,in_chans=1, chans_len=C,
                                 patch_size=(S1,8), embed_dim=1024, depth=24, num_heads=16,
                                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                 mlp_ratio=4,pos_array = talairach_pos,
                                 norm_layer=partial(nn.LayerNorm, eps=1e-6))
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    log_df = pd.DataFrame(columns=['mode', 'epoch', 'batch', 'loss'])

    for ep in range(epochs):
        model.train()
        for i, data in enumerate(dataloader):
            signal, img, label, category_name, rotation, response = data
            # print(i, signal.size(), img.size(), label.size(), category_name, rotation, response)

            signal = signal.float().to(device)
            loss, pred, mask, raw_x = model(signal,mask_ratio=0.75)

            print('epoch %d, batch %d, loss %.4f' % (ep, i, loss.item()))
            log_df = log_df.append({'mode':'train', 'epoch': ep, 'batch': i, 'loss': loss.item()}, ignore_index=True)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    log_df.to_csv(os.path.join(out_dir, 'log.csv'), index=False)
    torch.save(model.state_dict(), os.path.join(out_dir, 'model.pth'))


if __name__ == '__main__':
    main()

