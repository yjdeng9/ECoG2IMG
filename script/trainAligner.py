

import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import clip
from transformers import CLIPVisionModelWithProjection
from dalle2_pytorch import DiffusionPrior
from dalle2_pytorch.dalle2_pytorch import l2norm, default, exists
from dalle2_pytorch.train_configs import DiffusionPriorNetworkConfig
from dalle2_pytorch.dalle2_pytorch import RotaryEmbedding, CausalTransformer, SinusoidalPosEmb, MLP, Rearrange, repeat, rearrange, prob_mask_like, LayerNorm, RelPosBias, Attention, FeedForward


sys.path.append('/public8/lilab/student/yjdeng/brainIO/script/dataIntegration/')
from trainTaMAE_2d import MaskedAutoencoderViT_ECoG as MaskedAutoencoderViT
from dataLoader import single_IFP

class FlaggedCausalTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        norm_in = False,
        norm_out = True,
        attn_dropout = 0.,
        ff_dropout = 0.,
        final_proj = True,
        normformer = False,
        rotary_emb = True,
        causal=True
    ):
        super().__init__()
        self.init_norm = LayerNorm(dim) if norm_in else nn.Identity() # from latest BLOOM model and Yandex's YaLM

        self.rel_pos_bias = RelPosBias(heads = heads)

        rotary_emb = RotaryEmbedding(dim = min(32, dim_head)) if rotary_emb else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim = dim, causal = causal, dim_head = dim_head, heads = heads, dropout = attn_dropout, rotary_emb = rotary_emb),
                FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout, post_activation_norm = normformer)
            ]))

        self.norm = LayerNorm(dim, stable = True) if norm_out else nn.Identity()  # unclear in paper whether they projected after the classic layer norm for the final denoised image embedding, or just had the transformer output it directly: plan on offering both options
        self.project_out = nn.Linear(dim, dim, bias = False) if final_proj else nn.Identity()

    def forward(self, x):
        n, device = x.shape[1], x.device

        x = self.init_norm(x)

        attn_bias = self.rel_pos_bias(n, n + 1, device = device)

        for attn, ff in self.layers:
            x = attn(x, attn_bias = attn_bias) + x
            x = ff(x) + x

        out = self.norm(x)
        return self.project_out(out)

class VersatileDiffusionPriorNetwork(nn.Module):
    def __init__(
            self,
            dim,
            num_timesteps=None,
            num_time_embeds=1,
            # num_image_embeds = 1,
            # num_brain_embeds = 1,
            num_tokens=257,
            causal=True,
            learned_query_mode='none',
            **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.num_time_embeds = num_time_embeds
        self.continuous_embedded_time = not exists(num_timesteps)
        self.learned_query_mode = learned_query_mode

        self.to_time_embeds = nn.Sequential(
            nn.Embedding(num_timesteps, dim * num_time_embeds) if exists(num_timesteps) else nn.Sequential(
                SinusoidalPosEmb(dim), MLP(dim, dim * num_time_embeds)),
            # also offer a continuous version of timestep embeddings, with a 2 layer MLP
            Rearrange('b (n d) -> b n d', n=num_time_embeds)
        )

        if self.learned_query_mode == 'token':
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim))
        if self.learned_query_mode == 'pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens, dim) * scale)
        if self.learned_query_mode == 'all_pos_emb':
            scale = dim ** -0.5
            self.learned_query = nn.Parameter(torch.randn(num_tokens * 2 + 1, dim) * scale)
        self.causal_transformer = FlaggedCausalTransformer(dim=dim, causal=causal, **kwargs)

        self.null_brain_embeds = nn.Parameter(torch.randn(num_tokens, dim))
        self.null_image_embed = nn.Parameter(torch.randn(num_tokens, dim))

        self.num_tokens = num_tokens
        self.self_cond = False

    def forward_with_cond_scale(
            self,
            *args,
            cond_scale=1.,
            **kwargs
    ):
        logits = self.forward(*args, **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, brain_cond_drop_prob=1., image_cond_drop_prob=1, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
            self,
            image_embed,
            diffusion_timesteps,
            *,
            self_cond=None,
            ecog_embed=None,
            text_embed=None,
            brain_cond_drop_prob=0.,
            text_cond_drop_prob=None,
            image_cond_drop_prob=0.
    ):
        if text_embed is not None:
            ecog_embed = text_embed
        if text_cond_drop_prob is not None:
            brain_cond_drop_prob = text_cond_drop_prob

        image_embed = image_embed.view(len(image_embed), -1, 768)
        ecog_embed = ecog_embed.view(len(ecog_embed), -1, 768)

        # nn.Linear(768, 512)
        # self.projector = nn.Sequential(nn.linear(512, 768), nn.ReLU(), nn.Linear(512, dim))

        batch, _, dim, device, dtype = *image_embed.shape, image_embed.device, image_embed.dtype
        # num_time_embeds, num_image_embeds, num_brain_embeds = self.num_time_embeds, self.num_image_embeds, self.num_brain_embeds

        # classifier free guidance masks
        brain_keep_mask = prob_mask_like((batch,), 1 - brain_cond_drop_prob, device=device)
        brain_keep_mask = rearrange(brain_keep_mask, 'b -> b 1 1')

        image_keep_mask = prob_mask_like((batch,), 1 - image_cond_drop_prob, device=device)
        image_keep_mask = rearrange(image_keep_mask, 'b -> b 1 1')

        # mask out brain embeddings with null brain embeddings

        # import pdb; pdb.set_trace()
        null_brain_embeds = self.null_brain_embeds.to(ecog_embed.dtype)
        ecog_embed = torch.where(
            brain_keep_mask,
            ecog_embed,
            null_brain_embeds[None]
        )

        # mask out image embeddings with null image embeddings
        null_image_embed = self.null_image_embed.to(image_embed.dtype)
        image_embed = torch.where(
            image_keep_mask,
            image_embed,
            null_image_embed[None]
        )

        # whether brain embedding is used for conditioning depends on whether brain encodings are available for attention (for classifier free guidance, even though it seems from the paper it was not used in the prior ddpm, as the objective is different)
        # but let's just do it right
        if self.continuous_embedded_time:
            # if continuous cast to flat, else keep int for indexing embeddings
            diffusion_timesteps = diffusion_timesteps.type(dtype)
        time_embed = self.to_time_embeds(diffusion_timesteps)

        if self.learned_query_mode == 'token':
            learned_queries = repeat(self.learned_query, 'n d -> b n d', b=batch)
        elif self.learned_query_mode == 'pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b=batch)
            image_embed = image_embed + pos_embs
            learned_queries = torch.empty((batch, 0, dim), device=ecog_embed.device)
        elif self.learned_query_mode == 'all_pos_emb':
            pos_embs = repeat(self.learned_query, 'n d -> b n d', b=batch)
            learned_queries = torch.empty((batch, 0, dim), device=ecog_embed.device)
        else:
            learned_queries = torch.empty((batch, 0, dim), device=ecog_embed.device)

        tokens = torch.cat((
            ecog_embed,  # 257
            time_embed,  # 1
            image_embed,  # 257
            learned_queries  # 257
        ), dim=-2)
        if self.learned_query_mode == 'all_pos_emb':
            tokens = tokens + pos_embs

        # attend
        tokens = self.causal_transformer(tokens)

        # get learned query, which should predict the image embedding (per DDPM timestep)
        pred_image_embed = tokens[..., -self.num_tokens:, :]

        return pred_image_embed

def get_img_embedding(imgs, preprocess,clip_model,image_encoder,device='cuda',
                      hidden_state=True,clamp_embs=False,norm_embs=True):

    clip_emb = [preprocess(Image.fromarray(np.uint8(img)).convert('RGB')) for img in imgs]
    clip_emb = torch.stack(clip_emb)
    clip_emb = clip_emb.to(device)
    # print("clip_emb", clip_emb.shape)
    # clip_emb = clip_emb.unsqueeze(0).to(device)

    if hidden_state:
        clip_emb = image_encoder(clip_emb)
        clip_emb = clip_emb.last_hidden_state
        clip_emb = image_encoder.vision_model.post_layernorm(clip_emb)
        clip_emb = image_encoder.visual_projection(clip_emb)
    else:
        clip_emb = clip_model.encode_image(clip_emb)

    if clamp_embs:
        clip_emb = torch.clamp(clip_emb, -1.5, 1.5)
    if norm_embs:
        if hidden_state:
            # normalize all tokens by cls token's norm
            embeds_pooled = clip_emb[:, 0:1]
            clip_emb = clip_emb / torch.norm(embeds_pooled, dim=-1, keepdim=True)
            # clip_emb = clip_emb / torch.norm(clip_emb[:, 0], dim=-1).reshape(-1, 1, 1)
        else:
            clip_emb = nn.functional.normalize(clip_emb, dim=-1)

    return clip_emb


class EcoGDiffusionPrior(DiffusionPrior):
    def __init__(self, seed=None, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.seed = seed

    @torch.no_grad()
    def p_sample(self, x, t, text_cond=None, self_cond=None, clip_denoised=True, cond_scale=1.,
                 generator=None):

        if generator is None:
            if self.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=x.device)
                generator.manual_seed(self.seed)

        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=t, text_cond=text_cond,
                                                                          self_cond=self_cond,
                                                                          clip_denoised=clip_denoised,
                                                                          cond_scale=cond_scale)
        if generator is None:
            noise = torch.randn_like(x)
        else:
            # noise = torch.randn_like(x)
            noise = torch.randn(x.size(), device=x.device, dtype=x.dtype, generator=generator)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        pred = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        return pred, x_start

    @torch.no_grad()
    def p_sample_loop_ddpm(self, shape, text_cond, cond_scale=1., generator=None):
        batch, device = shape[0], self.device

        if generator is None:
            if self.seed is None:
                generator = None
            else:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(self.seed)


        if generator is None:
            image_embed = torch.randn(shape, device=device)
        else:
            image_embed = torch.randn(shape, device=device, generator=generator)
        x_start = None  # for self-conditioning

        if self.init_image_embed_l2norm:
            image_embed = l2norm(image_embed) * self.image_embed_scale


        for i in tqdm(reversed(range(0, self.noise_scheduler.num_timesteps)), desc='sampling loop time step',
                      total=self.noise_scheduler.num_timesteps, disable=True):
            times = torch.full((batch,), i, device=device, dtype=torch.long)

            self_cond = x_start if self.net.self_cond else None
            image_embed, x_start = self.p_sample(image_embed, times, text_cond=text_cond, self_cond=self_cond,
                                                 cond_scale=cond_scale,
                                                 generator=generator)

        if self.sampling_final_clamp_l2norm and self.predict_x_start:
            image_embed = self.l2norm_clamp_embed(image_embed)

        return image_embed

class ECoGProjector(nn.Module):
    def __init__(self, input_nc,emb_dim=1024):
        super().__init__()

        self.projector = nn.Sequential(nn.Conv1d(input_nc, 257, kernel_size=1), nn.ReLU(),
                                       nn.Linear(emb_dim, 768), nn.ReLU())

    def forward(self, ecog_embed):
        ecog_embed = self.projector(ecog_embed)
        return ecog_embed


def main():
    subj_id = sys.argv[1]
    epochs = int(sys.argv[2])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ifp_dir = '../../data/ifp_raw/'
    img_dir = '../../data/images/'
    talairach_dir = '../../data/loc_data/'
    meta_path = '../../data/meta_info.csv'
    mae_pretrain_dir = 'out_results/mae_2d_pos_subj/'

    seed = 42
    cp_dir = '../../data/batch_out_2d/diffusion_out_seed_%d' % (seed)

    mae_pretrain_path = os.path.join(mae_pretrain_dir, '%s/model.pth' % subj_id)
    if not os.path.exists(mae_pretrain_path):
        print("MAE model not found for %s" % subj_id)
        return

    out_dir = 'batch_out_2d/diffusion_out_%s_seed_42' % subj_id
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    ## 00. set params
    clip_size = 768
    out_dim = clip_size
    depth = 6
    dim_head = 64
    heads = clip_size // 64  # heads * dim_head = 12 * 64 = 768

    guidance_scale = 3.5
    timesteps = 100

    meta_info = pd.read_csv(meta_path)
    C = int(meta_info[meta_info['subj_id'] == subj_id]['num_channels'].values[0])
    T = 120
    S1 = C//8

    talairach_pos = pd.read_csv(os.path.join(talairach_dir,'talairach_%s.csv' % subj_id), index_col=0)

    ## 01. load models
    image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").eval()
    image_encoder = image_encoder.to(device)
    clip_variant = 'ViT-L/14'
    clip_model, preprocess = clip.load(clip_variant, device=device)

    ecog_encoder = MaskedAutoencoderViT(time_len=T,in_chans=1, chans_len=C,
                                 patch_size=(S1,8), embed_dim=1024, depth=24, num_heads=16,
                                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                 mlp_ratio=4, pos_array = talairach_pos)
    ecog_encoder = ecog_encoder.to(device)

    ecog_encoder.load_state_dict(torch.load(mae_pretrain_path))

    num_patches = ecog_encoder.patch_embed.num_patches

    ecog_projector = ECoGProjector(input_nc=num_patches+1).to(device)

    prior_network = VersatileDiffusionPriorNetwork(
        dim=out_dim,
        depth=depth,
        dim_head=dim_head,
        heads=heads,
        causal=False,
        num_tokens=257,
        learned_query_mode="pos_emb"
    ).to(device)

    diffusion_prior = DiffusionPrior(net=prior_network,
            image_embed_dim=out_dim,
            condition_on_text_encodings=False,
            timesteps=timesteps,
            image_embed_scale=None).to(device)

    try:
        diffusion_prior.load_state_dict(torch.load(os.path.join(cp_dir, 'diffusion_prior.pth')))
        ecog_projector.load_state_dict(torch.load(os.path.join(cp_dir, 'ecog_projector.pth')))
    except:
        print("No pre-trained model found, training from scratch.")

    optimizer = torch.optim.Adam([{'params': diffusion_prior.parameters(), 'lr': 1e-3},
                                    {'params': ecog_projector.parameters(), 'lr': 1e-3}], lr=1e-3)

    ## 02. load data

    x_index = meta_info.index.values
    y = meta_info['category'].values
    train_index, test_index, train_y, test_y = \
        train_test_split(x_index, y, stratify=y, test_size=0.2, random_state=seed)
    train_meta_info = meta_info.iloc[train_index]
    test_meta_info = meta_info.iloc[test_index]

    train_dataset = single_IFP(subj_id=subj_id, ifp_dir=ifp_dir, img_dir=img_dir,
                         meta_info=train_meta_info, time_len=T, response_type=1)
    train_dataloader = DataLoader(train_dataset, batch_size=6, shuffle=True)
    test_dataset = single_IFP(subj_id=subj_id, ifp_dir=ifp_dir, img_dir=img_dir,
                              meta_info=test_meta_info, time_len=T, response_type=1)
    test_dataloader = DataLoader(test_dataset, batch_size=6, shuffle=True)
    neg_dataset = single_IFP(subj_id=subj_id, ifp_dir=ifp_dir, img_dir=img_dir,
                             meta_info=meta_info, time_len=T, response_type=0)
    neg_dataloader = DataLoader(neg_dataset, batch_size=6, shuffle=True)

    log_df = pd.DataFrame(columns=['epoch', 'batch', 'loss'])

    ## start training
    for ep in range(epochs):
        diffusion_prior.net.train()
        ecog_projector.train()
        ecog_encoder.eval()

        for i, data in enumerate(train_dataloader):
            signal, img, label, category_name, rotation, response = data
            # print(i, signal.size(), img.size(), label.size(), category_name, rotation, response)

            signal = signal.float().to(device)

            ecog_embedding = ecog_encoder.ecog_encoder(signal)
            ecog_embedding = ecog_embedding.to(device)
            ecog_embedding = ecog_projector(ecog_embedding)

            img_embeedding = get_img_embedding(img, preprocess,clip_model,image_encoder).float().to(device)
            loss_prior = diffusion_prior(text_embed=ecog_embedding, image_embed=img_embeedding)

            optimizer.zero_grad()
            loss_prior.backward()
            optimizer.step()

            print("epoch: {}, batch: {}, loss: {}".format(ep, i, loss_prior.item()))
            log_df = log_df.append({'epoch': ep, 'batch': i, 'loss': loss_prior.item()}, ignore_index=True)

    log_df.to_csv(os.path.join(out_dir, 'log.csv'), index=False)

    save_model = True
    if save_model:
        torch.save(diffusion_prior.state_dict(), os.path.join(out_dir, 'diffusion_prior.pth'))
        torch.save(ecog_projector.state_dict(), os.path.join(out_dir, 'ecog_projector.pth'))


if __name__ == '__main__':
    main()
