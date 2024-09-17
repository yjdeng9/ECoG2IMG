
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch import randn as randn_tensor
from torch.utils.data import DataLoader
from dalle2_pytorch import DiffusionPrior
from diffusers import VersatileDiffusionPipeline
from transformers import CLIPTokenizer, CLIPImageProcessor, CLIPTextModelWithProjection,CLIPVisionModelWithProjection


from dataLoader import single_IFP
from trainTaMAE_2d import MaskedAutoencoderViT_ECoG as MaskedAutoencoderViT
from trainAligner import VersatileDiffusionPriorNetwork,ECoGProjector

@torch.no_grad()
def _encode_prompt(image_embeddings, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt_embeds=None):
    batch_size = image_embeddings.shape[0]

    # duplicate image embeddings for each generation per prompt, using mps friendly method
    bs_embed, seq_len, _ = image_embeddings.shape
    image_embeddings = image_embeddings.repeat(1, num_images_per_prompt, 1)
    image_embeddings = image_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)

    # get unconditional embeddings for classifier free guidance
    if do_classifier_free_guidance:
        if negative_prompt_embeds is None:
            negative_prompt_embeds = get_negative_prompt_embed(batch_size)

        # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
        seq_len = negative_prompt_embeds.shape[1]
        negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
        negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and conditional embeddings into a single batch
        # to avoid doing two forward passes
        image_embeddings = torch.cat([negative_prompt_embeds, image_embeddings])
    image_embeddings = image_embeddings.to(device)
    return image_embeddings


@torch.no_grad()
def get_negative_prompt_embed(batch_size=1,image_feature_extractor=None,image_encoder=None,device='cuda'):
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if image_feature_extractor is None:
        image_feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    if image_encoder is None:
        image_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)

    def normalize_embeddings(encoder_output):
        embeds = image_encoder.vision_model.post_layernorm(encoder_output.last_hidden_state)
        embeds = image_encoder.visual_projection(embeds)
        embeds_pooled = embeds[:, 0:1]
        embeds = embeds / torch.norm(embeds_pooled, dim=-1, keepdim=True)
        return embeds

    uncond_images = [np.zeros((512, 512, 3)) + 0.5] * batch_size
    uncond_images = image_feature_extractor(images=uncond_images, return_tensors="pt")
    pixel_values = uncond_images.pixel_values.to(device).to(image_encoder.dtype)
    # print('input shape', uncond_images.input_shape)
    negative_prompt_embeds = image_encoder(pixel_values)
    negative_prompt_embeds = normalize_embeddings(negative_prompt_embeds)

    # print('negative_prompt_embeds.shape', negative_prompt_embeds.shape)
    # print(negative_prompt_embeds.sum(axis=[1,2]))
    return negative_prompt_embeds


@torch.no_grad()
def vd_img2img(image_embeddings, vae, unet, scheduler ,device='', verbose=False ,guidance_scale: float = 7.5,
               num_images_per_prompt: int = 1, negative_prompt_embeds=None, num_inference_steps: int = 50 ,):
    # 0. Default height and width to unet
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    height = unet.config.sample_size * vae_scale_factor
    width = unet.config.sample_size * vae_scale_factor

    generator = torch.Generator(device=device).manual_seed(0)

    # 1. Check inputs. Raise error if not correct
    # check_inputs(image, height, width, callback_steps)

    # 2. Define call parameters
    batch_size = image_embeddings.shape[0]
    # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
    # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
    # corresponds to doing no classifier free guidance.
    do_classifier_free_guidance = guidance_scale > 1.0

    embeds_pooled = image_embeddings[:, 0:1]
    image_embeddings = image_embeddings / torch.norm(embeds_pooled, dim=-1, keepdim=True)
    # 3. Encode input prompt
    image_embeddings = _encode_prompt(
        image_embeddings, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt_embeds=negative_prompt_embeds
    )

    # np.save('image_embeddings.npy', image_embeddings.detach().cpu().numpy())



    # 4. Prepare timesteps
    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps

    # 5. Prepare latent variables
    num_channels_latents = unet.config.in_channels

    shape = \
    (batch_size * num_images_per_prompt, num_channels_latents, height // vae_scale_factor, width // vae_scale_factor)
    latents = randn_tensor(shape, generator=generator, device=device, dtype=unet.dtype)
    latents = latents * scheduler.init_noise_sigma
    latents = latents.to(device)

    # 6. Prepare extra step kwargs.

    # 7. Denoising loop
    for i, t in enumerate(timesteps):
        # expand the latents if we are doing classifier free guidance
        latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
        latent_model_input = scheduler.scale_model_input(latent_model_input, t)
        latent_model_input = latent_model_input.to(device)

        # predict the noise residual
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=image_embeddings).sample

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

    image = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]

    # image = image_processor.postprocess(image, output_type=output_type)
    # @staticmethod
    # def denormalize(images):
    #     """
    #     Denormalize an image array to [0,1].
    #     """
    #     return (images / 2 + 0.5).clamp(0, 1)

    return (image / 2 + 0.5).clamp(0, 1)


def main():
    subj_id = sys.argv[1]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    ifp_dir = '../../data/ifp_raw/'
    img_dir = '../../data/images/'
    talairach_dir = '../../data/loc_data/'
    meta_path = '../../data/meta_info.csv'
    mae_pretrain_dir = 'out_results/mae_2d_pos_subj/'
    vd_cache_dir = '../../data/pretrain/versatile-diffusion'

    seed = 42
    cp_dir = '../../data/batch_out_2d/diffusion_out_seed_%d' % (seed)

    out_dir = '../../data/batch_out_2d/recon_out_%s_seed_%d' % (subj_id, seed)
    show_dir = '../../data/batch_out_2d/recon_show_%s_seed_%d' % (subj_id, seed)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if not os.path.exists(show_dir):
        os.makedirs(show_dir)

    mae_pretrain_path = os.path.join(mae_pretrain_dir, '%s/model.pth' % subj_id)
    projector_pretrain_path = os.path.join(cp_dir, '%s/ecog_projector.pth' % subj_id)
    diffusion_prior_path = os.path.join(cp_dir, '%s/diffusion_prior.pth' % subj_id)

    show_num = 20

    clip_size = 768
    out_dim = clip_size
    depth = 6
    dim_head = 64
    heads = clip_size // 64  # heads * dim_head = 12 * 64 = 768

    guidance_scale = 3.5
    timesteps = 100
    timesteps_prior = 100

    ## Load data
    meta_info = pd.read_csv(meta_path)
    C = int(meta_info[meta_info['subj_id'] == subj_id]['num_channels'].values[0])
    T = 120
    S1 = C // 8

    talairach_pos = pd.read_csv(os.path.join(talairach_dir, 'talairach_%s.csv' % subj_id), index_col=0)

    x_index = meta_info.index.values
    y = meta_info['category'].values
    train_index, test_index, train_y, test_y = \
        train_test_split(x_index, y, stratify=y, test_size=0.2, random_state=42)
    train_meta_info = meta_info.iloc[train_index]
    test_meta_info = meta_info.iloc[test_index]

    train_dataset = single_IFP(subj_id=subj_id, ifp_dir=ifp_dir, img_dir=img_dir,
                               meta_info=train_meta_info, time_len=T, response_type=1)
    test_dataset = single_IFP(subj_id=subj_id, ifp_dir=ifp_dir, img_dir=img_dir,
                              meta_info=test_meta_info, time_len=T, response_type=1)
    neg_dataset = single_IFP(subj_id=subj_id, ifp_dir=ifp_dir, img_dir=img_dir,
                             meta_info=meta_info, time_len=T, response_type=0)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    neg_dataloader = DataLoader(neg_dataset, batch_size=1, shuffle=False)

    ## load model
    vd_pipe = VersatileDiffusionPipeline.from_pretrained(vd_cache_dir).to(device)
    unet = vd_pipe.image_unet
    vae = vd_pipe.vae
    scheduler = vd_pipe.scheduler

    ecog_encoder = MaskedAutoencoderViT(time_len=T,in_chans=1, chans_len=C,
                                 patch_size=(S1,8), embed_dim=1024, depth=24, num_heads=16,
                                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                                 mlp_ratio=4, pos_array = talairach_pos).to(device)
    num_patches = ecog_encoder.patch_embed.num_patches

    ecog_projector = ECoGProjector(input_nc=num_patches + 1).to(device)

    ecog_encoder.load_state_dict(torch.load(mae_pretrain_path))
    ecog_projector.load_state_dict(torch.load(projector_pretrain_path))

    prior_network = VersatileDiffusionPriorNetwork(dim=out_dim, depth=depth, dim_head=dim_head,
       heads=heads,causal=False, num_tokens=257, learned_query_mode="pos_emb" ).to(device)

    diffusion_prior = DiffusionPrior(net=prior_network,
       image_embed_dim=out_dim, condition_on_text_encodings=False, timesteps=timesteps,
       cond_drop_prob=0.2, image_embed_scale=None).to(device)

    diffusion_prior.load_state_dict(torch.load(diffusion_prior_path))

    clip_feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_encoder = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-large-patch14").to(device)

    ## start recon

    modes = ['train', 'test', 'negative']

    for mode in modes:
        if mode == 'train':
            dataloader = train_dataloader
        elif mode == 'test':
            dataloader = test_dataloader
        elif mode == 'negative':
            dataloader = neg_dataloader

        raw_imgs = []
        recon_imgs = []
        labels = []
        for i, data in enumerate(dataloader):
            signal, raw_img, label, category_name, rotation, response = data
            print(i, signal.size(), raw_img.size(), label.size(), category_name, rotation, response)

            signal = signal.float().to(device)

            ecog_embedding = ecog_encoder.ecog_encoder(signal)
            ecog_embedding = ecog_projector(ecog_embedding)

            brain_embeddings = diffusion_prior.p_sample_loop(ecog_embedding.shape,
                                                             text_cond=dict(text_embed=ecog_embedding),
                                                             cond_scale=1., timesteps=timesteps_prior)
            negative_prompt_embeds = get_negative_prompt_embed(batch_size=brain_embeddings.shape[0],
                                                               image_feature_extractor=clip_feature_extractor,
                                                               image_encoder=clip_encoder, device=device)

            img = vd_img2img(brain_embeddings, vae, unet, scheduler, device=device,
                             guidance_scale=guidance_scale, negative_prompt_embeds=negative_prompt_embeds)
            raw_imgs.append(raw_img.detach().cpu().numpy())
            recon_imgs.append(img.detach().cpu().numpy())
            labels.append(label.item())

            if i < show_num:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(raw_img[0, :, :].detach().cpu().numpy(), cmap='gray')
                plt.axis('off')
                out_img = img[0, :, :, :].detach().cpu().numpy()

                plt.subplot(1, 2, 2)
                plt.imshow(out_img.transpose(1, 2, 0), cmap='gray')
                plt.axis('off')

                plt.savefig(os.path.join(show_dir, 'img_recon_imgs_{}_{}.png'.format(i, category_name[0])))
                plt.close()

        raw_imgs = np.concatenate(raw_imgs, axis=0)
        recon_imgs = np.concatenate(recon_imgs, axis=0)
        print('recon_imgs.shape', recon_imgs.shape)
        np.save(os.path.join(out_dir, '%s_raw_imgs.npy'%mode), raw_imgs)
        np.save(os.path.join(out_dir, '%s_recon_imgs.npy'%mode), recon_imgs)
        np.save(os.path.join(out_dir, '%s_labels.npy'%mode), np.array(labels))


if __name__ == '__main__':
    main()
