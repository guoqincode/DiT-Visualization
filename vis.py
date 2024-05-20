import torch
import os
# import ipdb
from diffusers import PixArtAlphaPipeline
from models.transformer_2d import Transformer2DModel
from models.ptp_utils import register_attention_control, AttentionStore
from models.ptp_utils import get_self_attention_map, save_attention_map_as_image
import copy
from sklearn.decomposition import PCA
from torchvision import transforms as T
from math import sqrt
from PIL import Image
import numpy as np

# from diffusers.models.transformers.transformer_2d import Transformer2DModel
# torch2.x diffusers==0.26.3

def visualize_and_save_features_pca(feature_maps_fit_data,feature_maps_transform_data, transform_experiments, t, save_dir):
    feature_maps_fit_data = feature_maps_fit_data.cpu().numpy()
    pca = PCA(n_components=3)
    pca.fit(feature_maps_fit_data)
    feature_maps_pca = pca.transform(feature_maps_transform_data.cpu().numpy())  # N X 3
    feature_maps_pca = feature_maps_pca.reshape(len(transform_experiments), -1, 3)  # B x (H * W) x 3
    for i, experiment in enumerate(transform_experiments):
        pca_img = feature_maps_pca[i]  # (H * W) x 3
        h = w = int(sqrt(pca_img.shape[0]))
        pca_img = pca_img.reshape(h, w, 3)
        pca_img_min = pca_img.min(axis=(0, 1))
        pca_img_max = pca_img.max(axis=(0, 1))
        pca_img = (pca_img - pca_img_min) / (pca_img_max - pca_img_min)
        pca_img = Image.fromarray((pca_img * 255).astype(np.uint8))
        pca_img = T.Resize(512, interpolation=T.InterpolationMode.NEAREST)(pca_img)
        pca_img.save(os.path.join(save_dir, f"{experiment}_layer_{t}.png"))


generator = torch.Generator("cuda").manual_seed(1024)
pixart_transformer = Transformer2DModel.from_pretrained("PixArt-alpha/PixArt-XL-2-512x512", subfolder="transformer",torch_dtype=torch.float16,)
pipe = PixArtAlphaPipeline.from_pretrained(
    "PixArt-alpha/PixArt-XL-2-512x512", 
    transformer = pixart_transformer,
    torch_dtype=torch.float16)
pipe = pipe.to("cuda")

controller = AttentionStore()
register_attention_control(pipe,controller)

prompt = "An astronaut riding a horse."
negative_prompt = 'worst quality, low quality, bad anatomy, watermark, text, blurry'

images = pipe(prompt=prompt,negative_prompt=negative_prompt,height=512,width=512).images[0]

for i in range(28):
    attn_map = get_self_attention_map(controller,256,i,False)
    transform_attn_maps = copy.deepcopy(attn_map)
    visualize_and_save_features_pca(
            torch.cat([attn_map], dim=0),
            torch.cat([transform_attn_maps], dim=0),
            ['debug'],
            i,
            './self_attn_maps'
        )


images.save('generated_img.png')
