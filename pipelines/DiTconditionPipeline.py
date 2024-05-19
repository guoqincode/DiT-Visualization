import inspect
import re
import urllib.parse as ul
from typing import Callable, List, Optional, Tuple, Union
from typing import Any, Dict

from diffusers.models import AutoencoderKL, Transformer2DModel
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.schedulers import DPMSolverMultistepScheduler
import torch
import torch.nn.functional as F
from transformers import T5EncoderModel, T5Tokenizer

from models.transformer_2d import Transformer2DModel
# from DiTcondition.models.transformer_2d import DiTcondition as Transformer2DModel
from diffusers.pipelines import PixArtAlphaPipeline
from diffusers.pipelines.pixart_alpha.pipeline_pixart_alpha import ASPECT_RATIO_1024_BIN, ASPECT_RATIO_512_BIN, retrieve_timesteps
from diffusers.models.attention import GatedSelfAttentionDense
from copy import deepcopy

class DiTconditionPipeline(PixArtAlphaPipeline):
    def __init__(
        self, 
        tokenizer: T5Tokenizer, 
        text_encoder: T5EncoderModel, 
        vae: AutoencoderKL, 
        transformer: Transformer2DModel, 
        scheduler: DPMSolverMultistepScheduler):
        super().__init__(tokenizer, text_encoder, vae, transformer, scheduler)


    @torch.no_grad()
    def __call__(
        self, 
        prompt: Union[str, List[str]] = None, 
        negative_prompt: str = "", 
        num_inference_steps: int = 20, 
        timesteps: List[int] = None, 
        guidance_scale: float = 4.5, 
        num_images_per_prompt: Optional[int] = 1,
        height: Optional[int] = None,
        width: Optional[int] = None,
        eta: float = 0, 
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None, 
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True, 
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1, 
        clean_caption: bool = True, 
        use_resolution_binning: bool = True, 
        **kwargs
    ) -> Union[ImagePipelineOutput, Tuple]:
        return super().__call__(
            prompt, 
            negative_prompt, 
            num_inference_steps, 
            timesteps, 
            guidance_scale, 
            num_images_per_prompt, 
            height, 
            width, 
            eta, 
            generator, 
            latents, 
            prompt_embeds, 
            prompt_attention_mask, 
            negative_prompt_embeds, 
            negative_prompt_attention_mask, 
            output_type, 
            return_dict, 
            callback, 
            callback_steps, 
            clean_caption, 
            use_resolution_binning, 
            **kwargs)