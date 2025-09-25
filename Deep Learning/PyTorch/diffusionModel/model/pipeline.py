import torch
import numpy as np
from tqdm import tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENT_WIDTH =  512 // 8
LATENT_HEIGHT =  512 // 8

def generate(prompt: str,
             uncond_prompt: str,
             input_image=None,
             strength=0.8,
             do_cfg=True,
             cfg_scale=7.5,
             sampler_name="ddpm",
             n_infernece_steps=50,
             models={},
             seed=None,
             device=None,
             idel_device=None,
             tokenizer=None
             ):
    with torch.inference_mode():
        if not (0 < strength <=1):
            raise ValueError("strength must be between 0 and 1")
        if idel_device:
            to_idle: lambda x: x.to(idel_device)
        else:
            to_idle: lambda x: x
        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed()
        clip = models["clip"]
        clip = clip.to(device)
        if do_cfg:
            # convert the promt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encoder_plus([prompt], padding="max_length", max_length=77).input_ids
            # (Batch_size, Seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_size, Seq_len) -> (Batch_size, Seq_len, dim)
            cond_context = clip(cond_tokens)
            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_token = torch.tensor(uncond_token, dtype=torch.long, device=device)
            # (Batch_size, Seq_len) -> (Batch_size, Seq_len, dim)
            uncond_context = clip (uncond_token)
            # (2, Seq_len, dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else:
            # convert it into list of tokens
            tokens = tokenizer.batch_encoder_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, Seq_len, dim) = (1, 77, 768)
            context = clip(tokens)
        to_idle(clip)
        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_infernece_steps)
        else:
            raise ValueError(f"Unknow")
