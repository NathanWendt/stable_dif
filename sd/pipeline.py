import torch
import numpy as np
import tqdm as tqdm
from ddpm import DDPMSampler

WIDTH = 512
HEIGHT = 512
LATENTS_WIDTH = WIDTH // 8
LATENTS_HEIGHT = HEIGHT // 8

def generate(prompt: str, 
             uncond_prompt: str, # negative prompt or empty
             input_image=None, # image to image
             strength=0.8, # Strength means more noise, lower strength, output more similar to input image
             do_cfg=True, # classifier free, 2 outputs (one with prompt one without)
             cfg_scale=7.5, #cfg_scale: 0-14, how much do we want the model to focus on prompt?
             sampler_name="ddpm", 
             n_inference_steps=50,
             models={}, 
             seed=None, 
             device=None, 
             idle_device=None, 
             tokenizer=None
            ):
    
    with torch.no_grad():
        # inference
        if not (0 < strength <= 1):
            raise ValueError("strength must be between 0 and 1")
        
        if idle_device:
            to_idle: lambda x: x.to(device)
        else:
            to_idle: lambda x: x

        generator = torch.Generator(device=device)
        if seed is None:
            generate.seed()
        else:
            generator.manual_seed(seed)

        clip = models["clip"]
        clip.to(device)

        if do_cfg:
            # Convert the prompt into tokens using the tokenizer
            cond_tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            # (Batch_size, Seq_len)
            cond_tokens = torch.tensor(cond_tokens, dtype=torch.long, device=device)
            # (Batch_size, Seq_len) -> (Batch_size, Seq_len, Dim)
            cond_context = clip(cond_tokens)

            uncond_tokens = tokenizer.batch_encode_plus([uncond_prompt], padding="max_length", max_length=77).input_ids
            uncond_tokens = torch.tensor(uncond_tokens, dtype=torch.long, device=device)
            # (Batch_size, Seq_len) -> (Batch_size, Seq_len, Dim)
            uncond_context = clip(uncond_context)

            # (2, Seq_len, Dim) = (2, 77, 768)
            context = torch.cat([cond_context, uncond_context])
        else: 
            # convert into a list of tokens
            tokens = tokenizer.batch_encode_plus([prompt], padding="max_length", max_length=77).input_ids
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            # (1, 77, 768)
            context = clip(tokens)
        to_idle(clip)

        if sampler_name == "ddpm":
            sampler = DDPMSampler(generator)
            sampler.set_inference_steps(n_inference_steps)
        else:
            ValueError("No tengo")

        latents_shape = (1, 4, LATENTS_HEIGHT, LATENTS_WIDTH)

        if input_image:
            encoder = models["encoder"]
            encoder.to(device)

            input_image_tensor = input_image.resize((WIDTH, HEIGHT))
            input_image_tensor = np.array(input_image_tensor)
            # (Height, Width, Channel)
            input_image_tensor = torch.tensor(input_image_tensor, dtype=torch.float32)
            #rescale (0,255)->(-1,1)
            input_image_tensor = rescale(input_image_tensor, (0, 255), (-1, 1))
            # (Height, Width, Channel) -> (Batch_size, Height, Width, Channel)
            input_image_tensor = input_image_tensor.unsqueeze(0)
            # (Batch_size, Height, Width, Channel) -> (Batch_size, Channel, Height, Width)
            input_image_tensor = input_image_tensor.permute(0, 3, 1, 2)

            encoder_noise = torch.randn(latents_shape, generator=generator, device=device)
            # Input -> VAE encoder
            latents = encoder(input_image_tensor, encoder_noise)

            sampler.set_strength(strength=strength)
            latents = sampler.add_noise(latents, sampler.timesteps[0])

            to_idle(encoder)
        else: 
            # no ipnut image, use random noise N(0,1)
            latents = torch.randn(latents_shape, generator=generator, device=device)


        # Linear steps 1000, 999, ..., 0
        # n_inference_steps=50, 1000, 980, 960, ..., 0
        diffusion = models["diffusion"]
        diffusion.to(device)

        timesteps = tqdm(sampler.timesteps)
        for i, timestep in enumerate(timesteps):
            # (1, 320)
            time_embedding = get_time_embedding(timestep).to(device)
            
            # (Batch_size, 4, Latents_height, Latents_width) latents_height,width = 64,64
            model_input = latents

            if do_cfg:
                # (Batch_size, 4, Latents_height, Latents_width) -> # (2*Batch_size, 4, Latents_height, Latents_width)
                model_input = model_input.repeat(2, 1, 1, 1)
            
            # predicted noise by UNET
            model_output = diffusion(model_input, context, time_embedding)

            if do_cfg:
                output_cond, output_uncond = model_output.chunk(2)
                model_output = cfg_scale * (output_cond - output_uncond) + output_uncond

            # remove predicted noise by UNET
            latents = sampler.step(timestep, latents, model_output)
        
        to_idle(diffusion)

        decoder = models["decoder"]
        decoder.to(device)

        images = decoder(latents)
        to_idle(decoder)

        images = rescale(images, (-1, 1), (0, 255), clamp=True)
        # (Batch_size, 4, Latents_height, Latents_width) -> (Batch_size, 4, Latents_height, Latents_width) 
        images = images.permute(0, 2, 3, 1)
        images = images.to("cpu", torch.uint8).numpy()
        return images[0]

    
def rescale(x, old_range, new_range, clamp=False):
    old_min, old_max = old_range
    new_min, new_max = new_range
    x -= old_min
    x *= (new_max - new_min) / (old_max - old_min)
    x += new_min
    if clamp:
        x = x.clamp(new_min, new_max)
    
def get_time_embedding(timestep):
    # positional encoding
    # (160,)
    freqs = torch.pow(10000, -torch.arange(start=0, end=160, dtype=torch.float32) / 160)
    # (1, 160)
    x = torch.tensor([timestep], dtype=torch.float32)[:, None] * freqs[None]
    # (1, 320)
    return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)






        