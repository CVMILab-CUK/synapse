import inspect
import importlib
import warnings
from typing import Callable, List, Optional, Union

import torch
from k_diffusion.external import CompVisDenoiser, CompVisVDenoiser


from diffusers import DiffusionPipeline, LMSDiscreteScheduler, StableDiffusionMixin

from diffusers.image_processor import VaeImageProcessor
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.utils import logging

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def retrieve_timesteps(
    scheduler,
    num_inference_steps: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
    timesteps: Optional[List[int]] = None,
    sigmas: Optional[List[float]] = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`List[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`List[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps



class ModelWrapper:
    def __init__(self, model, alphas_cumprod):
        self.model = model
        self.alphas_cumprod = alphas_cumprod

    def apply_model(self, *args, **kwargs):
        if len(args) == 3:
            encoder_hidden_states = args[-1]
            args = args[:2]
        if kwargs.get("cond", None) is not None:
            encoder_hidden_states = kwargs.pop("cond")
        return self.model(*args, encoder_hidden_states=encoder_hidden_states, **kwargs).sample


class EEGDiffusionPipeline(DiffusionPipeline, StableDiffusionMixin):
    r"""
    Pipeline for eeg-to-image generation using Stable Diffusion.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.

        eeg_encoder ([`EEG_Model Encode`]):
           
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.

        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latents. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].

        feature_extractor ([`CLIPImageProcessor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    """

    # _optional_components = ["safety_checker", "feature_extractor"]/

    def __init__(
        self,
        vae,
        cond_models,
        unet,
        scheduler,
        ip_adaption_modules=None,
    ):
        super().__init__()
        # get correct sigmas from LMS
        # scheduler = LMSDiscreteScheduler.from_config(scheduler.config)
        self.register_modules(
            vae=vae,
            cond_models=cond_models,
            unet=unet,
            scheduler=scheduler,
            ip_adaption_modules=ip_adaption_modules,
        )

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (?) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to ? in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def set_sampler(self, scheduler_type: str):
        warnings.warn("The `set_sampler` method is deprecated, please use `set_scheduler` instead.")
        return self.set_scheduler(scheduler_type)

    def set_scheduler(self, scheduler_type: str):
        library = importlib.import_module("k_diffusion")
        sampling = getattr(library, "sampling")
        self.sampler = getattr(sampling, scheduler_type)
    
    def prepare_latents(self, batch_size, num_channels_latents, height, width, dtype, device, generator, latents=None):
        shape = (batch_size, num_channels_latents, height // 8, width // 8)
        if latents is None:
            if device.type == "mps":
                # randn does not work reproducibly on mps
                latents = torch.randn(shape, generator=generator, device="cpu", dtype=dtype).to(device)
            else:
                latents = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        else:
            if latents.shape != shape:
                raise ValueError(f"Unexpected latents shape, got {latents.shape}, expected {shape}")
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        return latents

    def decode_latents(self, latents, generator=None, return_type:str="torch"):
        assert return_type in ["torch", "numpy", "pil"], f"please check your return type Now : {return_type}"
        image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[0] # make image
        # image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1) # denormalize
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
        if return_type == "numpy":
            image = image.cpu().permute(0, 2, 3, 1).float().numpy()
        elif return_type == "pil":
            image = self.numpy_to_pil(image.cpu().permute(0, 2, 3, 1).float().numpy())
    
        return image
    


    @property
    def interrupt(self):
        return self._interrupt

    @torch.no_grad()
    def __call__(
        self,
        eeg: List[float],
        height: int = 512,
        width: int = 512,
        num_inference_steps: int = 250,
        timesteps: List[int] = None,
        guidance_scale: float = 7.5,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        sigmas: List[float] = None,
        latents: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
        **kwargs,
    ):
        r"""
        """
        batch_size = len(eeg)
        device = self.vae.device
        self._interrupt = False

        # Encode Embedding
        eeg_embeddings = self.cond_models(eeg.to(device))
        eeg_embeddings = eeg_embeddings.repeat_interleave(num_images_per_prompt, dim=0) # For make multiple images

        # Unconditional Embeddings for CFG
        uncond_tokens = torch.zeros_like(eeg_embeddings)
        
        # Concat For Unet
        eeg_embeddings = torch.cat([uncond_tokens, eeg_embeddings])

        # Ready Time Step Scheduler
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, timesteps, sigmas
        )

        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            eeg_embeddings.dtype,
            device,
            generator,
            latents,
        )


        if self.ip_adaption_modules is not None:
            eeg_adaption_embeddings = eeg_embeddings.mean(dim=1)
            eeg_adaption_embeddings = self.ip_adaption_modules(eeg_adaption_embeddings)
            eeg_embeddings          = torch.cat([eeg_embeddings, eeg_adaption_embeddings], dim=1)
            # eeg_embeddings = self.ip_adaption_modules(eeg_embeddings)

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        self.loss = 0.0

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                # expand the latents if we are doing classifier free guidance
                # latent_model_input = latents
                # For CFG
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=eeg_embeddings,
                    return_dict=False,
                )[0]

                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        image = self.decode_latents(latents, generator, "torch")
        # print("Decoded Image:", image.shape, image.max(), image,min())
        has_nsfw_concept = None
        # do_denormalize = [True] * image.shape[0]
        # image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)
        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
        
        

