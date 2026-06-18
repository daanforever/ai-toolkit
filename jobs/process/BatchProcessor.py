import random
import torch

from toolkit.clip_vision_adapter import ClipVisionAdapter
from toolkit.image_utils import reduce_contrast
from toolkit.timestep_debug import TimestepDistributionLogger
from toolkit.timestep_sampler import TimestepSampler
from toolkit.train_tools import get_torch_dtype
from toolkit.util.blended_blur_noise import get_blended_blur_noise
from toolkit.util.debug import is_debug_enabled


class GeneralBatchProcessor:
    """
    Handles preparation of training batches including latents, noise, timesteps, and prompts.
    Extracted from BaseSDTrainProcess to improve code organization.
    """
    
    def __init__(self, process_instance):
        """
        Initialize the batch processor.
        
        Args:
            process_instance: Instance of BaseSDTrainProcess with all dependencies
        """
        self.p = process_instance
        self._timestep_sampler = None
        self._timestep_debug_logger = None
    
    def process(self, batch):
        """
        Process batch and return prepared training data.
        
        Args:
            batch: DataLoaderBatchDTO
            
        Returns:
            tuple: (noisy_latents, noise, timesteps, conditioned_prompts, imgs)
        """
        with torch.no_grad():
            conditioned_prompts, do_double = self._prepare_prompts(batch)
            latents, imgs, unaugmented_latents, dtype, is_reg = self._prepare_latents(batch)
            timesteps, min_noise_steps, max_noise_steps, do_double = self._prepare_timesteps(
                batch, latents, is_reg, do_double
            )
            noisy_latents, noise, timesteps, imgs = self._prepare_noise_and_noisy_latents(
                batch, latents, unaugmented_latents, timesteps, conditioned_prompts,
                imgs, dtype, is_reg, do_double, min_noise_steps, max_noise_steps
            )
        
        return noisy_latents, noise, timesteps, conditioned_prompts, imgs
    
    def _prepare_prompts(self, batch):
        """
        Prepare and condition prompts for training.
        
        Args:
            batch: DataLoaderBatchDTO
            
        Returns:
            tuple: (conditioned_prompts, do_double)
        """
        with self.p.timer('prepare_prompt'):
            prompts = batch.get_caption_list()
            is_reg_list = batch.get_is_reg_list()

            is_any_reg = any([is_reg for is_reg in is_reg_list])

            do_double = self.p.train_config.short_and_long_captions and not is_any_reg

            if self.p.train_config.short_and_long_captions and do_double:
                # dont do this with regs. No point

                # double batch and add short captions to the end
                prompts = prompts + batch.get_caption_short_list()
                is_reg_list = is_reg_list + is_reg_list
            if self.p.model_config.refiner_name_or_path is not None and self.p.train_config.train_unet:
                prompts = prompts + prompts
                is_reg_list = is_reg_list + is_reg_list

            conditioned_prompts = []

            for prompt, is_reg in zip(prompts, is_reg_list):

                # make sure the embedding is in the prompts
                if self.p.embedding is not None:
                    prompt = self.p.embedding.inject_embedding_to_prompt(
                        prompt,
                        expand_token=True,
                        add_if_not_present=not is_reg,
                    )

                if self.p.adapter and isinstance(self.p.adapter, ClipVisionAdapter):
                    prompt = self.p.adapter.inject_trigger_into_prompt(
                        prompt,
                        expand_token=True,
                        add_if_not_present=not is_reg,
                    )

                # make sure trigger is in the prompts if not a regularization run
                if self.p.trigger_word is not None:
                    prompt = self.p.sd.inject_trigger_into_prompt(
                        prompt,
                        trigger=self.p.trigger_word,
                        add_if_not_present=not is_reg,
                    )

                if not is_reg and self.p.train_config.prompt_saturation_chance > 0.0:
                    # do random prompt saturation by expanding the prompt to hit at least 77 tokens
                    if random.random() < self.p.train_config.prompt_saturation_chance:
                        est_num_tokens = len(prompt.split(' '))
                        if est_num_tokens < 77:
                            num_repeats = int(77 / est_num_tokens) + 1
                            prompt = ', '.join([prompt] * num_repeats)


                conditioned_prompts.append(prompt)
        
        return conditioned_prompts, do_double
    
    def _prepare_latents(self, batch):
        """
        Prepare latents from images with optional standardization.
        
        Args:
            batch: DataLoaderBatchDTO
            
        Returns:
            tuple: (latents, imgs, unaugmented_latents, dtype, is_reg)
        """
        with self.p.timer('prepare_latents'):
            dtype = get_torch_dtype(self.p.train_config.dtype)
            imgs = None
            is_reg = any(batch.get_is_reg_list())
            if batch.tensor is not None:
                imgs = batch.tensor
                imgs = imgs.to(self.p.device_torch, dtype=dtype)
                # dont adjust for regs.
                if self.p.train_config.img_multiplier is not None and not is_reg:
                    # do it ad contrast
                    imgs = reduce_contrast(imgs, self.p.train_config.img_multiplier)
            if batch.latents is not None:
                latents = batch.latents.to(self.p.device_torch, dtype=dtype)
                batch.latents = latents
            else:
                # normalize to
                if self.p.train_config.standardize_images:
                    if self.p.sd.is_xl or self.p.sd.is_vega or self.p.sd.is_ssd:
                        target_mean_list = [0.0002, -0.1034, -0.1879]
                        target_std_list = [0.5436, 0.5116, 0.5033]
                    else:
                        target_mean_list = [-0.0739, -0.1597, -0.2380]
                        target_std_list = [0.5623, 0.5295, 0.5347]
                    # Mean: tensor([-0.0739, -0.1597, -0.2380])
                    # Standard Deviation: tensor([0.5623, 0.5295, 0.5347])
                    imgs_channel_mean = imgs.mean(dim=(2, 3), keepdim=True)
                    imgs_channel_std = imgs.std(dim=(2, 3), keepdim=True)
                    imgs = (imgs - imgs_channel_mean) / imgs_channel_std
                    target_mean = torch.tensor(target_mean_list, device=self.p.device_torch, dtype=dtype)
                    target_std = torch.tensor(target_std_list, device=self.p.device_torch, dtype=dtype)
                    # expand them to match dim
                    target_mean = target_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                    target_std = target_std.unsqueeze(0).unsqueeze(2).unsqueeze(3)

                    imgs = imgs * target_std + target_mean
                    batch.tensor = imgs

                    # show_tensors(imgs, 'imgs')

                latents = self.p.sd.encode_images(imgs)
                batch.latents = latents

            if self.p.train_config.standardize_latents:
                if self.p.sd.is_xl or self.p.sd.is_vega or self.p.sd.is_ssd:
                    target_mean_list = [-0.1075, 0.0231, -0.0135, 0.2164]
                    target_std_list = [0.8979, 0.7505, 0.9150, 0.7451]
                else:
                    target_mean_list = [0.2949, -0.3188, 0.0807, 0.1929]
                    target_std_list = [0.8560, 0.9629, 0.7778, 0.6719]

                latents_channel_mean = latents.mean(dim=(2, 3), keepdim=True)
                latents_channel_std = latents.std(dim=(2, 3), keepdim=True)
                latents = (latents - latents_channel_mean) / latents_channel_std
                target_mean = torch.tensor(target_mean_list, device=self.p.device_torch, dtype=dtype)
                target_std = torch.tensor(target_std_list, device=self.p.device_torch, dtype=dtype)
                # expand them to match dim
                target_mean = target_mean.unsqueeze(0).unsqueeze(2).unsqueeze(3)
                target_std = target_std.unsqueeze(0).unsqueeze(2).unsqueeze(3)

                latents = latents * target_std + target_mean
                batch.latents = latents

                # show_latents(latents, self.p.sd.vae, 'latents')


            if batch.unconditional_tensor is not None and batch.unconditional_latents is None:
                unconditional_imgs = batch.unconditional_tensor
                unconditional_imgs = unconditional_imgs.to(self.p.device_torch, dtype=dtype)
                unconditional_latents = self.p.sd.encode_images(unconditional_imgs)
                batch.unconditional_latents = unconditional_latents * self.p.train_config.latent_multiplier

            unaugmented_latents = None
            if self.p.train_config.loss_target == 'differential_noise':
                # we determine noise from the differential of the latents
                unaugmented_latents = self.p.sd.encode_images(batch.unaugmented_tensor)
        
        return latents, imgs, unaugmented_latents, dtype, is_reg
    
    def _prepare_timesteps(self, batch, latents, is_reg, do_double):
        """
        Prepare scheduler and sample timesteps for training.
        
        Args:
            batch: DataLoaderBatchDTO
            latents: Prepared latents tensor
            is_reg: Whether this is a regularization batch
            do_double: Whether to double the batch
            
        Returns:
            tuple: (timesteps, min_noise_steps, max_noise_steps, do_double)
        """
        with self.p.timer('prepare_scheduler'):
            
            batch_size = len(batch.file_items)
            min_noise_steps = self.p.train_config.min_denoising_steps
            max_noise_steps = self.p.train_config.max_denoising_steps
            if self.p.model_config.refiner_name_or_path is not None:
                # if we are not training the unet, then we are only doing refiner and do not need to double up
                if self.p.train_config.train_unet:
                    max_noise_steps = round(self.p.train_config.max_denoising_steps * self.p.model_config.refiner_start_at)
                    do_double = True
                else:
                    min_noise_steps = round(self.p.train_config.max_denoising_steps * self.p.model_config.refiner_start_at)
                    do_double = False

            num_train_timesteps = self.p.train_config.num_train_timesteps

            if self.p.train_config.noise_scheduler in ['custom_lcm']:
                # we store this value on our custom one
                self.p.sd.noise_scheduler.set_timesteps(
                    self.p.sd.noise_scheduler.train_timesteps, device=self.p.device_torch
                )
            elif self.p.train_config.noise_scheduler in ['lcm']:
                self.p.sd.noise_scheduler.set_timesteps(
                    num_train_timesteps, device=self.p.device_torch, original_inference_steps=num_train_timesteps
                )
            elif self.p.train_config.noise_scheduler == 'flowmatch':
                linear_timesteps = any([
                    self.p.train_config.linear_timesteps,
                    self.p.train_config.linear_timesteps2,
                    self.p.train_config.timestep_type == 'linear',
                    self.p.train_config.timestep_type == 'one_step',
                ])
                
                timestep_type = 'linear' if linear_timesteps else None
                if timestep_type is None:
                    timestep_type = self.p.train_config.timestep_type
                
                if self.p.train_config.timestep_type == 'next_sample':
                    # simulate a sample
                    num_train_timesteps = self.p.train_config.next_sample_timesteps
                    timestep_type = 'shift'
                
                patch_size = 1
                if self.p.sd.is_flux or 'flex' in self.p.sd.arch or self.p.sd.arch == 'zimage_diffsynth':
                    # flux is a patch size of 1, but latents are divided by 2, so we need to double it
                    patch_size = 2
                elif hasattr(self.p.sd.unet.config, 'patch_size'):
                    patch_size = self.p.sd.unet.config.patch_size
                
                self.p.sd.noise_scheduler.set_train_timesteps(
                    num_train_timesteps,
                    device=self.p.device_torch,
                    timestep_type=timestep_type,
                    latents=latents,
                    patch_size=patch_size,
                )
            else:
                self.p.sd.noise_scheduler.set_timesteps(
                    num_train_timesteps, device=self.p.device_torch
                )
        if self.p.sd.is_multistage:
            with self.p.timer('adjust_multistage_timesteps'):
                # get our current sample range
                boundaries = [1] + self.p.sd.multistage_boundaries
                boundary_max, boundary_min = boundaries[self.p.current_boundary_index], boundaries[self.p.current_boundary_index + 1]
                asc_timesteps = torch.flip(self.p.sd.noise_scheduler.timesteps, dims=[0])
                lo = len(asc_timesteps) - torch.searchsorted(asc_timesteps, torch.tensor(boundary_max * 1000, device=asc_timesteps.device), right=False)
                hi = len(asc_timesteps) - torch.searchsorted(asc_timesteps, torch.tensor(boundary_min * 1000, device=asc_timesteps.device), right=True)
                first_idx = (lo - 1).item() if hi > lo else 0
                last_idx  = (hi - 1).item() if hi > lo else 999
                min_noise_steps = first_idx
                max_noise_steps = last_idx

        # clip min max indicies
        min_noise_steps = max(min_noise_steps, 0)
        max_noise_steps = min(max_noise_steps, num_train_timesteps - 1)
        
                
        with self.p.timer('prepare_timesteps_indices'):
            content_or_style = self.p.train_config.content_or_style
            if is_reg:
                content_or_style = self.p.train_config.content_or_style_reg

            if self._timestep_sampler is None:
                self._timestep_sampler = TimestepSampler(
                    self.p.train_config, self.p.sd.noise_scheduler
                )
            result = self._timestep_sampler.sample(
                batch_size=batch_size,
                latents=latents,
                content_or_style=content_or_style,
                min_noise_steps=min_noise_steps,
                max_noise_steps=max_noise_steps,
                num_train_timesteps=num_train_timesteps,
                device=self.p.device_torch,
                step_num=self.p.step_num,
            )
            timesteps = result.timesteps
            timestep_indices = result.timestep_indices

        with self.p.timer('convert_timestep_indices_to_timesteps'):
            if is_debug_enabled() and (self.p.logging_config.log_every or 0) > 0:
                if self._timestep_debug_logger is None:
                    self._timestep_debug_logger = TimestepDistributionLogger(
                        self.p.train_config,
                        self.p.logging_config,
                        sd=self.p.sd,
                    )
                self._timestep_debug_logger.collect(
                    timestep_indices, timesteps, content_or_style,
                    self.p.step_num, self._timestep_sampler,
                )
                if self._timestep_debug_logger.should_log():
                    self._timestep_debug_logger.log_and_reset(
                        self.p.step_num,
                        min_noise_steps,
                        max_noise_steps,
                        self.p.sd.noise_scheduler.timesteps,
                    )
        
        return timesteps, min_noise_steps, max_noise_steps, do_double
    
    def _prepare_noise_and_noisy_latents(
        self, batch, latents, unaugmented_latents, timesteps, conditioned_prompts,
        imgs, dtype, is_reg, do_double, min_noise_steps, max_noise_steps
    ):
        """
        Prepare noise and create noisy latents for training.
        
        Args:
            batch: DataLoaderBatchDTO
            latents: Prepared latents tensor
            unaugmented_latents: Unaugmented latents (if applicable)
            timesteps: Sampled timesteps
            conditioned_prompts: Conditioned prompts list
            imgs: Images tensor
            dtype: Data type for tensors
            is_reg: Whether this is a regularization batch
            do_double: Whether to double the batch
            min_noise_steps: Minimum noise steps
            max_noise_steps: Maximum noise steps
            
        Returns:
            tuple: (noisy_latents, noise, timesteps, imgs)
        """
        batch_size = len(batch.file_items)
        
        with self.p.timer('prepare_noise'):
            # get noise
            noise = self.p.get_noise(latents, batch_size, dtype=dtype, batch=batch, timestep=timesteps)

            # add dynamic noise offset. Dynamic noise is offsetting the noise to the same channelwise mean as the latents
            # this will negate any noise offsets
            if self.p.train_config.dynamic_noise_offset and not is_reg:
                latents_channel_mean = latents.mean(dim=(2, 3), keepdim=True) / 2
                # subtract channel mean to that we compensate for the mean of the latents on the noise offset per channel
                noise = noise + latents_channel_mean

            if self.p.train_config.loss_target == 'differential_noise':
                differential = latents - unaugmented_latents
                # add noise to differential
                # noise = noise + differential
                noise = noise + (differential * 0.5)
                # noise = value_map(differential, 0, torch.abs(differential).max(), 0, torch.abs(noise).max())
                latents = unaugmented_latents

            noise_multiplier = self.p.train_config.noise_multiplier
            
            s = (noise.shape[0], noise.shape[1], 1, 1)
            if len(noise.shape) == 5:
                # if we have a 5d tensor, then we need to do it on a per batch item, per channel basis, per frame
                s = (noise.shape[0], noise.shape[1], noise.shape[2], 1, 1)
            
            noise = noise * noise_multiplier
            
            if self.p.train_config.do_signal_correction_noise:
                batch_noise = latents.clone().to(noise.device, dtype=noise.dtype)
                scn_scale = torch.randn(
                    batch_noise.shape[0], batch_noise.shape[1], 1, 1,
                    device=batch_noise.device, 
                    dtype=batch_noise.dtype
                ) * self.p.train_config.signal_correction_noise_scale
                batch_noise = batch_noise * scn_scale
                noise = noise + batch_noise 
            
            if self.p.train_config.random_noise_shift > 0.0:
                # get random noise -1 to 1
                noise_shift = torch.randn(
                    batch_size, latents.shape[1], 1, 1,
                    device=noise.device,
                    dtype=noise.dtype
                ) * self.p.train_config.random_noise_shift
                # add to noise
                noise += noise_shift
            
            if self.p.train_config.random_noise_multiplier > 0.0:
                sigma = self.p.train_config.random_noise_multiplier
                noise_multiplier = torch.exp(torch.randn(s, device=noise.device, dtype=noise.dtype) * sigma)
            
        with self.p.timer('make_noisy_latents'):

            latent_multiplier = self.p.train_config.latent_multiplier

            # handle adaptive scaling mased on std
            if self.p.train_config.adaptive_scaling_factor:
                std = latents.std(dim=(2, 3), keepdim=True)
                normalizer = 1 / (std + 1e-6)
                latent_multiplier = normalizer

            latents = latents * latent_multiplier
            
            if self.p.train_config.do_blank_stabilization:
                # zero out latents with blank prompts
                blank_latent = torch.zeros_like(latents)
                for i, prompt in enumerate(conditioned_prompts):
                    if prompt.strip() == '':
                        latents[i] = blank_latent[i]
            
            batch.latents = latents

            # normalize latents to a mean of 0 and an std of 1
            # mean_zero_latents = latents - latents.mean()
            # latents = mean_zero_latents / mean_zero_latents.std()

            if batch.unconditional_latents is not None:
                batch.unconditional_latents = batch.unconditional_latents * self.p.train_config.latent_multiplier


            noisy_latents = self.p.sd.add_noise(latents, noise, timesteps)

            # determine scaled noise
            # todo do we need to scale this or does it always predict full intensity
            # noise = noisy_latents - latents

            # https://github.com/huggingface/diffusers/blob/324d18fba23f6c9d7475b0ff7c777685f7128d40/examples/t2i_adapter/train_t2i_adapter_sdxl.py#L1170C17-L1171C77
            if self.p.train_config.loss_target == 'source' or self.p.train_config.loss_target == 'unaugmented':
                sigmas = self.p.get_sigmas(timesteps, len(noisy_latents.shape), noisy_latents.dtype)
                # add it to the batch
                batch.sigmas = sigmas
                # todo is this for sdxl? find out where this came from originally
                # noisy_latents = noisy_latents / ((sigmas ** 2 + 1) ** 0.5)

        def double_up_tensor(tensor: torch.Tensor):
            if tensor is None:
                return None
            return torch.cat([tensor, tensor], dim=0)

        if do_double:
            if self.p.model_config.refiner_name_or_path:
                # apply refiner double up
                refiner_timesteps = torch.randint(
                    max_noise_steps,
                    self.p.train_config.max_denoising_steps,
                    (batch_size,),
                    device=self.p.device_torch
                )
                refiner_timesteps = refiner_timesteps.long()
                # add our new timesteps on to end
                timesteps = torch.cat([timesteps, refiner_timesteps], dim=0)

                refiner_noisy_latents = self.p.sd.noise_scheduler.add_noise(latents, noise, refiner_timesteps)
                noisy_latents = torch.cat([noisy_latents, refiner_noisy_latents], dim=0)

            else:
                # just double it
                noisy_latents = double_up_tensor(noisy_latents)
                timesteps = double_up_tensor(timesteps)

            noise = double_up_tensor(noise)
            # prompts are already updated above
            imgs = double_up_tensor(imgs)
            batch.mask_tensor = double_up_tensor(batch.mask_tensor)
            batch.control_tensor = double_up_tensor(batch.control_tensor)

        noisy_latent_multiplier = self.p.train_config.noisy_latent_multiplier

        if noisy_latent_multiplier != 1.0:
            noisy_latents = noisy_latents * noisy_latent_multiplier

        # remove grads for these
        noisy_latents.requires_grad = False
        noisy_latents = noisy_latents.detach()
        noise.requires_grad = False
        noise = noise.detach()
        
        return noisy_latents, noise, timesteps, imgs
