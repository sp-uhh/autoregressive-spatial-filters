import pytorch_lightning as pl
import torch
from typing import Literal, Tuple
from torch.optim import Adam, lr_scheduler


class BaseExp(pl.LightningModule):
    def __init__(
            self,
            model: torch.nn.Module,
            audio_params = {
                'n_channel': 3,
                'n_speaker': 2,
                'sample_rate': 16000,
                'stft_length': 512,
                'stft_shift': 256,
            },
            data_params = {
                'noise_params': None
            },
            exp_params = {
                'ref_channel': 0,
                'cond_arange_params': [[-90, 90, 2], [-180, 180, 2]],
                'enhancement_type': 'miso',
                'optim_params': {
                    'lr': 0.001,
                    'scheduler_type': 'StepLR',
                    'scheduler_params': {
                        'gamma': 0.75,
                        'step_size': 50,
                    },
                },
                'loss_params': {
                    'loss_type': 'teschL1',
                    'loss_alpha': 10,
                },
            },
            logging_params = {
                'metric_params': {

                },
            },
            train_params = {
                'batch_size': 1
            },
            **kwargs,
        ):
        super().__init__()

        # STFT params
        self.stft_length = audio_params['stft_length']
        self.stft_shift = audio_params['stft_shift']
        self.sample_rate = audio_params['sample_rate'] 

        # experiment params
        self.model = model
        self.n_speaker = audio_params['n_speaker']
        self.n_channel = audio_params['n_channel']
        self.batch_size = train_params['batch_size']
        self.noise_params = data_params['noise_params']
        self.exp_params = exp_params
        self.optim_params = exp_params['optim_params']
        self.loss_params = exp_params['loss_params']
        self.ref_channel = exp_params['ref_channel']
        self.cond_arange_params = exp_params['cond_arange_params']
        self.n_el_doa = (self.cond_arange_params[0][1] - self.cond_arange_params[0][0]) // self.cond_arange_params[0][2] 
        self.n_az_doa = (self.cond_arange_params[1][1] - self.cond_arange_params[1][0]) // self.cond_arange_params[1][2]

        # logging params
        self.logging_params = logging_params
        self.metric_params = logging_params['metric_params']

    def shared_step(self, batch, batch_idx, stage: Literal['train', 'val']):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage='val')

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, stage='test')

    def configure_optimizers(self):
        optim = Adam(self.model.parameters(), lr=self.optim_params['lr'])
        scheduler_type = self.optim_params.get('scheduler_type')
        if scheduler_type is not None:
            scheduler = {
                'scheduler': getattr(lr_scheduler, scheduler_type)(optimizer=optim, **self.optim_params['scheduler_params']),
                'name': 'lr_schedule'
            }
            return {'optimizer': optim, 'lr_scheduler': scheduler}
        
        return optim
   
    def get_stft_rep(self, *td_signals, return_complex=True):
        """ 
        K. Tesch https://github.com/sp-uhh/deep-non-linear-filter
        """
        result = []
        window = torch.sqrt(torch.hann_window(self.stft_length)).to(device=self.device)
        for td_signal in td_signals:
            if len(td_signal.shape) == 1:  # single-channel
                stft = torch.stft(td_signal, self.stft_length, self.stft_shift, window=window, center=True,
                                  onesided=True, return_complex=return_complex)
                result.append(stft)
            else:  # multi-channel and/or multiple speakers
                signal_shape = td_signal.shape
                reshaped_signal = td_signal.reshape((signal_shape[:-1].numel(), signal_shape[-1]))
                stfts = torch.stft(reshaped_signal, self.stft_length, self.stft_shift, window=window, center=True,
                                   onesided=True, return_complex=return_complex)
                _, freq_dim, time_dim = stfts.shape
                stfts = stfts.reshape(signal_shape[:-1] + (freq_dim, time_dim))
                result.append(stfts)
        return result
                
    def get_td_rep(self, *stfts):
        """
        K. Tesch https://github.com/sp-uhh/deep-non-linear-filter
        """
        result = []
        window = torch.sqrt(torch.hann_window(self.stft_length)).to(device=self.device)
        for stft in stfts:
            has_complex_dim = stft.shape[-1] == 2
            if (not has_complex_dim and len(stft.shape) <= 3) or (has_complex_dim and len(stft.shape) <= 4):  # single-channel
                td_signal = torch.istft(stft, self.stft_length, self.stft_shift, window=window, center=True,
                                        onesided=True,
                                        return_complex=False)
                result.append(td_signal)
            else:  # multi-channel
                signal_shape = stft.shape
                if not has_complex_dim:
                    reshaped_signal = stft.reshape((signal_shape[:-2].numel(), signal_shape[-2], signal_shape[-1]))
                    td_signals = torch.istft(reshaped_signal, self.stft_length, self.stft_shift, window=window, center=True, onesided=True, return_complex=False)
                    _, n_samples = td_signals.shape
                    td_signals = td_signals.reshape(signal_shape[:-2]+(n_samples,))
                else:
                    reshaped_signal = stft.reshape((signal_shape[:-3].numel(), signal_shape[-3], signal_shape[-2], signal_shape[-1]))
                    td_signals = torch.istft(reshaped_signal, self.stft_length, self.stft_shift, window=window, center=True, onesided=True, return_complex=False)
                    _, n_samples = td_signals.shape
                    td_signals = td_signals.reshape(signal_shape[:-3]+(n_samples,))
                result.append(td_signals)
        return result

    def get_steering_vector(
        self,
        target_angle: torch.Tensor,  # (BATCH, SPEAKER, FRAMES, 2) / (BATCH, SPEAKER, FRAMES, 1)
        mic_pos: torch.Tensor,  # (BATCH, CHANNEL, 3)
    ) -> torch.Tensor:  # (BATCH, SPEAKER, CHANNEL, FREQ, FRAMES)
        n_channel = mic_pos.shape[-2]
        n_freq = self.stft_length // 2 + 1
        mic_delta = torch.stack(
            [mic_pos[:, i] - mic_pos[:, self.ref_channel] for i in range(n_channel)], dim=-2
        )  # (BATCH, CHANNEL, 3)
        if target_angle.shape[-1] == 1:
            # only azimuth
            azimuth = target_angle[..., -1] / 180 * torch.pi  # (BATCH, SPEAKER, FRAMES)
            source_normal = torch.stack(
                (torch.cos(azimuth), torch.sin(azimuth)), dim=-1
            )  # (BATCH, SPEAKER, FRAMES, 2)
            mic_delta = mic_delta[..., :2]  # (CHANNEL, 2)
        elif target_angle.shape[-1] == 2:
            # (elevation, azimuth)
            elevation = target_angle[..., 0] / 180 * torch.pi  # (BATCH, SPEAKER, FRAMES)
            azimuth = target_angle[..., 1] / 180 * torch.pi  # (BATCH, SPEAKER, FRAMES)
            source_normal = torch.stack(
                (
                    torch.cos(azimuth) * torch.cos(elevation),
                    torch.sin(azimuth) * torch.cos(elevation),
                    torch.sin(elevation)
                ), dim=-1
            )  # (BATCH, SPEAKER, FRAMES, 3)
        else:
            raise ValueError
        delay = torch.sum(source_normal[..., None, :, :] * mic_delta[:, None, :, None, :], dim=-1) / 343 * self.sample_rate  # (BATCH, SPEAKER, CHANNEL, FRAMES)
        return torch.exp(
            1j * 2 * torch.pi / self.stft_length * torch.arange(n_freq, device=self.device)[:, None] * delay[..., None, :]
        ) / n_channel**0.5  # (..., CHANNEL, FREQ, FRAMES)

    def get_spatial_spectrum(
        self,
        spatial_stft: torch.Tensor,  # (BATCH, SPEAKER, CHANNEL, FREQ, FRAMES)
        batch: dict,
    ) -> torch.Tensor:  # (BATCH, SPEAKER, DOA, FRAMES)
        spatial_stft_norm = torch.nn.functional.normalize(spatial_stft, dim=-3)  # (..., CHANNEL, FREQ, FRAMES)
        angles = torch.arange(*self.exp_params['cond_arange_params'][1], device=self.device)  # (AZ)
        angles = angles + batch['mic_rot'][:, None, None]  # (BATCH, SPEAKER, AZ)
        steering_vector = self.get_steering_vector(
            angles[..., None], batch['mic_pos']
        )  # (BATCH, SPEAKER, CHANNEL, FREQ, AZ)
        spatial_similarity = torch.sum(
            torch.conj(spatial_stft_norm[..., None, :]) * steering_vector[..., None], dim=-4
        )  # (BATCH, SPEAKER, FREQ, AZ, FRAMES)
        return torch.mean(
            torch.abs(spatial_similarity)**2, dim=-3
        ) # (BATCH, SPEAKER, AZ, FRAMES)
    
    def get_spherical_isotropic_noise(
        self,
        mix_stft: torch.Tensor,  # (BATCH, CHANNEL, FREQ, FRAMES)
        mic_pos: torch.Tensor,  # (BATCH, CHANNEL, 3)
        rng: torch.Generator,
    ) -> torch.Tensor:  # (BATCH, CHANNEL, FREQ, FRAMES)
        """
        Habets et al. DOI: 10.1121/1.2799929
        """
        n_batch, n_channel, n_freq, n_frames = mix_stft.shape
        n_points = self.noise_params['n_points']
        
        # fibonacci lattice on sphere
        indices = torch.arange(0, n_points, dtype=torch.float32, device=self.device) + 0.5
        phi = torch.acos(1 - 2*indices/n_points)
        theta = torch.pi * (1 + 5**0.5) * indices
        noise_normal = torch.stack(
            [
                torch.sin(phi) * torch.cos(theta), 
                torch.sin(phi) * torch.sin(theta), 
                torch.cos(phi)
            ], dim=-1
        )  # (NOISE, 3)
        
        # compute steering vectors
        mic_delta = torch.stack(
            [mic_pos[:, i] - mic_pos[:, self.ref_channel] for i in range(n_channel)], dim=-2
        )  # (BATCH, CHANNEL, 3)
        delay = torch.sum(
            noise_normal[..., None, :] * mic_delta[:, None], dim=-1
        ) / 343 * self.sample_rate  # (BATCH, NOISE, CHANNEL)
        noise_phase = 2 * torch.pi / self.stft_length * torch.arange(n_freq, device=self.device) * delay[..., None]  # (BATCH, NOISE, CHANNEL, FREQ)
        noise_steering_vector = torch.cos(noise_phase) + 1j * torch.sin(noise_phase)  # (BATCH, NOISE, CHANNEL, FREQ)
        
        # draw white noise
        noise_stft = torch.randn(
            n_batch, n_points, n_freq, n_frames, generator=rng, device=self.device, dtype=torch.complex64
        )  # (BATCH, NOISE, FREQ, FRAMES)

        # compute isotropic average
        isotropic_noise_stft = torch.einsum(
            'bnft, bncf -> bcft', noise_stft, noise_steering_vector
        ) / n_points # (BATCH, CHANNEL, FREQ, FRAMES)
        
        # scale to SNR
        snr_low, snr_high = self.noise_params['snr_range']
        snr = (snr_high - snr_low) * torch.rand(n_batch, generator=rng, device=self.device) + snr_low
        snr_scale = 10**(- snr / 20) \
            * torch.std(mix_stft[:, self.ref_channel].abs(), dim=(-2, -1)) \
            / torch.std(isotropic_noise_stft[:, self.ref_channel].abs(), dim=(-2, -1))  # (BATCH)
        isotropic_noise_stft = snr_scale[:, None, None, None] * isotropic_noise_stft  # (BATCH, CHANNEL, FREQ, FRAMES)
        
        return isotropic_noise_stft

    def preprocessing(self, batch):
        # get rng
        batch['seed'] = sum(batch['seed'])
        batch['rng'] = torch.Generator(self.device).manual_seed(batch['seed'])

        if 'noisy_stft' in batch.keys() and 'dry_stft' in batch.keys():
            # cached dataset is loaded, skip stft and noise computation
            pass
        else:
             # transform to stft
            for signal_name in [
                    '_'.join(key.split('_')[:-1]) for key in batch.keys() if key.split('_')[-1] == 'td'
                ]:
                batch[f'{signal_name}_stft'], = self.get_stft_rep(
                    batch[f'{signal_name}_td']
                )

             # add diffuse noise
            if self.noise_params is None:
                batch['noisy_stft'] = batch['mix_stft']  # (BATCH, CHANNEL, FREQ, FRAMES)
            else:
                batch['noise_stft'] = self.get_spherical_isotropic_noise(
                    batch['mix_stft'], batch['mic_pos'], batch['rng']
                )  # (BATCH, CHANNEL, FREQ, FRAMES)
                batch['noisy_stft'] = batch['mix_stft'] + batch['noise_stft']

        # target is anechoic speech signal
        if self.exp_params['enhancement_type'] == 'miso':
            batch['target_stft'] = batch['dry_stft'][..., [self.ref_channel], :, :]  # (BATCH, SPEAKER, 1, FREQ, FRAMES)
        else:
            batch['target_stft'] = batch['dry_stft']  # (BATCH, SPEAKER, CHANNEL, FREQ, FRAMES)

        # input is noisy stacked for all speakers
        batch['input_stft'] = batch['noisy_stft'][:, None].expand(
            -1, self.n_speaker, -1, -1, -1
        )  # (BATCH, SPEAKER, CHANNEL, FREQ, FRAMES)         

        # broadcast doa to stft frames
        target_angle = batch['target_angle']  # (BATCH, SPEAKER, FRAMES, 2)
        target_idx = torch.stack(
            (
                torch.clamp(
                    (target_angle[..., 0] - self.cond_arange_params[0][0]) / self.cond_arange_params[0][2], 
                    min=0, max=self.n_el_doa-1
                ),
                torch.clamp(
                    (target_angle[..., 1] - self.cond_arange_params[1][0]) / self.cond_arange_params[1][2],
                    min=0, max=self.n_az_doa-1
                )
            ), dim=-1
        ).to(dtype=torch.long)  # (BATCH, SPEAKER, FRAMES, 2)
        n_frames = batch['input_stft'].shape[-1]
        if target_angle.shape[-2] > 1:
            assert n_frames == target_angle.shape[-2], (n_frames, target_angle.shape[-2])
            batch['target_idx'] = target_idx  # (BATCH, SPEAKER, FRAMES, 2)
        else:
            batch['target_idx'] = target_idx.expand(-1, -1, n_frames, -1)  # (BATCH, SPEAKER, FRAMES, 2)
            batch['target_angle'] = target_angle.expand(-1, -1, n_frames, -1)  # (BATCH, SPEAKER, FRAMES, 2)

    def postprocessing(self, batch):
        # transform to td
        for signal_name in [
                '_'.join(key.split('_')[:-1]) for key in batch.keys() if key.split('_')[-1] == 'stft'
        ]:
            batch[f'{signal_name}_td'], = self.get_td_rep(
                batch[f'{signal_name}_stft']
            )
        