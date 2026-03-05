import numpy as np
from typing import Literal
from scipy.signal import fftconvolve
import gpuRIR
from sfm.data.base_librimix import BaseLibriMix
from sfm.data import utils


class LibriMix(BaseLibriMix):
    '''
    Spatialized LibriMix dataset similar to Tesch et al. https://ieeexplore.ieee.org/document/10096098
    '''
    def __init__(
        self,
        dataset_name: Literal['train', 'dev', 'test'] = 'train',
        audio_params: dict = {
            'n_channel': 3,
            'n_speaker': 2,
            'sample_rate': 16000
        },
        data_params: dict = {
            'corpus_params': {
                'cache_path': None,
                'audio_time': 5.0,
                'version': 100,
            },
            'acoustic_params': {
                'room': {'size': [[4.0, 4.0, 2.2], [8.0, 8.0, 3.5]], 't60': [0.2, 0.5]},
                'speaker': {'array_dist': [1.0, 2.0], 'height': [1.6, 0.08], 'wall_dist': 0.5, 'spk_az_dist': 15},
                'array': {'type': 'circular', 'radius': 0.05, 'height': [1.5, 0.08], 'wall_dist': 1.0}
            },
        },
        **kwargs,
    ):
        super().__init__(dataset_name, audio_params, data_params)

        # initialize attributes
        self.n_channel = audio_params['n_channel']
        self.acoustic_params = data_params['acoustic_params']

    def get_scenario(
        self,
        example: dict,
    ):
        rng = example['rng']

         # room parameters
        room_sz = rng.uniform(
            *self.acoustic_params['room']['size']
        )  # (3)
        example['room_sz'] = room_sz
        example['t60'] = rng.uniform(
            *self.acoustic_params['room']['t60']
        )

        # microphone placement
        array_pos = np.append(
            room_sz[:2] * rng.uniform(*self.acoustic_params['array']['rel_pos'], size=2), 
            np.clip(
                rng.normal(*self.acoustic_params['array']['height']),
                a_min=0.2, a_max=room_sz[-1] - 0.2
            )  # at least 20cm from floor / ceiling
        )  # (3)
        if self.acoustic_params['array']['type'] == 'circular':
            mic_pos = self.acoustic_params['array']['radius'] * np.stack(
                (
                    np.cos(np.linspace(0, 2 * np.pi, num=self.n_channel, endpoint=False)), 
                    np.sin(np.linspace(0, 2 * np.pi, num=self.n_channel, endpoint=False)), 
                    np.zeros(self.n_channel)
                ), axis=-1
            )  # (CHANNEL, 3)
        else:
            raise NotImplementedError
        mic_rot = rng.uniform(-np.pi, np.pi)
        example['mic_pos'] = array_pos + np.sum(
            np.array(
                (
                    (np.cos(mic_rot), -np.sin(mic_rot), 0), 
                    (np.sin(mic_rot), np.cos(mic_rot), 0), 
                    (0, 0, 1)
                )
            ) * mic_pos[:, None, :], axis=-1
        )  # (CHANNEL, 3)
        example['mic_rot'] = mic_rot / np.pi * 180
        example['array_pos'] = array_pos  # (3) 

        # speaker start positions
        spk_az_az_min = self.acoustic_params['speaker']['spk_az_dist'] / 180 * np.pi
        spk_start_min_wall_dist = self.acoustic_params['speaker']['wall_dist']
        spk_start_min_array_dist = self.acoustic_params['speaker']['array_dist'][0]
        spk_start_bbox = (
            spk_start_min_wall_dist * np.ones(2), room_sz[:2] - spk_start_min_wall_dist
        )  # (2, 2) = (low, high)
        while True:
            # draw random position
            spk_pos_start = np.concatenate(
                (
                    rng.uniform(*spk_start_bbox, (self.n_speaker, 2)),  # (SPEAKER, 2)
                    np.clip(
                        rng.normal(*self.acoustic_params['speaker']['height'], self.n_speaker),
                        a_min=0.2, a_max=room_sz[-1] - 0.2
                    )[:, None]  # (SPEAKER, 1) at least 20cm from floor / ceiling
                ), axis=-1
            )  # (SPEAKER, 3)

            # check distance to array condition
            spk_array_dist_start = np.linalg.norm(
                spk_pos_start[:, :2] - array_pos[:2], axis=-1
            )
            if spk_array_dist_start.min() < spk_start_min_array_dist:
                continue

            # check azimuth condition
            spk_az_start = utils.cart2sph(
                spk_pos_start - array_pos
            )[:, -1]  # (SPEAKER)
            spk_az_start_sorted = np.sort(spk_az_start)
            if np.cos(np.diff(spk_az_start_sorted, append=spk_az_start_sorted[0])).max() > np.cos(spk_az_az_min):
                continue

            # all checks complete
            break

        example['spk_pos'] = spk_pos_start[:, None]  # (SPEAKER, SAMPLES=1, 3)
        example['spk_traj'] = spk_pos_start[:, None]  # (SPEAKER, FRAMES=1, 3)

        # compute DoA
        doa = utils.cart2sph(
            example['spk_traj'] - example['array_pos']
        )[..., 1:]  # (SPEAKER, 1, 2) (elevation, azimuth)
        doa[..., -1] = np.angle(np.exp(1j * (doa[..., -1] - mic_rot)))  # correct for array rotation
        example['target_angle'] = doa / np.pi * 180  # (SPEAKER, FRAMES=1, 2)
        example['mic_rot'] = mic_rot / np.pi * 180 

    def get_convolution(
        self,
        signal: np.ndarray,  # (SPEAKER, SAMPLES)
        rir: np.ndarray,  # (SPEAKER, FRAMES, CHANNEL, SAMPLES')
    ) -> np.ndarray:  # (SPEAKER, CHANNEL, SAMPLES)
        n_samples = signal.shape[-1]
        n_frames = rir.shape[-3]
        if n_frames > 1:
            # moving sources
            raise NotImplementedError(n_frames)
        else:
            # standart convolution for stationary sources
            return fftconvolve(
                signal[..., None, :], rir.squeeze(-3), mode='full', axes=-1
            )[..., :n_samples]  # (..., CHANNEL, SAMPLES)
       
    def get_simulation(
        self, 
        example: dict
    ):
        # set simulation parameters according to https://github.com/DavidDiazGuerra/Cross3D/blob/master/acousticTrackingDataset.py
        Tdiff = gpuRIR.att2t_SabineEstimator(12, example['t60']) # time ISM -> diffuse model
        Tmax = gpuRIR.att2t_SabineEstimator(40, example['t60'])  # simulation time
        beta = gpuRIR.beta_SabineEstimation(example['room_sz'], example['t60'])
        n_img = gpuRIR.t2n(Tdiff, example['room_sz'])
        n_frames = example.get('audio_frames', 1)  # singleton dimension for a static speaker

        # compute RIRs and convolve with clean speech (=images)
        reverb_rir = gpuRIR.simulateRIR(
            example['room_sz'],  # (3)
            beta,
            example['spk_traj'].reshape(self.n_speaker * n_frames, 3),  # (SPEAKER*FRAMES, 3)
            example['mic_pos'],  # (CHANNEL, 3)
            n_img, 
            Tmax, 
            self.sample_rate, 
            Tdiff=Tdiff,
        ).reshape(
            self.n_speaker, n_frames, self.n_channel, -1
        )  # (SPEAKER, FRAMES, CHANNEL, SAMPLES')
        example['image_td'] = self.get_convolution(
            example['clean_td'], reverb_rir
        )  # (SPEAKER, CHANNEL, SAMPLES)

        # additive mixture between reverb. signals
        example['mix_td'] = example['image_td'].sum(0)  # (CHANNEL, SAMLES)

        # compute dry RIRs and convolve with clean speech (=targets)
        dry_len = np.linalg.norm(
            example['spk_traj'][..., None, :, :2] - example['mic_pos'][..., None, :2], axis=-1  # (SPEAKER, CHANNEL, FRAMES, 2)
        ).max()  # max distance speaker-array
        dry_Tmax = dry_len / 343 + 100 / self.sample_rate  # 100 samples more due to bandlimited sinc interpolation
        dry_rir = gpuRIR.simulateRIR(
            example['room_sz'],  # (3)
            beta,
            example['spk_traj'].reshape(self.n_speaker * n_frames, 3),  # (SPEAKER*FRAMES, 3)
            example['mic_pos'],  # (CHANNEL, 3)
            [1, 1, 1],  # only one image source
            dry_Tmax,
            self.sample_rate,
        ).reshape(
            self.n_speaker, n_frames, self.n_channel, -1
        )  # (SPEAKER, FRAMES, CHANNEL, SAMPLES')    
        example['dry_td'] = self.get_convolution(
            example['clean_td'], dry_rir
        )  # (SPEAKER, CHANNEL, SAMPLES)
    
    def __getitem__(self, idx):
        if self.dataset_path is not None:
            # use cached dataset
            example = self.load_example(idx)
        else:
            # init example
            example = self.init_example(idx)

            # load corpus utterances
            self.get_utterances(example)

            # create scenario and spatialize audio
            self.get_scenario(example)
            self.get_simulation(example)

            # convert to torch tensors
            self.get_tensors(example)

        return example
    