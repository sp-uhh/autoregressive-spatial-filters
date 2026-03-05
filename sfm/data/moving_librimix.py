import numpy as np
from sfm.data.librimix import LibriMix
from gpuRIR import gpuRIR_bind_simulator
from sfm.data import utils


class MovingLibriMix(LibriMix):
    '''
    Social force motion model based on Helbing et al. 10.1103/PhysRevE.51.4282
    '''
    def __init__(
        self,
        audio_params: dict = {
            'n_channel': 3,
            'n_speaker': 2,
            'sample_rate': 16000,
            'stft_length': 512,
            'stft_shift': 256,
        },
        **kwargs,
    ):
        super().__init__(audio_params=audio_params, **kwargs)

        # init STFT params
        self.stft_length = audio_params['stft_length']
        self.stft_shift = audio_params['stft_shift']

        # init motion model params
        self.motion_params = self.acoustic_params['motion_params']

    def get_convolution(
        self,
        signal: np.ndarray,  # (SPEAKER, SAMPLES)
        rir: np.ndarray,  # (SPEAKER, FRAMES, CHANNEL, SAMPLES')
    ) -> np.ndarray:  # (SPEAKER, CHANNEL, SAMPLES)
        n_samples = signal.shape[-1]
        n_frames = rir.shape[-3]
        if n_frames > 1:
            # moving sources
            n_rir = rir.shape[-1]
            tmp_batch = self.n_speaker * n_frames
            rir = rir.reshape(tmp_batch, self.n_channel, n_rir)  # (BATCH, CHANNEL, SAMPLES')
            assert self.stft_length // 2 == self.stft_shift, (
                self.stft_length, self.stft_shift, 'only implemented for 0.5 shift'
            )

            # shift and pad signal according to centered stft
            signal = np.pad(
                signal, ((0, 0), (self.stft_shift, self.stft_shift - n_samples % self.stft_shift))
            )  # (SPEAKER, SAMPLES)

            # weighted overlap-add
            idx = np.arange(self.stft_length)[None, :] + np.arange(n_frames)[:, None] * self.stft_shift  # (FRAMES, WINDOW)
            segments = np.hanning(self.stft_length)[None, :] * signal[:, idx]  # (SPEAKER, FRAMES, WINDOW)
            segments = segments.reshape(tmp_batch, self.stft_length).astype('float32', order='C', copy=False)  # (BATCH, WINDOW)
            convolution = gpuRIR_bind_simulator.gpu_conv(segments, rir).reshape(
                self.n_speaker, n_frames, self.n_channel, -1
            )  # (SPEAKER, FRAMES, CHANNEL, SAMPLES')
            filtered_signal = np.zeros((self.n_speaker, self.n_channel, signal.shape[-1] + n_rir - 1))  # (SPEAKER, CHANNEL, SAMPLES'')
            for frame in range(n_frames):
                filtered_signal[..., frame*self.stft_shift : frame*self.stft_shift + self.stft_length + n_rir-1] += convolution[:, frame, :, :self.stft_length + n_rir-1]

            # shift back and crop
            return filtered_signal[..., self.stft_shift: self.stft_shift + n_samples]  # (SPEAKER, CHANNEL, SAMPLES)
        else:
            super().get_convolution(signal, rir)

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
        spk_start_min_wall_dist = \
            self.acoustic_params['speaker']['wall_dist'] + self.motion_params['repulsive_force']['wall_potential_B']
        spk_start_min_array_dist = \
            self.acoustic_params['speaker']['array_dist'][0] + self.motion_params['repulsive_force']['array_potential_B']
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

        # parametrize motion model
        n_frames = example['audio_samples'] // self.stft_shift + 1  # match STFT implementation
        delta_t = example['audio_time'] / n_frames
        example['audio_frames'] = n_frames
        driving_dist_min = self.motion_params['driving_force']['driving_dist']
        driving_vel_abs = np.clip(
            rng.normal(*self.motion_params['driving_force']['vel'], size=self.n_speaker), a_min=0, a_max=np.inf
        )  # (SPEAKER)
        vel_abs_max = self.motion_params['driving_force']['vel_max']
        wall_dist_min = self.acoustic_params['speaker']['wall_dist']
        wall_potential_B = self.motion_params['repulsive_force']['wall_potential_B']
        wall_potential_A = 0.5 * driving_vel_abs**2 * np.exp(wall_dist_min / wall_potential_B)  # (SPEAKER)
        spk_potential_B = self.motion_params['repulsive_force']['spk_potential_B']
        spk_potential_A = self.motion_params['repulsive_force']['spk_potential_A']
        spk_stride_t = self.motion_params['repulsive_force']['spk_stride_t']
        array_dist_min = self.acoustic_params['speaker']['array_dist'][0]
        array_potential_B = self.motion_params['repulsive_force']['array_potential_B']
        array_potential_A = 0.5 * driving_vel_abs**2 * np.exp(
            2 * array_dist_min / array_potential_B
        )  # (SPEAKER)

        # draw driving position
        spk_driving_bbox = (
            self.acoustic_params['speaker']['wall_dist'] * np.ones(2), room_sz[:2] - self.acoustic_params['speaker']['wall_dist']
        )  # (2, 2) = (low, high)
        def get_driving_pos() -> np.ndarray:  # (SPEAKER, 2)
            while True:
                # draw random position
                driving_pos = rng.uniform(*spk_driving_bbox, (self.n_speaker, 2))  # (SPEAKER, 2)

                # check for distance to array
                driving_pos_array_dist = np.linalg.norm(
                    driving_pos[:, :2] - array_pos[:2], axis=-1
                )
                if driving_pos_array_dist.min() > self.acoustic_params['speaker']['array_dist'][0]:
                    break
            return driving_pos

        # init arrays
        pos = np.concatenate(
            (spk_pos_start[None, :, :2], np.zeros((n_frames-1, self.n_speaker, 2))), axis=0
        )  # (FRAMES, SPEAKER, 2)
        vel = np.zeros((n_frames, self.n_speaker, 2))  # (FRAMES, SPEAKER, 2)
        driving_pos_start = get_driving_pos()  # (SPEAKER, 2), (SPEAKER, 2)
        driving_pos = np.broadcast_to(driving_pos_start, (n_frames, self.n_speaker, 2)).copy()  # (FRAMES, SPEAKER, 2)
        driving_force = np.empty((n_frames, self.n_speaker, 2))  # (FRAMES, SPEAKER, 2)
        boundary_force = np.empty((n_frames, self.n_speaker, 2))  # (FRAMES, SPEAKER, 2)
        spk_spk_force = np.empty((n_frames, self.n_speaker, self.n_speaker, 2))  # (FRAMES, SPEAKER, SPEAKER, 2)

        # circular force 
        def circular_force(
            pos_delta: np.ndarray,  # (..., 2)
            potential_A: float,
            potential_B: float,
        ) -> np.ndarray:  # (..., 2)
            pos_norm = np.linalg.norm(pos_delta, axis=-1, keepdims=True) + 1e-8  # (..., 1)

            return potential_A * np.exp(- pos_norm / potential_B) * (
                pos_delta / pos_norm
            ) / potential_B

        # elliptical force
        def elliptical_force(
            pos_delta: np.ndarray,  # (..., 2)
            vel_delta: np.ndarray,  # (..., 2)
            potential_A: float,
            potential_B: float,
            step_size: float = spk_stride_t,
        ) -> np.ndarray:  # (..., 2)
            pos_norm = np.linalg.norm(pos_delta, axis=-1, keepdims=True) + 1e-8  # (..., 1)
            step_delta = step_size * vel_delta  # (..., 2)
            step_norm = np.linalg.norm(step_delta, axis=-1, keepdims=True)  # (..., 1)
            pos_step_delta = pos_delta + step_delta  # (..., 2)
            pos_step_norm = np.linalg.norm(pos_step_delta, axis=-1, keepdims=True) + 1e-8  # (..., 1)

            semi_minor = np.sqrt(
                np.clip(
                    (pos_norm + pos_step_norm)**2 - step_norm**2, a_max=np.inf, a_min=1e-8
                )
            )

            return potential_A * np.exp(- semi_minor / potential_B) * (pos_norm + pos_step_norm) / semi_minor * (
                pos_delta / pos_norm + pos_step_delta / pos_step_norm
            ) / potential_B

        # forward/solve motion model
        for frame in range(n_frames):
            # update/compute driving force
            driving_delta = driving_pos[frame] - pos[frame]  # (SPEAKER, 2)
            driving_dist = np.linalg.norm(driving_delta, axis=-1) # (SPEAKER)
            driving_pos_change = driving_dist < driving_dist_min # change this
            if np.any(driving_pos_change): 
                driving_pos[frame:, driving_pos_change] = get_driving_pos()[driving_pos_change]  # (SPEAKER, 2)
                driving_delta = driving_pos[frame] - pos[frame]  # (SPEAKER, 2)
                driving_dist = np.linalg.norm(driving_delta, axis=-1) # (SPEAKER)
            driving_vel = driving_delta / (driving_dist[:, None] + 1e-8) * driving_vel_abs[:, None]  # (SPEAKER, 2)
            driving_force[frame] = (driving_vel - vel[frame]) / self.motion_params['driving_force']['vel_relax_t']

            # compute boundary forces
            boundary_force[frame] = circular_force(
                pos_delta=pos[frame, :, None, :] * np.eye(2),  # (SPEAKER, 2, 2)
                potential_A=wall_potential_A[:, None, None],  # (SPEAKER, 1, 1)
                potential_B=wall_potential_B
            ).sum(-2) + circular_force(
                pos_delta=pos[frame, :, None, :] * np.eye(2) - room_sz[:2] * np.eye(2),  # (SPEAKER, 2, 2)
                potential_A=wall_potential_A[:, None, None],
                potential_B=wall_potential_B
            ).sum(-2) + elliptical_force(
                pos_delta=pos[frame] - array_pos[:2],  # (SPEAKER, 2)
                vel_delta=vel[frame],  # (SPEAKER, 2)
                potential_A=array_potential_A[:, None],  # (SPEAKER, 1)
                potential_B=array_potential_B
            )  # (SPEAKER, 2)

            # compute repulsive force between speakers
            spk_spk_force[frame] = elliptical_force(
                pos_delta=pos[frame, None] - pos[frame, :, None],  # (SPEAKER, SPEAKER, 2)
                vel_delta=vel[frame, None] - vel[frame, :, None],
                potential_A=spk_potential_A,
                potential_B=spk_potential_B
            )  # (SPEAKER, SPEAKER, 2)
            spk_spk_force[frame, np.arange(self.n_speaker), np.arange(self.n_speaker), :] = 0.0  # set self-force to 0
            spk_force = spk_spk_force[frame].sum(-3)  # (SPEAKER, 2)

            if frame + 1 < n_frames:
                # update position and velocity
                vel[frame+1] = vel[frame] + delta_t * (driving_force[frame] + boundary_force[frame] + spk_force)  # (SPEAKER, 2)
                pos[frame+1] = pos[frame] + delta_t * vel[frame]  # (SPEAKER, 2)

                # hard constraints
                pos[frame + 1] = np.clip(
                    pos[frame + 1], np.zeros(2) + 1e-3, room_sz[:2] - 1e-3
                )  # (SPEAKER, 2)  enforce speakers remain in room
                vel_abs = np.linalg.norm(vel[frame+1], axis=-1, keepdims=True)  # (SPEAKER, 1)
                vel[frame+1] *= np.clip(vel_abs, a_min=0, a_max=vel_abs_max) / vel_abs  # (SPEAKER, 2)  clamp to max velocity

        # store arrays
        example['driving_pos'] = driving_pos.swapaxes(0, 1)  # (SPEAKER, FRAMES, 2) 
        example['driving_force'] = driving_force.swapaxes(0, 1)  # (SPEAKER, FRAMES, 2)    
        example['boundary_force'] = boundary_force.swapaxes(0, 1)  # (SPEAKER, FRAMES, 2)  
        example['spk_spk_force'] = spk_spk_force.transpose(1, 2, 0, 3)  # (SPEAKER, SPEAKER, FRAMES, 2)    
        example['spk_traj'] = np.concatenate(
            (pos, np.broadcast_to(spk_pos_start[None, :, [2]], [n_frames, self.n_speaker, 1])), axis=-1
        ).swapaxes(0, 1)  # (SPEAKER, FRAMES, 2)    

        # interpolate speaker trajectories
        example['spk_pos'] = np.empty(shape=(self.n_speaker, example['audio_samples'], 3))  # (SPEAKER, SAMPLES, 3)
        for spk_idx in range(self.n_speaker):
            for coordinate_idx in range(3):
                example['spk_pos'][spk_idx, :, coordinate_idx] = np.interp(
                    np.arange(example['audio_samples']), 
                    np.linspace(0, example['audio_samples'], num=example['audio_frames'], endpoint=False), 
                    example['spk_traj'][spk_idx, :, coordinate_idx]
                )

        # compute DoA
        doa = utils.cart2sph(
            example['spk_traj'] - example['array_pos']
        )[..., 1:]  # (SPEAKER, FRAMES, 2) (elevation, azimuth)
        doa[..., -1] = np.angle(np.exp(1j * (doa[..., -1] - mic_rot)))  # correct for array rotation
        example['target_angle'] = doa / np.pi * 180  # (SPEAKER, FRAMES, 2)
