import os
import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import soundfile as sf
import hashlib
from typing import Literal


class BaseLibriMix(Dataset):
    '''
    Base class to load clean LibriMix utterance pairs from https://github.com/JorisCos/LibriMix
    '''
    def __init__(
        self,
        dataset_name: Literal['train', 'dev', 'val', 'test'] = 'train',
        audio_params: dict = {
            'n_speaker': 2,
            'sample_rate': 16000,
        },
        data_params: dict = {
            'corpus_params': {
                'cache_path': None,
                'audio_time': 5.0,
                'version': 100,
            },
        },
        **kwargs
    ):
        super().__init__()

        # load path structure
        self.librispeech_path = os.environ['LIBRISPEECH']
        self.librimix_meta_path = os.environ['LIBRIMIX_META']
        self.dataset_path = None if data_params['corpus_params']['cache_path'] is None else os.path.join(
            data_params['corpus_params']['cache_path'], 
            self.__class__.__name__[:-3] +  str(audio_params['n_speaker']) + self.__class__.__name__[-3:],
            dataset_name
        )

        # initialize attributes
        self.dataset_name = 'dev' if dataset_name == 'val' else dataset_name
        self.audio_params = audio_params
        self.data_params = data_params
        self.corpus_params = data_params['corpus_params']
        self.n_speaker = audio_params['n_speaker']
        self.sample_rate = audio_params['sample_rate']
        if self.corpus_params['audio_time'] is not None:
            self.audio_samples = int(self.corpus_params['audio_time'] * self.sample_rate)
        else:
            self.audio_samples = None

         # load librimix metadata
        if 'train' not in self.dataset_name:
            self.librimix_meta_df = pd.read_csv(
                (
                        self.librimix_meta_path + f'Libri{self.n_speaker}Mix/libri{self.n_speaker}mix_{self.dataset_name}-clean.csv'
                ),
                header=0,
                index_col=False
            )
        else:
            if self.corpus_params['version'] == 460:
                # concatenate 100 and 360
                self.librimix_meta_df = pd.concat(
                    [
                        pd.read_csv(
                            (
                                self.librimix_meta_path + f'Libri{self.n_speaker}Mix/libri{self.n_speaker}mix_{self.dataset_name}-clean-{version}.csv'
                            ),
                            header=0,
                            index_col=False
                        ) for version in [100, 360]
                    ], ignore_index=True
                )
            else:
                assert self.corpus_params['version'] in [100, 360], self.corpus_params['version']
                self.librimix_meta_df = pd.read_csv(
                    (
                        self.librimix_meta_path + f"Libri{self.n_speaker}Mix/libri{self.n_speaker}mix_{self.dataset_name}-clean-{self.corpus_params['version']}.csv"
                    ),
                    header=0,
                    index_col=False
                )

        # pairing according to LibriMix
        self.meta_df = pd.DataFrame({
            'mixture_ID': self.librimix_meta_df['mixture_ID'],
            'source_ID': self.librimix_meta_df['mixture_ID'].str.split('_'),
            'source_path': self.librimix_meta_df[
                [f'source_{spk_idx}_path' for spk_idx in range(1, self.n_speaker+1)]
            ].values.tolist(),
            'source_gain': self.librimix_meta_df[
                [f'source_{spk_idx}_gain' for spk_idx in range(1, self.n_speaker+1)]
            ].values.tolist()
        })

    def init_example(self, idx: int) -> dict:
        # load meta
        example = self.meta_df.loc[idx].to_dict()
        example['idx'] = idx

        # get deterministic rng
        entropy = hashlib.sha256(example['mixture_ID'].encode()).digest()[:4]
        example['seed'] = int.from_bytes(entropy[:4], 'big', signed=False)
        example['rng'] = np.random.default_rng(seed=example['seed'])

        return example
    
    def load_example(self, idx: int) -> dict:
        example_path = os.path.join(self.dataset_path, f'example_{idx}.pt')
        if not os.path.exists(example_path):
            raise FileNotFoundError(f"File not found: {example_path}")

        try:
            return torch.load(example_path, map_location='cpu', weights_only=False)
        except (RuntimeError, EOFError) as e:
            raise RuntimeError(f"File could not be loaded {example_path}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error while loading {example_path}") from e
  
    def get_utterances(
        self,
        example: dict,
    ):
        # load clean speech signals
        clean_td = []
        for source_path, source_gain in zip(example['source_path'], example['source_gain']):
            c_td, sr = sf.read(
                os.path.join(self.librispeech_path, source_path)
            )
            assert sr == self.sample_rate, (sr, self.sample_rate)
            clean_td.append(c_td * source_gain)  # scale according to librimix
        
        # determine length of samples acc. to 'max' alignment
        if self.audio_samples is None:
            audio_samples = max([len(c_td) for c_td in clean_td])
        else:
            audio_samples = self.audio_samples
        example['audio_samples'] = audio_samples
        example['audio_time'] = audio_samples / self.sample_rate
        
        # stack signals by appending zeros
        example['clean_td'] = np.stack(
            [
                np.pad(c_td, (0, audio_samples - len(c_td))) if len(c_td) < audio_samples else c_td[:audio_samples]
                for c_td in clean_td
            ], axis=0
        )  # (SPEAKER, SAMPLES)  

    def get_tensors(
        self,
        example: dict,
    ):
        for key, value in example.items():
            if isinstance(value, np.ndarray):
                example[key] = torch.from_numpy(
                    value.astype(np.float32)
                )
            elif isinstance(value, (int, float)) and key != 'seed':
                example[key] = torch.tensor(value, dtype=torch.float32)

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        if self.dataset_path is not None:
            # use cached dataset
            example = self.load_example(idx)
        else:
            # init example
            example = self.init_example(idx)

            # load corpus utterances
            self.get_utterances(example)

            # convert to torch tensors
            self.get_tensors(example)

        return example
