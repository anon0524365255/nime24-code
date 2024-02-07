"""
Datasets for training and testing.
"""
import logging
import os
from pathlib import Path
from typing import List
from typing import Optional
from typing import Union

import lightning as L
import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader

from torchdrum.np import OnsetFrames

# Setup logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(level=os.environ.get("LOGLEVEL", "INFO"))


class FeatureDifferenceDataset(torch.utils.data.Dataset):
    """
    A dataset that generates random feature differences by sampling a polynomial

    """

    def __init__(
        self,
        size: int,  # Number of samples in the dataset
        poly: List[np.array],  # Polynomial coefficients for the output feature
        reference: Union[
            float, List[float]
        ] = 0.5,  # Reference point to calculate difference from
        mapping: List[List[int]] = None,  # Mapping from input to output features
        return_norm: bool = False,  # Whether to return the feature norm
    ):
        super().__init__()

        # Should have received a list of polynomial coefficients
        assert isinstance(poly, list)
        assert all(isinstance(p, np.ndarray) for p in poly)

        self.size = size
        self.poly = poly
        self.n_features = len(poly)
        if mapping is None:
            mapping = list([i] for i in range(self.n_features))
        self.mapping = mapping
        self.in_features = len(mapping)
        self.return_norm = return_norm

        # Confirm mapping is valid
        output_features = set(range(self.n_features))
        output_features_mapped = list()
        for m in mapping:
            assert isinstance(m, list)
            output_features_mapped.extend(m)

        assert len(output_features_mapped) == len(output_features)
        assert output_features == set(output_features_mapped)

        # Calculate the reference feature values
        if isinstance(reference, float):
            reference = [reference] * len(poly)
        self.ref_feature = torch.tensor(
            [np.polyval(p, r) for p, r in zip(poly, reference)]
        )
        self.generator = torch.Generator()
        self.norm = self.get_feature_norm()

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        self.generator.manual_seed(idx)

        # Sample from uniform distribution, which defines the input feature
        rand = torch.rand(self.in_features, generator=self.generator)

        # Calculate the output feature values by evaluating the polynomial
        out_val = []
        in_val = []
        for i in range(self.in_features):
            in_val.append(rand[i].item())
            for k in self.mapping[i]:
                out_val.append(np.polyval(self.poly[k], rand[i].item()))

        # Convert to tensors and calculate the difference
        in_val = torch.tensor(in_val)
        out_val = torch.tensor(out_val)
        diff = out_val - self.ref_feature

        if self.return_norm:
            return in_val.float(), diff.float(), self.norm
        else:
            return in_val.float(), diff.float()

    def get_feature_norm(self):
        """
        Returns the range of the input features
        """
        x = np.linspace(0, 1, 1000)
        range_norm = []
        for poly in self.poly:
            y = np.polyval(poly, x)
            feature_range = y.max() - y.min()
            range_norm.append(feature_range)

        feature_norm = torch.tensor(range_norm)[None, ...]
        return torch.abs(1.0 / feature_norm).float()


class IterableFeatureDifferenceDataset(torch.utils.data.IterableDataset):
    """
    Wrapper for the FeatureDifferenceDataset that makes it iterable
    """

    def __init__(
        self,
        start: int,  # Start index of dataset
        end: int,  # Stop index of dataset
        dataset: FeatureDifferenceDataset,  # FeatureDifferenceDataset to wrap
    ):
        super().__init__()
        assert end > start, "dataset only works with end >= start"
        self.start = start
        self.stop = end
        self.current = start
        self.dataset = dataset
        assert isinstance(dataset, FeatureDifferenceDataset)

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "dataset only works with num_workers=0"
        self.current = self.start
        return self

    def __next__(self):
        if self.current >= self.stop:
            raise StopIteration
        else:
            self.current += 1
            return self.dataset[self.current - 1]


def create_feature_dataset(
    input_features: List[
        np.array
    ],  # List of input features to generate the dataset from
    size: int,  # Number of samples in the dataset
    poly_order: int = 3,  # Order of the polynomial to fit to the input features
    reference: float = 0.5,  # Reference point to calculate difference from
    mapping: List[List[int]] = None,  # Mapping from input to output features
    return_norm: bool = False,  # Whether to return the feature norm
    sort_idx: int = None,  # Index of feature to sort by
):
    """
    Helper function to generate a feature difference dataset from a
    numpy array of features
    """
    assert isinstance(input_features, list)
    assert all(isinstance(f, np.ndarray) for f in input_features)

    # Optionally sort the input features based on a particular feature
    if sort_idx is not None:
        argsort = np.argsort(input_features[sort_idx])
    else:
        argsort = np.arange(len(input_features[0]))

    poly = []
    for f in input_features:
        x = np.linspace(0, 1, len(f))
        p = np.polyfit(x, f[argsort], poly_order)
        poly.append(p)

    return FeatureDifferenceDataset(
        size, poly, reference=reference, mapping=mapping, return_norm=return_norm
    )


class FeatureDifferenceDataModule(L.LightningDataModule):
    """
    A LightningDataModule for the FeatureDifferenceDataset
    """

    def __init__(
        self,
        audio_path: Union[Path, str],  # Path to an audio file
        feature: torch.nn.Module,  # A feature extractor
        train_start: int = 0,  # Start index of the training dataset
        train_end: int = 99999,  # Stop index of the training dataset
        val_start: int = 100000,  # Start index of the validation dataset
        val_end: int = 100999,  # Stop index of the validation dataset
        test_start: int = 101000,  # Start index of the test dataset
        test_end: int = 101999,  # Stop index of the test dataset
        batch_size: int = 64,  # Batch size
        mapping: List[List[int]] = None,  # Mapping from input to output features
        return_norm: bool = False,  # Whether to return the feature norm
        sort_idx: int = None,  # Index of feature to sort by
    ):
        super().__init__()
        self.audio_path = Path(audio_path)
        self.feature = feature
        self.train_start = train_start
        self.train_end = train_end
        self.val_start = val_start
        self.val_end = val_end
        self.test_start = test_start
        self.test_end = test_end
        self.batch_size = batch_size
        self.mapping = mapping
        self.return_norm = return_norm
        self.sort_idx = sort_idx

    def prepare_data(self) -> None:
        """
        Load the audio file and prepare the dataset
        """
        # Load the audio file
        audio, sr = torchaudio.load(self.audio_path)

        # Calculate the onset frames
        onset_frames = OnsetFrames(
            sr,
            frame_size=sr,
            on_thresh=10.0,
            wait=20000,
            backtrack=16,
            overlap_buffer=1024,
        )
        frames = onset_frames(audio)
        frames = torch.from_numpy(frames).float()

        # Check that onsets were detected
        if len(frames) == 0:
            raise RuntimeError("No onsets detected in the audio file.")
        else:
            log.info(f"Detected {len(frames)} onsets.")

        # Feature Extraction
        features = self.feature(frames)
        features = torch.split(features, 1, dim=-1)

        # TODO: why are there gradients here?
        features = [f.squeeze(-1).detach().numpy() for f in features]

        try:
            self.dataset = create_feature_dataset(
                features,
                0,
                poly_order=3,
                mapping=self.mapping,
                return_norm=self.return_norm,
                sort_idx=self.sort_idx,
            )
        except np.linalg.LinAlgError as e:
            raise RuntimeError(
                "Creating the dataset failed. Check that the input audio file. "
                f"Unabled to fit polynomial to features. Resulted in error: {e}"
            )

    def setup(self, stage: str):
        """
        Assign train/val/test datasets for use in dataloaders.

        Args:
            stage: Current stage (fit, validate, test)
        """
        assert hasattr(self, "dataset"), "Must call prepare_data() first"
        if stage == "fit":
            self.train_dataset = IterableFeatureDifferenceDataset(
                self.train_start, self.train_end, self.dataset
            )
            self.val_dataset = IterableFeatureDifferenceDataset(
                self.val_start, self.val_end, self.dataset
            )
        elif stage == "validate":
            self.train_dataset = IterableFeatureDifferenceDataset(
                self.train_start, self.train_end, self.dataset
            )
            self.val_dataset = IterableFeatureDifferenceDataset(
                self.val_start, self.val_end, self.dataset
            )
        elif stage == "test":
            self.test_dataset = IterableFeatureDifferenceDataset(
                self.test_start, self.test_end, self.dataset
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
        )


class TimbreIntervalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        size: int,  # Number of samples in the dataset
        target_diff: torch.Tensor,  # Target difference to match
        norm: torch.Tensor = None,  # Feature norm
        reference_audio: torch.Tensor = None,  # Reference audio
        target_audio: torch.Tensor = None,  # Target audio
        sample_rate: int = 44100,  # Sample rate of the audio
    ):
        super().__init__()
        self.size = size
        self.target_diff = target_diff
        self.norm = norm
        self.reference_audio = reference_audio
        self.target_audio = target_audio
        self.sample_rate = sample_rate

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if self.norm is None:
            return self.target_diff

        return self.target_diff, self.norm


class TimbreIntervalDataModule(L.LightningDataModule):
    """
    A LightningDataModule for the FeatureDifferenceDataset
    """

    def __init__(
        self,
        audio_path: Union[Path, str],  # Path to an audio file
        feature: torch.nn.Module,  # A feature extractor
        target_idx: float = 1.0,  # Index of the target frames (1.0 = last, 0.0 = first)
        train_start: int = 0,  # Start index of the training dataset
        num_train: int = 100000,  # Number of training samples
        num_val: int = 1000,  # Number of validation samples
        num_test: int = 1000,  # Number of test samples
        return_norm: bool = False,  # Whether to return the feature norm
    ):
        super().__init__()
        self.audio_path = Path(audio_path)
        self.feature = feature
        self.target_idx = target_idx
        self.train_start = train_start
        self.num_train = num_train
        self.num_val = num_val
        self.num_test = num_test
        self.return_norm = return_norm

    def prepare_data(self) -> None:
        """
        Load the audio file and prepare the dataset
        """
        # Load the audio file
        audio, sr = torchaudio.load(self.audio_path)
        self.sample_rate = sr

        # Calculate the onset frames
        onset_frames = OnsetFrames(
            sr,
            frame_size=sr,
            on_thresh=10.0,
            wait=10000,
            backtrack=16,
            overlap_buffer=1024,
        )
        frames = onset_frames(audio)
        frames = torch.from_numpy(frames).float()

        # Check that onsets were detected
        if len(frames) == 0:
            raise RuntimeError("No onsets detected in the audio file.")
        else:
            log.info(f"Detected {len(frames)} onsets.")

        ref_idx = len(frames) // 2
        target_idx = int(self.target_idx * (len(frames) - 1))
        assert ref_idx != target_idx, "Reference and target indices must be different"

        # Extract the reference and target frames
        self.reference_frame = frames[ref_idx : ref_idx + 1]
        self.target_frame = frames[target_idx : target_idx + 1]

        frames = torch.cat([self.reference_frame, self.target_frame], dim=0)
        features = self.feature(frames)

        self.diff = features[1] - features[0]
        self.norm = None

    def setup(self, stage: str):
        """
        Assign train/val/test datasets for use in dataloaders.

        Args:
            stage: Current stage (fit, validate, test)
        """
        assert hasattr(self, "diff"), "Must call prepare_data() first"
        if stage == "fit":
            self.train_dataset = TimbreIntervalDataset(
                self.num_train,
                self.diff,
                self.norm,
                reference_audio=self.reference_frame,
                target_audio=self.target_frame,
                sample_rate=self.sample_rate,
            )
        else:
            raise NotImplementedError("Only fit is supported")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=1,
            num_workers=0,
        )


class OnsetFeatureDataset(torch.utils.data.Dataset):
    """
    Dataset that returns pairs of onset features with full features
    """

    def __init__(
        self,
        onset_features: torch.Tensor,  # Onset features
        full_features: torch.Tensor,  # Full features
        weight: Optional[torch.Tensor] = None,  # Feature weighting
        onset_ref: Optional[torch.Tensor] = None,  # Onset feature values for reference
    ):
        super().__init__()
        self.onset_features = onset_features
        self.full_features = full_features
        assert self.onset_features.shape[0] == self.full_features.shape[0]
        self.size = self.onset_features.shape[0]
        self.weight = weight
        self.onset_ref = onset_ref

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        onset_features = self.onset_features[idx]
        if self.onset_ref is not None:
            onset_features = self.onset_features[idx] - self.onset_ref

        if self.weight is None:
            return onset_features, self.full_features[idx]

        return onset_features, self.full_features[idx], self.weight


class OnsetFeatureDataModule(L.LightningDataModule):
    """
    A LightningDataModule for datasets with onset features
    """

    def __init__(
        self,
        audio_path: Union[Path, str],  # Path to an audio file or directory
        feature: torch.nn.Module,  # A feature extractor
        onset_feature: torch.nn.Module,  # A feature extractor for short onsets
        sample_rate: int = 48000,  # Sample rate to compute features at
        batch_size: int = 64,  # Batch size
        return_norm: bool = False,  # Whether to return the feature norm
        center_onset: bool = False,  # Whether to return reference onset features as 0
        val_split: float = 0.0,  # Fraction of data to use for validation
        test_split: float = 0.0,  # Fraction of data to use from testing
        data_seed: int = 0,  # Seed for random data splits
    ):
        super().__init__()
        self.audio_path = Path(audio_path)
        self.feature = feature
        self.onset_feature = onset_feature
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.return_norm = return_norm
        self.center_onset = center_onset
        self.val_split = val_split
        self.test_split = test_split
        self.data_seed = data_seed

    def prepare_data(self) -> None:
        """
        Load the audio file and prepare the dataset
        """
        # Calculate the onset frames
        onset_frames = OnsetFrames(
            self.sample_rate,
            frame_size=self.sample_rate,
            on_thresh=10.0,
            wait=10000,
            backtrack=16,
            overlap_buffer=1024,
        )

        # Load audio files from a directory
        if self.audio_path.is_dir():
            audio_files = list(self.audio_path.glob("*.wav"))
            assert len(audio_files) > 0, "No audio files found in directory"
            log.info(f"Found {len(audio_files)} audio files.")

            audio = []
            for f in audio_files:
                x, sr = torchaudio.load(f)

                # Resample if necessary
                if sr != self.sample_rate:
                    x = torchaudio.transforms.Resample(sr, self.sample_rate)(x)

                # Onset detection and frame extraction to ensure all audio
                # is the same length and is aligned at an onset
                frames = onset_frames(x)
                frames = torch.from_numpy(frames).float()
                audio.append(frames)

            audio = torch.cat(audio, dim=0)
            log.info(f"{len(audio)} samples after onset detection.")

        elif self.audio_path.is_file and self.audio_path.suffix == ".wav":
            x, sr = torchaudio.load(self.audio_path)

            # Resample if necessary
            if sr != self.sample_rate:
                x = torchaudio.transforms.Resample(sr, self.sample_rate)(x)

            # Onset detection and frame extraction
            audio = onset_frames(x)
            audio = torch.from_numpy(audio).float()

            log.info(f"Found {len(audio)} samples.")

        else:
            raise RuntimeError("Invalid audio path")

        # Cache audio
        self.audio = audio

        # Compute full features
        self.full_features = self.feature(audio)
        loudsort = torch.argsort(self.full_features[:, 0], descending=True)
        idx = int(len(loudsort) * 0.5)
        idx = loudsort[idx]

        # Cache the index of the reference sample
        self.ref_idx = idx

        # Compute the difference between the features of each audio and the centroid
        self.diff = self.full_features - self.full_features[idx]
        assert torch.allclose(self.diff[idx], torch.zeros_like(self.diff[idx]))

        # Create a per feature weighting
        self.norm = torch.max(self.diff, dim=0)[0] - torch.min(self.diff, dim=0)[0]
        self.norm = torch.abs(1.0 / self.norm).float()

        # Compute onset features for each sample
        self.onset_features = self.onset_feature(audio)

        # Normalize onset features so each feature is in the range [0, 1]
        self.onset_features = self.onset_features - self.onset_features.min(dim=0)[0]
        self.onset_features = self.onset_features / self.onset_features.max(dim=0)[0]
        assert torch.all(self.onset_features >= 0.0)
        assert torch.all(self.onset_features <= 1.0)

        # Split the training data into train and test sets
        if self.test_split > 0.0:
            self.train_ids, self.test_ids = self.split_data(
                loudsort, self.ref_idx, self.test_split
            )
        else:
            self.train_ids = loudsort

        # Split the remaing data into train and validation sets
        if self.val_split > 0.0:
            self.train_ids, self.val_ids = self.split_data(
                self.train_ids, self.ref_idx, self.val_split
            )

        # Log the number of samples in each set
        log.info(f"Training samples: {len(self.train_ids)}")
        if hasattr(self, "val_ids"):
            log.info(f"Validation samples: {len(self.val_ids)}")
        if hasattr(self, "test_ids"):
            log.info(f"Test samples: {len(self.test_ids)}")

    def split_data(self, ids: torch.Tensor, ref_idx: int, split: float):
        """
        Select a subset of the data for validation
        """
        assert split > 0.0 and split < 1.0

        # Chunk the data into number of groups equal to the numbe of validation samples
        # and then select a random sample from each chunk.
        chunk_size = int(len(ids) * split) + 1
        chunks = torch.chunk(ids, chunk_size)

        train_ids = []
        val_ids = []

        g = torch.Generator()
        g.manual_seed(self.data_seed)
        for chunk in chunks:
            idx = torch.randint(0, len(chunk), (1,), generator=g).item()
            # Ensure the validation sample is not the reference sample
            if chunk[idx] == ref_idx:
                idx = (idx + 1) % len(chunk)

            val_ids.append(chunk[idx].item())
            train_ids.extend(chunk[chunk != chunk[idx]].tolist())

        assert len(train_ids) + len(val_ids) == len(ids)
        assert len(set(train_ids).intersection(set(val_ids))) == 0
        return torch.tensor(train_ids), torch.tensor(val_ids)

    def setup(self, stage: str):
        """
        Assign train/val/test datasets for use in dataloaders.

        Args:
            stage: Current stage (fit, validate, test)
        """
        assert hasattr(self, "onset_features"), "Must call prepare_data() first"
        assert hasattr(self, "full_features"), "Must call prepare_data() first"

        onset_feature_ref = None
        if self.center_onset:
            onset_feature_ref = self.onset_features[self.ref_idx]

        norm = self.norm if self.return_norm else None
        if stage == "fit":
            self.train_dataset = OnsetFeatureDataset(
                self.onset_features[self.train_ids],
                self.diff[self.train_ids],
                norm,
                onset_ref=onset_feature_ref,
            )
            if hasattr(self, "val_ids"):
                self.val_dataset = OnsetFeatureDataset(
                    self.onset_features[self.val_ids],
                    self.diff[self.val_ids],
                    norm,
                    onset_ref=onset_feature_ref,
                )
        elif stage == "validate":
            if hasattr(self, "val_ids"):
                self.val_dataset = OnsetFeatureDataset(
                    self.onset_features[self.val_ids],
                    self.diff[self.val_ids],
                    norm,
                    onset_ref=onset_feature_ref,
                )
            else:
                raise ValueError("No validation data available")
        elif stage == "test":
            if hasattr(self, "test_ids"):
                self.test_dataset = OnsetFeatureDataset(
                    self.onset_features[self.test_ids],
                    self.diff[self.test_ids],
                    norm,
                    onset_ref=onset_feature_ref,
                )
            else:
                self.train_dataset = OnsetFeatureDataset(
                    self.onset_features,
                    self.diff,
                    norm,
                    onset_ref=onset_feature_ref,
                )
        else:
            raise NotImplementedError("Unknown stage")

    def train_dataloader(self, shuffle=True):
        batch_size = min(self.batch_size, len(self.train_dataset))
        if batch_size < self.batch_size:
            log.warning(
                f"Reducing batch size to {batch_size}, "
                "only that many samples available"
            )

        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=shuffle,
        )

    def val_dataloader(self):
        if not hasattr(self, "val_dataset"):
            return None

        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )

    def test_dataloader(self):
        if not hasattr(self, "test_dataset"):
            log.info("No test dataset available, using full dataset for testing")
            return self.train_dataloader(shuffle=False)

        log.info("Testing on the test dataset")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=0,
            shuffle=False,
        )
