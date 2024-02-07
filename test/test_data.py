from pathlib import Path

import numpy as np
import pytest
import torch
import torchaudio

from torchdrum.data import create_feature_dataset
from torchdrum.data import FeatureDifferenceDataModule
from torchdrum.data import FeatureDifferenceDataset
from torchdrum.data import IterableFeatureDifferenceDataset
from torchdrum.data import OnsetFeatureDataModule


def test_feature_diff_dataset_init():
    poly = [np.array([1, 2, 3])]
    dataset = FeatureDifferenceDataset(100, poly, reference=0.5)
    assert len(dataset) == 100
    assert np.allclose(dataset.poly[0], [1, 2, 3])
    assert dataset.ref_feature == np.polyval([1, 2, 3], 0.5)


def test_feature_diff_dataset_single():
    poly = [np.array([0, 1, 1])]
    dataset = FeatureDifferenceDataset(1, poly)
    x, diff = dataset[0]

    # Expected values, 0.5 is the default reference point
    y_hat = np.polyval(poly[0], x.item())
    y = np.polyval(poly[0], 0.5)

    assert diff.item() == y_hat - y


def test_feature_diff_dataset_multi():
    poly = [np.array([1, 1]), np.array([1, 1, 2])]
    dataset = FeatureDifferenceDataset(1, poly)
    x, diff = dataset[0]

    assert x.shape == (2,)
    assert diff.shape == (2,)

    y_hat = np.polyval(poly[0], x[0].item())
    y = np.polyval(poly[0], 0.5)
    assert np.allclose(diff[0].item(), y_hat - y)

    y_hat = np.polyval(poly[1], x[1].item())
    y = np.polyval(poly[1], 0.5)
    assert np.allclose(diff[1].item(), y_hat - y)


def test_create_feature_dataset_multi():
    features_1 = np.array([0, 1, 2, 3, 4])
    features_2 = np.array([0, 1, 4, 9, 16])
    features = [features_1, features_2]

    dataset = create_feature_dataset(features, 100, poly_order=2)
    assert len(dataset) == 100

    x, diff = dataset[0]
    assert x.shape == (2,)
    assert diff.shape == (2,)


def test_iterable_feature_dataset():
    poly = [np.array([1, 1]), np.array([1, 1, 2]), np.array([1, 1])]
    feature_dataset = FeatureDifferenceDataset(0, poly)
    dataset = IterableFeatureDifferenceDataset(0, 10, feature_dataset)
    i = 0
    for batch in dataset:
        assert batch[0].shape == (3,)
        assert batch[1].shape == (3,)
        i += 1

    assert i == 10


def test_data_module_prepare_data(mocker):
    # Create a dummy audio clip with some onsets
    test_audio = torch.rand(1, 44100)
    amp_env = torch.linspace(1, 0, 44100)
    test_audio = test_audio * amp_env
    test_audio = torch.cat((test_audio, test_audio, test_audio), dim=-1)

    load_audio = mocker.patch("torchaudio.load", return_value=(test_audio, 44100))

    feature = torch.nn.Linear(44100, 3)
    data_module = FeatureDifferenceDataModule(Path("test.wav"), feature)
    data_module.prepare_data()

    load_audio.assert_called_once_with(Path("test.wav"))


def test_onset_feature_datamodule_init(tmp_path):
    audio_path = tmp_path / "test.wav"
    feature = torch.nn.Linear(44100, 3)
    onset_feature = torch.nn.Linear(44100, 1)
    data_module = OnsetFeatureDataModule(audio_path, feature, onset_feature)
    assert data_module.audio_path == audio_path
    assert data_module.feature == feature


@pytest.fixture
def audio_folder(tmp_path):
    """Create a folder with 8 noisy audio files"""
    for i in range(8):
        audio_path = tmp_path / f"{i}.wav"
        test_audio = torch.rand(1, 44100) * 2.0 - 1.0
        amp_env = torch.linspace(1, 0, 44100)
        test_audio = test_audio * amp_env
        torchaudio.save(audio_path, test_audio, 44100)

    yield tmp_path

    # Clean up
    for p in tmp_path.glob("*.wav"):
        p.unlink()


def test_onset_feature_datamodule_prepare(audio_folder):
    feature = torch.nn.Linear(44100, 6)
    onset_feature = torch.nn.Linear(44100, 3)
    data_module = OnsetFeatureDataModule(
        audio_folder, feature, onset_feature, sample_rate=44100
    )
    data_module.prepare_data()

    assert hasattr(data_module, "full_features")
    assert hasattr(data_module, "onset_features")
    assert data_module.full_features.shape == (8, 6)
    assert data_module.onset_features.shape == (8, 3)


def test_onset_feature_datamodule_setup(audio_folder):
    feature = torch.nn.Linear(44100, 6)
    onset_feature = torch.nn.Linear(44100, 3)
    data_module = OnsetFeatureDataModule(
        audio_folder, feature, onset_feature, sample_rate=44100
    )
    data_module.prepare_data()
    data_module.setup("fit")

    assert hasattr(data_module, "dataset")
    assert len(data_module.dataset) == 8
    o, f, w = data_module.dataset[0]
    assert o.shape == (3,)
    assert f.shape == (6,)
    assert w.shape == f.shape
