# torchdrum

Differentiable drum synthesizer and PyTorch training code to accompany the 2024 NIME submission: *Real-time Timbre Remapping with Differentiable DSP*

**Note to reviewers:** This repo is in an in-progress state. Upon release this repo will contain all code and data necassary to reproduce the experiments in the submitted paper. We also plan to include a link to a Google Colab notebook that will allow individuals to train their own models to load into the accompanying plugin.

## Installation

```bash
pip install -e ".[dev]"
```

## Snare Drum Data

```bash
mkdir audio
cd audio
wget https://pub-814e66019388451395cf43c0b6f10300.r2.dev/sdss_filtered.zip
unzip sdss_filtered.zip
```

## Carson Performance Data

```bash
mkdir -p audio
cd audio
wget https://pub-814e66019388451395cf43c0b6f10300.r2.dev/carson.zip
unzip carson.zip
```
