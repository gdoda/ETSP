# Detecting AI-Generated Speech: Synthetic Voice Classification

## Overview

The pipeline implements a binary classification system to distinguish between **bonafide** (real human speech) and **spoof** (AI-generated/synthesized speech) audio samples. It trains and evaluates three different deep learning architectures:

| Model   | Input Type                  | Architecture                                        |
| ------- | --------------------------- | --------------------------------------------------- |
| **CNN** | Mel-spectrograms (images)   | 4-layer Convolutional Neural Network with attention |
| **ViT** | Mel-spectrograms (images)   | Vision Transformer with 6 encoder layers            |
| **RNN** | MFCC features (time-series) | Bidirectional LSTM with attention mechanism         |

## Requirements

You can run this project either locally or through Google Colab by using the provided Jupyter notebook: [Colab_Run.ipynb](/Colab_Run.ipynb).

If you run this locally, make sure to download [ASVspoof 2021 dataset](https://zenodo.org/records/4837263).
Change the variable `raw_audio_dir` in [config.py](/src/config.py) to match the path where you extracted the raw audio files.

Project dependencies are managed with **conda**. See [environment.yml](environment.yml) for the full list.

## Directory structure

```bash
.
├── README.md
├── Colab_Run.ipynb
├── environment.yml
├── data
│   ├── mel_spectrograms
│   │       └── ***.png
│   └── raw_audio
│           └── ***.flac
├── docs
│   └── IMPLEMENTATION.md
├── main.py
├── models
│   └── ***_best.pth
├── report
│   └── main.pdf
├── src
│   ├── audio_processor.py
│   ├── config.py
│   ├── data_loader.py
│   ├── metrics.py
│   ├── models.py
│   └── trainer.py
```

### Review configuration and parameters in [config.py](/src/config.py). Explaination of the content is available in the implementation details section: [docs/IMPLEMENTATION.md](/docs/IMPLEMENTATION.md)

## Run the complete project

### Local Setup

```bash
# Create and activate conda environment
conda env create -f environment.yml
conda activate etsp

# Run the pipeline
python main.py
```

### Google Colab

Use the provided [Colab_Run.ipynb](/Colab_Run.ipynb) notebook which handles all setup automatically.

### Monitor training progress (of RNN, CNN and ViT training)
