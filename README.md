# Detecting AI-Generated Speech: Synthetic Voice Classification

## Requirementes

You can run this project either locally or through Google Colab by using [Colab_Run.ipynb](/Colab_Run.ipynb).

If you run this locally, make sure to download [ASVspoof 2021 dataset](https://zenodo.org/records/4837263).
Change the variable `raw_audio_dir` in [config.py](/src/config.py) to match the path where you extracted the raw audio files.

Project dependencies are provided in [pyproject.toml](pyproject.toml). We use `uv` to manage dependencies.

## Directory structure

```bash
.
├── README.md
├── data
│   ├── mel_spectrograms
│   │       └── ***.png
│   └── raw_audio
│           └── ***.flac
├── main.py
├── models
├── pyproject.toml
├── report
├── src
│   ├── audio_processor.py
│   ├── config.py
│   ├── data_loader.py
│   ├── models.py
│   └── trainer.py
└── uv.lock
```

## Review configuration and parameters in [config.py](/src/config.py)

## Run the complete project

You can run the project locally as

```bash
source .venv/bin/activate
python main.py
```

Or on Google Colab with the provided [notebook](/Colab_Run.ipynb).

### Monitor training progress (of RNN, CNN and ViT training)
