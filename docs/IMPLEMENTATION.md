# Synthetic Voice Detection Pipeline Documentation

This document provides detailed documentation of the machine learning pipeline for detecting AI-generated speech using the ASVspoof 2021 dataset.

## Table of Contents

1. [Pipeline Overview](#1-pipeline-overview)
2. [Architecture Diagram](#2-architecture-diagram)
3. [Data Flow](#3-data-flow)
4. [Component Details](#4-component-details)
5. [Model Architectures](#5-model-architectures)
6. [Training Process](#6-training-process)
7. [Configuration Reference](#7-configuration-reference)
8. [Google Colab Execution](#8-google-colab-execution)

---

## 1. Pipeline Overview

### High-Level Execution Flow

```
main.py
   │
   ├── 1. AudioProcessor.process_dataset()
   │      └── Convert .flac files → mel-spectrogram images (.png)
   │
   ├── 2. AudioDataLoader.create_data_loaders()
   │      ├── Image loaders (CNN, ViT) → SpectrogramDataset
   │      └── Audio loaders (RNN) → MFCCDataset
   │
   ├── 3. Train CNN Model
   │      └── ModelTrainer.train() → saves cnn_best.pth
   │
   ├── 4. Train Vision Transformer
   │      └── ModelTrainer.train() → saves vit_best.pth
   │
   ├── 5. Train RNN Model
   │      └── ModelTrainer.train() → saves rnn_best.pth
   │
   └── 6. Generate Report → project_report.json
```

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INPUT: ASVspoof 2021 Dataset                      │
│                    (raw_audio/*.flac + trial_metadata.txt)                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUDIO PROCESSOR (audio_processor.py)                │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────────────┐  │
│  │  Load Protocol  │───▶│   Load Audio     │───▶│  Create Mel-Spectrogram│  │
│  │  Labels (.txt)  │    │  (.flac → numpy) │    │                        │  │
│  └─────────────────┘    └──────────────────┘    └────────────────────────┘  │
│                                                           │                 │
│                                                           ▼                 │
│                                              ┌───────────────────────┐      │
│                                              │  Save as PNG Image    │      │
│                                              │  + Metadata JSON      │      │
│                                              └───────────────────────┘      │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA LOADER (data_loader.py)                        │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Stratified Train/Val/Test Split                   │   │
│  │                         (70% / 15% / 15%)                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                          │                              │                   │
│                          ▼                              ▼                   │
│  ┌─────────────────────────────────┐    ┌─────────────────────────────────┐ │
│  │     SpectrogramDataset          │    │        MFCCDataset              │ │
│  │  (for CNN and ViT)              │    │      (for RNN)                  │ │
│  │                                 │    │                                 │ │
│  │  • Load PNG images              │    │  • Load raw .flac audio         │ │
│  │  • Resize to 224×224            │    │  • Extract MFCC features        │ │
│  │  • Apply augmentation           │    │  • Include delta & delta-delta  │ │
│  │    - Time masking               │    │  • Apply waveform augmentation  │ │
│  │    - Frequency masking          │    │    - Gaussian noise             │ │
│  │  • Normalize                    │    │    - Volume scaling             │ │
│  │                                 │    │    - Time shift                 │ │
│  └─────────────────────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                          │                              │
                          ▼                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODELS (models.py)                             │
│                                                                             │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────────────┐   │
│  │    CustomCNN     │  │ VisionTransformer│  │       AudioRNN           │   │
│  │                  │  │                  │  │                          │   │
│  │Input: (1,224,224)│  │Input: (1,224,224)│  │ Input: (n_frames, 39)    │   │
│  │                  │  │                  │  │                          │   │
│  │ • 4 Conv blocks  │  │ • Patch embed    │  │ • Input projection       │   │
│  │ • BatchNorm      │  │ • 6 Transformer  │  │ • 2-layer BiLSTM         │   │
│  │ • MaxPool        │  │   encoder layers │  │ • Attention mechanism    │   │
│  │ • Attention      │  │ • CLS token      │  │ • Classification head    │   │
│  │ • Global pool    │  │ • Pos embedding  │  │                          │   │
│  │                  │  │                  │  │                          │   │
│  │ Output: 2 classes│  │ Output: 2 classes│  │ Output: 2 classes        │   │
│  └──────────────────┘  └──────────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINER (trainer.py)                              │
│                                                                             │
│  • CrossEntropyLoss with class weights (handles 90/10 imbalance)            │
│  • Adam optimizer (lr=0.001, weight_decay=1e-4)                             │
│  • ReduceLROnPlateau scheduler                                              │
│  • Early stopping (patience=10 epochs)                                      │
│  • Metrics: Accuracy, Precision, Recall, F1-Score                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               OUTPUT                                        │
│                                                                             │
│  models/                      report/                                       │
│  ├── cnn_best.pth            └── project_report.json                        │
│  ├── vit_best.pth                                                           │
│  └── rnn_best.pth                                                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Data Flow

### 3.1 Input Data Structure

The pipeline expects the ASVspoof 2021 LA evaluation dataset:

```
data/
├── raw_audio/
│   ├── LA_E_1000001.flac
│   └── LA_E_1000002.flac
└── trial_metadata.txt
```

**trial_metadata.txt format:**

```
speaker_id  audio_filename  codec  transmission  attack_type  label  trimming  subset
LA_0001     LA_E_1000001    -      -             -            bonafide  -       eval
LA_0002     LA_E_1000002    -      -             A07          spoof     -       eval
```

### 3.2 Label Mapping

| Label | Class    | Description                     |
| ----- | -------- | ------------------------------- |
| 0     | bonafide | Real human speech               |
| 1     | spoof    | AI-generated/synthesized speech |

**Class Distribution:** The dataset is highly imbalanced (~90% spoof, ~10% bonafide). This is handled through:

- Class weights in loss function: `[5.0, 0.56]` (bonafide, spoof)
- Stratified train/val/test splitting

### 3.3 Processed Data Output

After processing, the pipeline creates:

```
data/
├── mel_spectrograms/
│   ├── LA_E_1000001.png
│   ├── LA_E_1000002.png
│   └── ...
└── processed_metadata.json
```

**processed_metadata.json structure:**

```json
[
  {
    "file_path": "data/mel_spectrograms/LA_E_1000001.png",
    "raw_audio_path": "data/raw_audio/LA_E_1000001.flac",
    "label": 0,
    "original_name": "LA_E_1000001.flac"
  },
  ...
]
```

---

## 4. Component Details

### 4.1 AudioProcessor (`src/audio_processor.py`)

Handles all audio processing operations.

#### Key Methods:

| Method                     | Description                                                    |
| -------------------------- | -------------------------------------------------------------- |
| `load_protocol_labels()`   | Parses ASVspoof protocol file to extract ground truth labels   |
| `load_audio()`             | Loads .flac files, resamples to 16kHz, pads/trims to 4 seconds |
| `create_mel_spectrogram()` | Converts waveform to 128-band log-mel spectrogram              |
| `extract_mfcc()`           | Extracts 13 MFCCs + delta + delta-delta (39 features total)    |
| `process_dataset()`        | Main method: processes all files and saves spectrograms        |

#### Audio Processing Parameters:

```python
sample_rate = 16000      # Hz
duration = 4.0           # seconds (64,000 samples)
n_fft = 2048             # FFT window size
hop_length = 512         # Hop between frames
n_mels = 128             # Mel frequency bands
fmin = 20                # Minimum frequency (Hz)
fmax = 8000              # Maximum frequency (Hz)
n_mfcc = 13              # MFCC coefficients
```

#### Spectrogram Generation Flow:

```
Raw Audio (.flac)
       │
       ▼ torchaudio.load()
Waveform (numpy array)
       │
       ▼ Resample to 16kHz if needed
       │
       ▼ Pad/Trim to 64,000 samples
       │
       ▼ Normalize: audio / max(|audio|)
       │
       ▼ librosa.feature.melspectrogram()
Mel Spectrogram (128 bands × 126 frames)
       │
       ▼ librosa.power_to_db()
Log-Mel Spectrogram
       │
       ▼ Min-Max Normalization
       │
       ▼ matplotlib.savefig()
PNG Image (saved to disk)
```

### 4.2 AudioDataLoader (`src/data_loader.py`)

Manages dataset creation and batching.

#### Dataset Classes:

**SpectrogramDataset** (for CNN/ViT):

- Loads PNG spectrogram images
- Resizes to 224×224 pixels
- Converts to grayscale tensor (1 channel)
- Applies training augmentation:
  - Time masking (horizontal stripe erasure)
  - Frequency masking (vertical stripe erasure)
- Normalizes with mean=0.485, std=0.229

**MFCCDataset** (for RNN):

- Loads raw .flac audio on-the-fly
- Extracts MFCC features (39 dimensions)
- Applies waveform augmentation during training:
  - Gaussian noise (30% probability)
  - Volume scaling (30% probability)
  - Time shift (30% probability)
- Returns tensor of shape `(n_frames, 39)`

#### Data Splitting:

```python
train_ratio = 0.70   # 70% training
val_ratio = 0.15     # 15% validation
test_ratio = 0.15    # 15% testing
random_state = 42    # Reproducible splits
```

Uses sklearn's `train_test_split` with stratification to maintain class distribution across splits.

### 4.3 ModelTrainer (`src/trainer.py`)

Handles the training loop for all models.

#### Training Configuration:

| Parameter            | Value | Description             |
| -------------------- | ----- | ----------------------- |
| `learning_rate`      | 0.001 | Initial learning rate   |
| `weight_decay`       | 1e-4  | L2 regularization       |
| `epochs`             | 50    | Maximum training epochs |
| `patience_limit`     | 10    | Early stopping patience |
| `scheduler_patience` | 5     | LR reduction patience   |
| `scheduler_factor`   | 0.5   | LR reduction factor     |

#### Training Loop:

```
For each epoch:
    │
    ├── train_epoch()
    │   ├── Set model to train mode
    │   ├── For each batch:
    │   │   ├── Forward pass
    │   │   ├── Compute weighted CrossEntropyLoss
    │   │   ├── Backward pass
    │   │   └── Optimizer step
    │   └── Return: train_loss, train_accuracy
    │
    ├── validate_epoch()
    │   ├── Set model to eval mode
    │   ├── For each batch (no gradients):
    │   │   ├── Forward pass
    │   │   └── Compute loss and predictions
    │   └── Return: val_loss, val_accuracy, precision, recall, f1
    │
    ├── scheduler.step(val_loss)  # Reduce LR if plateau
    │
    ├── If val_accuracy > best:
    │   ├── Save model checkpoint
    │   └── Reset patience counter
    │
    └── If patience exhausted: Early stop
```

---

## 5. Model Architectures

### 5.1 CustomCNN

A 4-layer CNN with squeeze-and-excitation style attention.

```
Input: (batch, 1, 224, 224)
        │
        ▼
┌────────────────────────────────┐
│  Conv2d(1→64, 3×3) + BN + ReLU │
│  MaxPool2d(2×2)                │ → (batch, 64, 112, 112)
└────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────┐
│  Conv2d(64→128, 3×3) + BN + ReLU│
│  MaxPool2d(2×2)                 │ → (batch, 128, 56, 56)
└─────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Conv2d(128→256, 3×3) + BN + ReLU│
│  MaxPool2d(2×2)                  │ → (batch, 256, 28, 28)
└──────────────────────────────────┘
        │
        ▼
┌──────────────────────────────────┐
│  Conv2d(256→512, 3×3) + BN + ReLU│
│  MaxPool2d(2×2)                  │ → (batch, 512, 14, 14)
└──────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│  Attention Block:              │
│  AdaptiveAvgPool → Conv1×1 → σ │ → (batch, 512, 14, 14)
│  Element-wise multiply         │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│  AdaptiveAvgPool2d(1)          │ → (batch, 512, 1, 1)
│  Flatten                       │ → (batch, 512)
│  Dropout(0.3)                  │
│  Linear(512→2)                 │ → (batch, 2)
└────────────────────────────────┘
```

### 5.2 VisionTransformer (ViT)

Transformer-based architecture operating on image patches.

```
Input: (batch, 1, 224, 224)
        │
        ▼
┌─────────────────────────────────┐
│  Patch Embedding:               │
│  Conv2d(1→768, 16×16, stride=16)│ → (batch, 768, 14, 14)
│  Flatten + Transpose            │ → (batch, 196, 768)
└─────────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Prepend CLS Token            │ → (batch, 197, 768)
│  Add Positional Embedding     │
└───────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│  6× TransformerEncoderLayer:   │
│  • MultiHeadAttention (8 heads)│
│  • FFN (768→3072→768)          │
│  • LayerNorm + Residual        │
│  • Dropout (0.1)               │
│  • GELU activation             │
└────────────────────────────────┘
        │
        ▼
┌────────────────────────────────┐
│  LayerNorm                     │
│  Extract CLS token [:, 0]      │ → (batch, 768)
│  Linear(768→2)                 │ → (batch, 2)
└────────────────────────────────┘
```

**ViT Parameters:**

- Patch size: 16×16
- Number of patches: (224/16)² = 196
- Embedding dimension: 768
- Number of transformer layers: 6
- Number of attention heads: 8
- MLP ratio: 4.0 (hidden dim = 3072)

### 5.3 AudioRNN

Bidirectional LSTM with attention for time-series classification.

```
Input: (batch, n_frames, 39)  # ~126 frames for 4s audio
        │
        ▼
┌───────────────────────────────┐
│  Input Projection:            │
│  Linear(39→256) + LayerNorm   │
│  ReLU + Dropout(0.3)          │ → (batch, n_frames, 256)
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Bidirectional LSTM:          │
│  2 layers, hidden_size=256    │
│  Dropout between layers       │ → (batch, n_frames, 512)
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Attention Mechanism:         │
│  Linear(512→256) + Tanh       │
│  Linear(256→1)                │
│  Softmax over time            │ → attention_weights (batch, n_frames)
│                               │
│  Weighted sum of LSTM outputs │ → context_vector (batch, 512)
└───────────────────────────────┘
        │
        ▼
┌───────────────────────────────┐
│  Classification Head:         │
│  Dropout + Linear(512→256)    │
│  ReLU + Dropout               │
│  Linear(256→2)                │ → (batch, 2)
└───────────────────────────────┘
```

**MFCC Input Features (39 dimensions):**

- 13 static MFCC coefficients
- 13 delta (first derivative)
- 13 delta-delta (second derivative)

---

## 6. Training Process

### 6.1 Loss Function

**Weighted Cross-Entropy Loss:**

```python
class_weights = [5.0, 0.56]  # [bonafide, spoof]
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

The weights compensate for the ~90/10 class imbalance:

- Bonafide (minority): weight = 5.0
- Spoof (majority): weight = 0.56

### 6.2 Optimizer and Scheduler

```python
optimizer = Adam(lr=0.001, weight_decay=1e-4)
scheduler = ReduceLROnPlateau(patience=5, factor=0.5)
```

The scheduler reduces learning rate by half when validation loss plateaus for 5 epochs.

### 6.3 Early Stopping

Training stops if validation accuracy doesn't improve for 10 consecutive epochs.

### 6.4 Model Checkpointing

Best models are saved when validation accuracy improves:

```
models/
├── cnn_best.pth
├── vit_best.pth
└── rnn_best.pth
```

Each checkpoint contains:

- `model_state_dict`: Model weights
- `optimizer_state_dict`: Optimizer state
- `train_accuracies`: Training accuracy history
- `val_accuracies`: Validation accuracy history
- `train_losses`: Training loss history
- `val_losses`: Validation loss history

### 6.5 Evaluation Metrics

During training, the following metrics are computed:

| Metric        | Description                                    |
| ------------- | ---------------------------------------------- |
| **Accuracy**  | Overall classification accuracy                |
| **Precision** | Weighted precision across classes              |
| **Recall**    | Weighted recall across classes                 |
| **F1-Score**  | Weighted harmonic mean of precision and recall |

Additional metrics available in `metrics.py`:

- **EER (Equal Error Rate)**: Standard ASVspoof metric
- **Balanced Accuracy**: Average of per-class recalls
- **ROC AUC**: Area under ROC curve
- **Per-class Recall**: Separate recall for bonafide and spoof

---

## 7. Configuration Reference

All configuration parameters are centralized in `src/config.py`:

### Paths

| Parameter         | Default                        | Description               |
| ----------------- | ------------------------------ | ------------------------- |
| `raw_audio_dir`   | `data/raw_audio`               | Input .flac files         |
| `spectrogram_dir` | `data/mel_spectrograms`        | Output spectrogram images |
| `metadata_file`   | `data/processed_metadata.json` | Processing metadata       |
| `protocol_file`   | `data/trial_metadata.txt`      | ASVspoof labels           |
| `model_dir`       | `models`                       | Saved model checkpoints   |
| `report_file`     | `models/project_report.json`   | Final report              |

### Audio Processing

| Parameter     | Default | Description                 |
| ------------- | ------- | --------------------------- |
| `sample_rate` | 16000   | Target sample rate (Hz)     |
| `duration`    | 4.0     | Audio clip length (seconds) |
| `n_fft`       | 2048    | FFT window size             |
| `hop_length`  | 512     | Hop between frames          |
| `n_mels`      | 128     | Mel frequency bands         |
| `fmin`        | 20      | Minimum frequency (Hz)      |
| `fmax`        | 8000    | Maximum frequency (Hz)      |
| `n_mfcc`      | 13      | Number of MFCC coefficients |

### Data Loading

| Parameter     | Default | Description            |
| ------------- | ------- | ---------------------- |
| `batch_size`  | 32      | Training batch size    |
| `train_ratio` | 0.7     | Training split ratio   |
| `val_ratio`   | 0.15    | Validation split ratio |
| `test_ratio`  | 0.15    | Test split ratio       |
| `num_workers` | 2       | DataLoader workers     |
| `img_size`    | 224     | Image resize dimension |

### Data Augmentation

| Parameter          | Default | Description                 |
| ------------------ | ------- | --------------------------- |
| `use_augmentation` | True    | Enable augmentation         |
| `time_mask_param`  | 30      | Max frames for time masking |
| `freq_mask_param`  | 20      | Max bins for freq masking   |
| `noise_factor`     | 0.005   | Gaussian noise amplitude    |

### Model Architecture

| Parameter         | Default          | Description                |
| ----------------- | ---------------- | -------------------------- |
| `num_classes`     | 2                | Output classes (real/fake) |
| `dropout_rate`    | 0.3              | Dropout probability        |
| `cnn_channels`    | [64,128,256,512] | CNN layer channels         |
| `vit_patch_size`  | 16               | ViT patch dimensions       |
| `vit_embed_dim`   | 768              | ViT embedding dimension    |
| `vit_depth`       | 6                | ViT transformer layers     |
| `vit_num_heads`   | 8                | ViT attention heads        |
| `rnn_hidden_size` | 256              | LSTM hidden dimension      |
| `rnn_num_layers`  | 2                | LSTM layer count           |

### Training

| Parameter        | Default     | Description             |
| ---------------- | ----------- | ----------------------- |
| `learning_rate`  | 0.001       | Initial learning rate   |
| `weight_decay`   | 1e-4        | L2 regularization       |
| `epochs`         | 50          | Maximum epochs          |
| `patience_limit` | 10          | Early stopping patience |
| `class_weights`  | [5.0, 0.56] | Loss class weights      |

---

## 8. Google Colab Execution

### 8.1 Notebook Structure (`Colab_Run.ipynb`)

The notebook automates the complete setup and execution:

| Cell | Purpose                                                                  |
| ---- | ------------------------------------------------------------------------ |
| 1    | Mount Google Drive, install `condacolab` (triggers runtime restart)      |
| 2    | Clone GitHub repository, create conda environment from `environment.yml` |
| 3    | Download/extract dataset (MAX_FILES=10000)                               |
| 4    | Run `main.py` pipeline                                                   |

---

## 9. Output Files

After successful execution:

```
models/
├── cnn_best.pth          # Best CNN checkpoint
├── vit_best.pth          # Best ViT checkpoint
└── rnn_best.pth          # Best RNN checkpoint

report/
└── project_report.json   # Training summary with metrics

data/
├── mel_spectrograms/     # Generated spectrogram images
└── processed_metadata.json  # File-label mapping
```

**project_report.json example:**

```json
{
  "project_title": "Synthetic Voice Detection System",
  "dataset_used": "ASVspoof2021_LA_eval",
  "total_samples": 50000,
  "model_performance": {
    "CNN": {
      "best_validation_accuracy": "92.50%",
      "best_epoch": 15,
      "f1_score": "0.9180"
    },
    "Vision_Transformer": {...},
    "RNN": {...}
  }
}
```
