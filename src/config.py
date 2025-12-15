import torch


class Config:
    # Paths
    raw_audio_dir = "data/raw_audio"  # Directory containing .flac files
    spectrogram_dir = "data/mel_spectrograms"  # Output directory for spectrogram images
    metadata_file = "data/processed_metadata.json"
    protocol_file = "data/trial_metadata.txt"  # ASVspoof protocol file with labels
    model_dir = "models"
    report_file = model_dir + "/project_report.json"

    # Audio Processing
    sample_rate = 16000  # Hz
    duration = 4.0  # seconds
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    fmin = 20  # Hz
    fmax = 8000  # Hz

    # MFCC Features (for RNN)
    n_mfcc = 13
    use_mfcc_deltas = True  # Include delta and delta-delta features

    # Data Loading
    batch_size = 32
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    random_state = 42  # For reproducible splits
    num_workers = 2
    pin_memory = True
    img_size = 224
    img_normalize_mean = [0.485]
    img_normalize_std = [0.229]

    # Data Augmentation
    use_augmentation = True
    time_mask_param = 30  # Max frames to mask in time
    freq_mask_param = 20  # Max bins to mask in frequency
    noise_factor = 0.005  # Gaussian noise amplitude

    # Class Weights (for imbalanced dataset: ~90% spoof, ~10% bonafide)
    # Higher weight for minority class (bonafide)
    class_weights = torch.tensor([5.0, 0.56])  # [bonafide, spoof]

    # Model Architecture
    num_classes = 2  # real vs fake
    in_channels = 1  # grayscale spectrograms
    dropout_rate = 0.3

    # CNN
    cnn_channels = [64, 128, 256, 512]

    # Vision Transformer
    vit_patch_size = 16
    vit_embed_dim = 768
    vit_depth = 6
    vit_num_heads = 8
    vit_mlp_ratio = 4.0
    vit_dropout = 0.1

    # RNN (with MFCC input)
    rnn_hidden_size = 256
    rnn_num_layers = 2
    rnn_input_size = 39  # n_mfcc * 3 (static + delta + delta-delta) if use_mfcc_deltas

    # Training
    learning_rate = 0.001  # For CNN/RNN
    vit_learning_rate = 1e-4  # For transformer
    weight_decay = 1e-4
    epochs = 50
    patience_limit = 10  # early stopping patience
    scheduler_patience = 5
    scheduler_factor = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def mfcc_feature_dim(self):
        """Returns the MFCC feature dimension based on config."""
        return self.n_mfcc * 3 if self.use_mfcc_deltas else self.n_mfcc


config = Config()
