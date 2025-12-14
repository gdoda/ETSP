import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import os

from src.config import config
from src.audio_processor import AudioProcessor


class SpectrogramDataset(Dataset):
    """Dataset for CNN/ViT models using spectrogram images."""

    def __init__(self, data_list, is_training=True):
        self.data_list = data_list
        self.is_training = is_training

        # Base transforms
        base_transforms = [
            transforms.Resize((config.img_size, config.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=config.img_normalize_mean,
                std=config.img_normalize_std,
            ),
        ]

        # Add augmentation for training
        if is_training and config.use_augmentation:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((config.img_size, config.img_size)),
                    transforms.ToTensor(),
                    # Time masking (horizontal stripe)
                    transforms.RandomErasing(
                        p=0.5,
                        scale=(0.02, 0.1),
                        ratio=(0.3, 0.5),
                        value=0,
                    ),
                    # Frequency masking (vertical stripe)
                    transforms.RandomErasing(
                        p=0.5,
                        scale=(0.02, 0.1),
                        ratio=(2.0, 3.3),
                        value=0,
                    ),
                    transforms.Normalize(
                        mean=config.img_normalize_mean,
                        std=config.img_normalize_std,
                    ),
                ]
            )
        else:
            self.transform = transforms.Compose(base_transforms)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        label = item["label"]

        try:
            img = Image.open(item["file_path"]).convert("L")
            img = self.transform(img)
            return img, torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading image {idx}: {e}")
            dummy_img = torch.zeros(config.in_channels, config.img_size, config.img_size)
            return dummy_img, torch.tensor(0, dtype=torch.long)


class MFCCDataset(Dataset):
    """Dataset for RNN models using MFCC features."""

    def __init__(self, data_list, is_training=True):
        self.data_list = data_list
        self.is_training = is_training
        self.target_length = int(config.sample_rate * config.duration)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        label = item["label"]

        try:
            # Get audio path
            if "raw_audio_path" in item:
                audio_path = item["raw_audio_path"]
            else:
                original_name = item.get("original_name", "")
                if not original_name:
                    original_name = os.path.splitext(os.path.basename(item["file_path"]))[0] + ".flac"
                audio_path = AudioProcessor.get_raw_audio_path(original_name)

            # Load audio
            audio = AudioProcessor.load_audio(audio_path, self.target_length)
            if audio is None:
                raise ValueError(f"Failed to load audio: {audio_path}")

            # Apply waveform augmentation during training
            if self.is_training and config.use_augmentation:
                audio = self._augment_waveform(audio)

            # Extract MFCC features
            mfcc = AudioProcessor.extract_mfcc(
                audio, include_deltas=config.use_mfcc_deltas
            )

            # Convert to tensor: (n_features, n_frames) -> transpose for RNN (n_frames, n_features)
            mfcc_tensor = torch.FloatTensor(mfcc).transpose(0, 1)

            return mfcc_tensor, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            print(f"Error loading MFCC for item {idx}: {e}")
            # Return dummy data
            n_frames = self.target_length // config.hop_length + 1
            dummy_mfcc = torch.zeros(n_frames, config.mfcc_feature_dim)
            return dummy_mfcc, torch.tensor(0, dtype=torch.long)

    def _augment_waveform(self, audio):
        """Apply waveform-level augmentation."""
        # Add Gaussian noise (30% probability)
        if np.random.random() < 0.3:
            noise = np.random.randn(len(audio)) * config.noise_factor
            audio = audio + noise

        # Volume scaling (30% probability)
        if np.random.random() < 0.3:
            scale = np.random.uniform(0.8, 1.2)
            audio = audio * scale

        # Time shift (30% probability)
        if np.random.random() < 0.3:
            shift_max = int(0.1 * len(audio))
            shift = np.random.randint(-shift_max, shift_max)
            audio = np.roll(audio, shift)

        return audio


class AudioDataLoader:
    """Handles data splitting and DataLoader creation."""

    def __init__(self):
        pass

    def create_data_loaders(self, data_list, load_raw_audio=False):
        """
        Create train/val/test DataLoaders with stratified splitting.

        Args:
            data_list: List of dicts with 'file_path', 'raw_audio_path', 'label'
            load_raw_audio: If True, use MFCCDataset for RNN; else use SpectrogramDataset

        Returns:
            train_loader, val_loader, test_loader
        """
        # Extract labels for stratification
        labels = [item["label"] for item in data_list]

        # First split: train vs (val + test)
        train_data, temp_data, train_labels, temp_labels = train_test_split(
            data_list,
            labels,
            test_size=(config.val_ratio + config.test_ratio),
            stratify=labels,
            random_state=config.random_state,
        )

        # Second split: val vs test
        val_ratio_adjusted = config.val_ratio / (config.val_ratio + config.test_ratio)
        val_data, test_data, _, _ = train_test_split(
            temp_data,
            temp_labels,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_labels,
            random_state=config.random_state,
        )

        # Print split info
        data_type = "MFCC (RNN)" if load_raw_audio else "Spectrogram (CNN/ViT)"
        print(f"\nDataset split ({data_type}):")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Val:   {len(val_data)} samples")
        print(f"  Test:  {len(test_data)} samples")

        # Print class distribution per split
        self._print_class_distribution("Train", train_data)
        self._print_class_distribution("Val", val_data)
        self._print_class_distribution("Test", test_data)

        # Choose dataset class
        if load_raw_audio:
            DatasetClass = MFCCDataset
        else:
            DatasetClass = SpectrogramDataset

        # Create datasets
        train_dataset = DatasetClass(train_data, is_training=True)
        val_dataset = DatasetClass(val_data, is_training=False)
        test_dataset = DatasetClass(test_data, is_training=False)

        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=min(config.batch_size, len(train_data)),
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=min(config.batch_size, len(val_data)),
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=min(config.batch_size, len(test_data)),
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )

        return train_loader, val_loader, test_loader

    def _print_class_distribution(self, split_name, data):
        """Print class distribution for a data split."""
        bonafide = sum(1 for d in data if d["label"] == 0)
        spoof = sum(1 for d in data if d["label"] == 1)
        total = len(data)
        print(f"    {split_name}: bonafide={bonafide} ({100*bonafide/total:.1f}%), spoof={spoof} ({100*spoof/total:.1f}%)")
