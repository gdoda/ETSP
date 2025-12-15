import os
import warnings
import numpy as np
import librosa
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from src.config import config

# Suppress librosa's audioread fallback warnings (PySoundFile doesn't support FLAC on all platforms)
warnings.filterwarnings("ignore", message="PySoundFile failed")
warnings.filterwarnings("ignore", message="librosa.core.audio.__audioread_load")

multiprocessing.set_start_method("spawn", force=True)


class AudioProcessor:
    def __init__(self):
        self.target_length = int(config.sample_rate * config.duration)
        self.labels_map = None  # Will be loaded from protocol file

    def load_protocol_labels(self, protocol_path=None):
        """Load ground truth labels from ASVspoof protocol file."""
        if protocol_path is None:
            protocol_path = config.protocol_file

        if not os.path.exists(protocol_path):
            print(f"Warning: Protocol file not found at {protocol_path}")
            return None

        # ASVspoof protocol format: speaker_id, audio_filename, codec, transmission, attack_type, label, trimming, subset
        column_names = [
            "speaker_id",
            "audio_filename",
            "codec",
            "transmission",
            "attack_type",
            "label",
            "trimming",
            "subset",
        ]

        df = pd.read_csv(protocol_path, sep=r"\s+", names=column_names, engine="python")

        # Create mapping: filename -> label (0=bonafide, 1=spoof)
        self.labels_map = {}
        for _, row in df.iterrows():
            filename = row["audio_filename"]
            if not filename.endswith(".flac"):
                filename = filename + ".flac"
            # Map bonafide=0, spoof=1
            self.labels_map[filename] = 0 if row["label"] == "bonafide" else 1

        print(f"Loaded {len(self.labels_map)} labels from protocol file")

        # Print class distribution
        bonafide_count = sum(1 for v in self.labels_map.values() if v == 0)
        spoof_count = sum(1 for v in self.labels_map.values() if v == 1)
        print(
            f"  Bonafide: {bonafide_count} ({100 * bonafide_count / len(self.labels_map):.1f}%)"
        )
        print(
            f"  Spoof: {spoof_count} ({100 * spoof_count / len(self.labels_map):.1f}%)"
        )

        return self.labels_map

    def get_label(self, filename):
        """Get label for a given audio filename."""
        if self.labels_map is None:
            self.load_protocol_labels()

        basename = os.path.basename(filename)
        if not basename.endswith(".flac"):
            basename = basename + ".flac"

        if self.labels_map and basename in self.labels_map:
            return self.labels_map[basename]

        # Fallback: return -1 for unknown
        return -1

    @staticmethod
    def get_raw_audio_path(filename):
        """Constructs the full path to the raw audio file."""
        return os.path.join(config.raw_audio_dir, filename)

    @staticmethod
    def load_audio(audio_path, target_length=None):
        """Loads and pads/trims audio to the configured target length."""
        if target_length is None:
            target_length = int(config.sample_rate * config.duration)

        try:
            audio, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)

            # Pad or trim to target length
            if len(audio) > target_length:
                audio = audio[:target_length]
            else:
                padding = target_length - len(audio)
                audio = np.pad(audio, (0, padding), mode="constant")

            # Normalize audio
            audio = audio / (np.max(np.abs(audio)) + 1e-8)
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None

    def create_mel_spectrogram(self, audio):
        """Converts raw audio to a log-mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=config.sample_rate,
            n_mels=config.n_mels,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            fmin=config.fmin,
            fmax=config.fmax,
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Min-max normalization
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (
            log_mel_spec.max() - log_mel_spec.min() + 1e-8
        )
        return log_mel_spec

    @staticmethod
    def extract_mfcc(audio, include_deltas=True):
        """
        Extract MFCC features from audio waveform.

        Args:
            audio: numpy array of audio samples
            include_deltas: if True, include delta and delta-delta features

        Returns:
            mfcc_features: numpy array of shape (n_features, n_frames)
                          n_features = n_mfcc * 3 if include_deltas else n_mfcc
        """
        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=config.sample_rate,
            n_mfcc=config.n_mfcc,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
        )

        if include_deltas:
            # Compute delta (first derivative)
            mfcc_delta = librosa.feature.delta(mfcc)
            # Compute delta-delta (second derivative)
            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
            # Stack: [mfcc, delta, delta-delta]
            mfcc_features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
        else:
            mfcc_features = mfcc

        # Normalize per-coefficient (across time)
        mean = mfcc_features.mean(axis=1, keepdims=True)
        std = mfcc_features.std(axis=1, keepdims=True) + 1e-8
        mfcc_features = (mfcc_features - mean) / std

        return mfcc_features

    @staticmethod
    def save_spectrogram_image(mel_spec, save_path):
        """Saves the spectrogram as an image file using PIL (much faster than matplotlib)."""
        # mel_spec is already normalized to [0, 1], scale to [0, 255]
        img_array = (mel_spec * 255).astype(np.uint8)
        # Flip vertically to match matplotlib's origin='lower' behavior
        img_array = np.flipud(img_array)
        img = Image.fromarray(img_array, mode="L")
        img.save(save_path)

    def process_dataset(self, input_dir, output_dir, num_workers=None):
        """
        Processes all .flac files in the input directory, converts them to
        spectrogram images, and returns a metadata list.

        Args:
            input_dir: Directory containing .flac files
            output_dir: Directory to save spectrogram images
            num_workers: Number of parallel workers (default: CPU count)
        """
        print("Processing ASVspoof2021_LA_eval dataset...")

        # Load labels from protocol file
        self.load_protocol_labels()

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        input_path = Path(input_dir)

        # Get all flac files
        audio_files = list(input_path.rglob("*.flac"))
        print(f"Found {len(audio_files)} audio files in eval dataset")

        # Filter files that have labels and prepare work items
        work_items = []
        skipped_no_label = 0
        for audio_file in audio_files:
            label = self.get_label(audio_file.name)
            if label == -1:
                skipped_no_label += 1
                continue
            img_path = output_path / audio_file.with_suffix(".png").name
            work_items.append(
                (str(audio_file), str(img_path), label, self.target_length)
            )

        if skipped_no_label > 0:
            print(
                f"Warning: Skipping {skipped_no_label} files without labels in protocol"
            )

        print(f"Processing {len(work_items)} files with labels...")

        # Determine number of workers
        if num_workers is None:
            num_workers = min(multiprocessing.cpu_count(), 8)
        print(f"Using {num_workers} parallel workers")

        # Process in parallel
        processed_data = []
        failed = 0

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_single_file, item): item for item in work_items
            }

            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing files"
            ):
                result = future.result()
                if result is not None:
                    processed_data.append(result)
                else:
                    failed += 1

        if failed > 0:
            print(f"Warning: {failed} files failed to process")

        print(f"Completed! Processed {len(processed_data)} files from eval dataset")

        # Print final class distribution
        bonafide = sum(1 for d in processed_data if d["label"] == 0)
        spoof = sum(1 for d in processed_data if d["label"] == 1)
        print(f"Final dataset: {bonafide} bonafide, {spoof} spoof")

        return processed_data


def _process_single_file(args):
    """
    Worker function to process a single audio file.
    Must be defined at module level for pickling in multiprocessing.
    """
    audio_path, img_path, label, target_length = args

    try:
        # Load with librosa (automatically resamples and converts to mono)
        audio, _ = librosa.load(audio_path, sr=config.sample_rate, mono=True)

        # Pad or trim to target length
        if len(audio) > target_length:
            audio = audio[:target_length]
        else:
            padding = target_length - len(audio)
            audio = np.pad(audio, (0, padding), mode="constant")

        # Normalize audio
        audio = audio / (np.max(np.abs(audio)) + 1e-8)

        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=config.sample_rate,
            n_mels=config.n_mels,
            hop_length=config.hop_length,
            n_fft=config.n_fft,
            fmin=config.fmin,
            fmax=config.fmax,
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

        # Min-max normalization
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (
            log_mel_spec.max() - log_mel_spec.min() + 1e-8
        )

        # Save as image using PIL (fast)
        img_array = (log_mel_spec * 255).astype(np.uint8)
        img_array = np.flipud(img_array)
        img = Image.fromarray(img_array, mode="L")
        img.save(img_path)

        return {
            "file_path": img_path,
            "raw_audio_path": audio_path,
            "label": label,
            "original_name": Path(audio_path).name,
        }
    except Exception:
        return None
