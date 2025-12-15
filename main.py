"""Main entry point for synthetic voice detection pipeline."""

import os

# Matplotlib backend setup
os.environ.pop("MPLBACKEND", None)
import matplotlib

matplotlib.use("Agg")

import json
from pathlib import Path

from src.audio_processor import AudioProcessor
from src.data_loader import AudioDataLoader
from src.models import CustomCNN, VisionTransformer, AudioRNN
from src.trainer import ModelTrainer
from src.config import config
from src.metrics import print_metrics_summary, print_model_comparison, generate_report


def load_or_process_data():
    """Load existing processed data or process raw audio files."""
    if not os.path.exists(config.raw_audio_dir):
        raise FileNotFoundError(f"Dataset not found at {config.raw_audio_dir}")

    flac_files = list(Path(config.raw_audio_dir).rglob("*.flac"))
    print(f"Found {len(flac_files)} .flac files")

    if len(flac_files) == 0:
        raise FileNotFoundError(f"No .flac files found in {config.raw_audio_dir}")

    if os.path.exists(config.metadata_file):
        with open(config.metadata_file, "r") as f:
            return json.load(f)

    processor = AudioProcessor()
    processed_data = processor.process_dataset(
        input_dir=config.raw_audio_dir, output_dir=config.spectrogram_dir
    )

    with open(config.metadata_file, "w") as f:
        json.dump(processed_data, f, indent=2)

    return processed_data


def create_data_loaders(processed_data):
    """Create train/val/test data loaders for all model types."""
    loader = AudioDataLoader()

    print("Creating data loaders...")
    print("  Image loaders (CNN, ViT)...")
    img_loaders = loader.create_data_loaders(processed_data, load_raw_audio=False)

    print("  Audio loaders (RNN)...")
    audio_loaders = loader.create_data_loaders(processed_data, load_raw_audio=True)

    return img_loaders, audio_loaders


def train_models(img_loaders, audio_loaders):
    """Train all models and return their trainers and metrics."""
    img_train, img_val, img_test = img_loaders
    audio_train, audio_val, audio_test = audio_loaders

    models_config = [
        ("CNN", CustomCNN(), img_train, img_val, img_test),
        ("ViT", VisionTransformer(), img_train, img_val, img_test),
        ("RNN", AudioRNN(), audio_train, audio_val, audio_test),
    ]

    trainers = {}
    val_metrics = {}
    test_metrics = {}

    for name, model, train_loader, val_loader, test_loader in models_config:
        print(f"\n{'=' * 60}")
        print(f"Training {name}...")
        print(f"{'=' * 60}")

        trainer = ModelTrainer(model, train_loader, val_loader, name.lower())
        val_metrics[name] = trainer.train()

        print(f"\nEvaluating {name} on test set...")
        test_metrics[name] = trainer.evaluate(test_loader)
        print_metrics_summary(test_metrics[name], f"{name} Test Set")

        trainers[name] = trainer

    return trainers, val_metrics, test_metrics


def main():
    """Run the complete synthetic voice detection pipeline."""
    print("=" * 60)
    print("  SYNTHETIC VOICE DETECTION PIPELINE")
    print("=" * 60)

    # Step 1: Data processing
    print("\n[1/4] Processing dataset...")
    processed_data = load_or_process_data()
    print(f"Total samples: {len(processed_data)}")

    # Step 2: Create data loaders
    print("\n[2/4] Creating data loaders...")
    img_loaders, audio_loaders = create_data_loaders(processed_data)

    # Step 3: Train and evaluate models
    print("\n[3/4] Training models...")
    trainers, val_metrics, test_metrics = train_models(img_loaders, audio_loaders)

    # Print comparison
    print_model_comparison(test_metrics)

    # Step 4: Generate report
    print("\n[4/4] Generating report...")
    generate_report(val_metrics, test_metrics, len(processed_data))

    print("\n" + "=" * 60)
    print("  PIPELINE COMPLETED")
    print("=" * 60)
    print(f"Models saved: {config.model_dir}/")
    print(f"Report saved: {config.report_file}")


if __name__ == "__main__":
    main()
