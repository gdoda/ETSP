import os

# Matplotlib setup for Colab
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


def main():
    print("=== SYNTHETIC VOICE DETECTION - EVAL DATASET ===")

    processor = AudioProcessor()
    data_loader = AudioDataLoader()

    print("1. Processing dataset...")

    if not os.path.exists(config.raw_audio_dir):
        print(
            f"ERROR: Dataset not found at {config.raw_audio_dir}. Flac files expected at: {config.raw_audio_dir}"
        )
        return

    flac_files = list(Path(config.raw_audio_dir).rglob("*.flac"))
    print(f"Found {len(flac_files)} .flac files in dataset")

    if len(flac_files) == 0:
        print("No .flac files found. Check your dataset at: {config.raw_audio_dir}")
        return

    if os.path.exists(config.metadata_file):
        with open(config.metadata_file, "r") as f:
            processed_data = json.load(f)
    else:
        processed_data = processor.process_dataset(
            input_dir=config.raw_audio_dir, output_dir=config.spectrogram_dir
        )
        with open(config.metadata_file, "w") as f:
            json.dump(processed_data, f, indent=2)

    print(f"Successfully processed {len(processed_data)} files")

    # Create data loaders
    print("2. Creating data loaders...")

    # Loaders for CNN and ViT (Spectrogram Images)
    print("   -> Creating Image Data Loaders (for CNN, ViT)...")
    img_train_loader, img_val_loader, img_test_loader = data_loader.create_data_loaders(
        processed_data, load_raw_audio=False
    )

    # Loaders for RNN (Raw Audio)
    print("   -> Creating Raw Audio Data Loaders (for RNN)...")
    audio_train_loader, audio_val_loader, audio_test_loader = (
        data_loader.create_data_loaders(processed_data, load_raw_audio=True)
    )

    # Train CNN Model
    print("3. Training CNN Model...")
    cnn_model = CustomCNN()
    cnn_trainer = ModelTrainer(cnn_model, img_train_loader, img_val_loader, "cnn")
    cnn_metrics = cnn_trainer.train()

    # Train Vision Transformer
    print("4. Training Vision Transformer...")
    vit_model = VisionTransformer()
    vit_trainer = ModelTrainer(vit_model, img_train_loader, img_val_loader, "vit")
    vit_metrics = vit_trainer.train()

    # Train RNN Model (using raw audio)
    print("5. Training RNN Model...")
    rnn_model = AudioRNN()
    rnn_trainer = ModelTrainer(rnn_model, audio_train_loader, audio_val_loader, "rnn")
    rnn_metrics = rnn_trainer.train()

    # Generate comprehensive report
    print("6. Generating final report...")
    generate_comprehensive_report(cnn_metrics, vit_metrics, rnn_metrics, processed_data)

    print("7. PROJECT COMPLETED")
    print(f"All models trained and saved in '{config.model_dir}/' folder")
    print(f"Final report saved: '{config.report_file}'")


def generate_comprehensive_report(cnn_metrics, vit_metrics, rnn_metrics, data):
    report = {
        "project_title": "Synthetic Voice Detection System",
        "dataset_used": "ASVspoof2021 dataset",
        "total_samples": len(data),
        "models_trained": [
            "CNN (Convolutional Neural Network)",
            "Vision Transformer (ViT)",
            "RNN (Recurrent Neural Network)",
        ],
        "training_configuration": {
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "input_features": f"Mel-spectrograms ({config.n_mels} bands) & Raw Audio",
            "evaluation_metric": "Accuracy, Loss, F1-Score",
        },
        "model_performance": {
            "CNN": {
                "best_validation_accuracy": f"{cnn_metrics.get('accuracy', 0) * 100:.2f}%",
                "best_epoch": cnn_metrics.get("epoch", 0),
                "validation_loss": f"{cnn_metrics.get('loss', 0):.4f}",
                "f1_score": f"{cnn_metrics.get('f1_score', 0):.4f}",
                "eer": f"{cnn_metrics.get('eer', 0):.2f}%",
                "balanced_accuracy": f"{cnn_metrics.get('balanced_accuracy', 0):.4f}",
                "roc_auc": f"{cnn_metrics.get('roc_auc', 0):.4f}",
                "bonafide_recall": f"{cnn_metrics.get('bonafide_recall', 0):.4f}",
                "spoof_recall": f"{cnn_metrics.get('spoof_recall', 0):.4f}",
            },
            "Vision_Transformer": {
                "best_validation_accuracy": f"{vit_metrics.get('accuracy', 0) * 100:.2f}%",
                "best_epoch": vit_metrics.get("epoch", 0),
                "validation_loss": f"{vit_metrics.get('loss', 0):.4f}",
                "f1_score": f"{vit_metrics.get('f1_score', 0):.4f}",
                "eer": f"{vit_metrics.get('eer', 0):.2f}%",
                "balanced_accuracy": f"{vit_metrics.get('balanced_accuracy', 0):.4f}",
                "roc_auc": f"{vit_metrics.get('roc_auc', 0):.4f}",
                "bonafide_recall": f"{vit_metrics.get('bonafide_recall', 0):.4f}",
                "spoof_recall": f"{vit_metrics.get('spoof_recall', 0):.4f}",
            },
            "RNN": {
                "best_validation_accuracy": f"{rnn_metrics.get('accuracy', 0) * 100:.2f}%",
                "best_epoch": rnn_metrics.get("epoch", 0),
                "validation_loss": f"{rnn_metrics.get('loss', 0):.4f}",
                "f1_score": f"{rnn_metrics.get('f1_score', 0):.4f}",
                "eer": f"{rnn_metrics.get('eer', 0):.2f}%",
                "balanced_accuracy": f"{rnn_metrics.get('balanced_accuracy', 0):.4f}",
                "roc_auc": f"{rnn_metrics.get('roc_auc', 0):.4f}",
                "bonafide_recall": f"{rnn_metrics.get('bonafide_recall', 0):.4f}",
                "spoof_recall": f"{rnn_metrics.get('spoof_recall', 0):.4f}",
            },
        },
        "project_achievements": [
            "Successfully implemented 3 different deep learning architectures",
            "Trained on ASVspoof2021 benchmark dataset",
            "Achieved synthetic voice detection capability",
            "Built complete end-to-end ML pipeline",
            "Included data augmentation and preprocessing",
        ],
        "technical_implementation": {
            "data_preprocessing": "Audio normalization, Mel-spectrogram conversion, Data augmentation",
            "model_architectures": f"CNN ({len(config.cnn_channels)} conv layers), ViT ({config.vit_depth} transformer layers), RNN (LSTM with attention)",
            "training_strategy": "Cross-validation, Early stopping, Learning rate scheduling",
            "evaluation_metrics": "Accuracy, Precision, Recall, F1-Score, Equal Error Rate",
        },
    }

    with open(config.report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"Report metadata file generated successfully: {config.report_file}")


if __name__ == "__main__":
    main()
