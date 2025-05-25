import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from src.dataset import ComposerDataset
from src.model import ComposerClassifier

DATASET_PATH = "/home/mbilski/dir/test.pkl"
MODEL_PATH = "/home/mbilski/dir/model.pkl"
OUTPUT_PATH = "/home/mbilski/dir/predictions.txt"
BATCH_SIZE = 32


def pad_collate_fn(batch):
    sequences = batch
    sequence_lengths = [len(s) for s in sequences]
    sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    return sequences_padded, sequence_lengths


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    full_dataset = ComposerDataset(DATASET_PATH, is_test=True)

    print(f"Loading model from {MODEL_PATH}...")
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        hyperparameters = checkpoint.get("config", {})

        EMBEDDING_DIM = hyperparameters.get("embedding_dim", 128)
        HIDDEN_DIM = hyperparameters.get("hidden_dim", 256)
        NUM_LSTM_LAYERS = hyperparameters.get("num_lstm_layers", 1)
        DROPOUT_RATE = hyperparameters.get("dropout", 0.1)
        FC_LAYERS = hyperparameters.get("fc_layers", [128, 64])

        model = ComposerClassifier(
            vocab_size=full_dataset.vocab_size,
            embed_dim=EMBEDDING_DIM,
            num_lstm_layers=NUM_LSTM_LAYERS,
            hidden_dim=HIDDEN_DIM,
            fc_dims=FC_LAYERS,
            num_classes=5,
            dropout=DROPOUT_RATE,
        ).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure the model path is correct and the model architecture matches.")
        return

    test_loader = DataLoader(
        full_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        collate_fn=pad_collate_fn,
    )

    print(f"Loaded test dataset from {DATASET_PATH} with {len(full_dataset)} samples.")
    print("Starting prediction...")

    predictions = []
    with torch.no_grad():
        for i, (sequences, seq_lengths) in enumerate(test_loader):
            sequences = sequences.to(device)

            outputs = model(sequences, seq_lengths)
            predicted_labels = torch.argmax(outputs, dim=1)
            predictions.extend(predicted_labels.cpu().numpy())

            if (i + 1) % 50 == 0:
                print(f"Processed batch [{i+1}/{len(test_loader)}]")

    print("Prediction finished. Saving results...")

    output_dir = os.path.dirname(OUTPUT_PATH)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(OUTPUT_PATH, "w") as f:
        for pred in predictions:
            f.write(f"{pred}\n")

    print(f"Predictions saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
