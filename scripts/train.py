import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from sklearn.metrics import accuracy_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split

from src.dataset import ComposerDataset
from src.model import ComposerClassifier

DATASET_PATH = "/home/mrozek/ssne-2025l/composer-rnn-classifier/data/train.pkl"
VAL_SPLIT_RATIO = 0.1
LEARNING_RATE = 1e-4
EPOCHS = 200
BATCH_SIZE = 32
EMBEDDING_DIM = 128
HIDDEN_DIM = 256
NUM_LSTM_LAYERS = 1
DROPOUT_RATE = 0.0
FC_LAYERS = [128, 64]
MODEL_SAVE_PATH = (
    "/home/mrozek/ssne-2025l/composer-rnn-classifier/models/last_saved_model.pkl"
)
SAVE_EVERY_N_EPOCHS = 5


def pad_collate_fn(batch):
    if len(batch[0]) == 2:
        sequences, labels = zip(*batch)
        sequence_lengths = [len(s) for s in sequences]
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)
        labels = torch.stack(labels)

        return sequences_padded, labels, sequence_lengths
    else:
        sequences = batch
        sequence_lengths = [len(s) for s in sequences]
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0)

        return sequences_padded, sequence_lengths


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    hyperparameters = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "embedding_dim": EMBEDDING_DIM,
        "hidden_dim": HIDDEN_DIM,
        "num_lstm_layers": NUM_LSTM_LAYERS,
        "dropout": DROPOUT_RATE,
        "val_split_ratio": VAL_SPLIT_RATIO,
        "fc_layers": FC_LAYERS,
    }
    wandb.init(project="composer-rnn-classifier", config=hyperparameters)

    full_dataset = ComposerDataset(DATASET_PATH, is_test=False)
    dataset_size = len(full_dataset)
    val_size = int(VAL_SPLIT_RATIO * dataset_size)
    train_size = dataset_size - val_size

    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=1,
        collate_fn=pad_collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        collate_fn=pad_collate_fn,
    )

    print(
        f"Loaded dataset from {DATASET_PATH}"
        f"Train size: {train_size}"
        f"Val size: {val_size}"
    )

    model = ComposerClassifier(
        vocab_size=full_dataset.vocab_size,
        embed_dim=EMBEDDING_DIM,
        num_lstm_layers=NUM_LSTM_LAYERS,
        hidden_dim=HIDDEN_DIM,
        fc_dims=FC_LAYERS,
        num_classes=5,
        dropout=DROPOUT_RATE,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    start_epoch = 0
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Loading checkpoint from {MODEL_SAVE_PATH}...")
        try:
            checkpoint = torch.load(MODEL_SAVE_PATH, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            print(f"Resuming training from epoch {start_epoch}")
        except Exception as e:
            print(f"Could not load checkpoint: {e}. Starting from scratch.")
            start_epoch = 0

    print(
        f"Number of parameters in RNN model: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )
    print("Starting training...")

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss_total = 0.0
        train_predictions = []
        train_true_labels = []
        for i, (sequences, labels, seq_lengths) in enumerate(train_loader):
            optimizer.zero_grad()
            sequences = sequences.to(device)
            labels = labels.to(device)

            outputs = model(sequences, seq_lengths)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            train_predictions.extend(predictions.cpu().detach().numpy())
            train_true_labels.extend(labels.cpu().detach().numpy())
            if (i + 1) % 50 == 0:
                print(
                    f"Epoch [{epoch+1}/{EPOCHS}], Batch [{i+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.4f})"
                )

        avg_train_loss_total = train_loss_total / len(train_loader)
        train_acc = accuracy_score(train_true_labels, train_predictions)

        scheduler.step()

        model.eval()
        val_predictions = []
        val_true_labels = []
        val_loss_total = 0.0
        with torch.no_grad():
            for sequences, labels, seq_lengths in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                outputs = model(sequences, seq_lengths)
                loss = criterion(outputs, labels)

                val_loss_total += loss.item()
                predictions = torch.argmax(outputs, dim=1)
                val_predictions.extend(predictions.cpu().detach().numpy())
                val_true_labels.extend(labels.cpu().detach().numpy())

        avg_val_loss_total = val_loss_total / len(val_loader)
        val_acc = accuracy_score(val_true_labels, val_predictions)

        print(f"Epoch {epoch+1}/{EPOCHS}:")
        print(f"  Train Loss: {avg_train_loss_total:.4f})")
        print(f"  Train Acc: {train_acc:.4f})")
        print(f"  Val Loss:   {avg_val_loss_total:.4f})")
        print(f"  Val Acc: {val_acc:.4f})")
        log_dict = {
            "train/total_loss": avg_train_loss_total,
            "train/accuracy": train_acc,
            "val/total_loss": avg_val_loss_total,
            "val/accuracy": val_acc,
            "val/cm": wandb.plot.confusion_matrix(
                probs=None, y_true=val_true_labels, preds=val_predictions
            ),
        }

        wandb.log(log_dict)

        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0 or (epoch + 1) == EPOCHS:
            print(f"Saving model checkpoint at epoch {epoch+1}...")
            save_data = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "config": hyperparameters,
            }
            try:
                os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
                torch.save(save_data, MODEL_SAVE_PATH)
                print(f"Checkpoint saved to {MODEL_SAVE_PATH}")
            except Exception as e:
                print(f"Error saving checkpoint: {e}")

    wandb.finish()
    print("Training finished.")
    print(f"Final model checkpoint saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
