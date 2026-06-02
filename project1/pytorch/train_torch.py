import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "common"))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import TextClassifier
from utils import get_tokenizer, load_csv, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS, ARTIFACTS_DIR
from evaluate import evaluate, plot_confusion_matrix
from monitor import ResourceMonitor


class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, vocab_size=6400):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        token_ids = self.tokenizer.encode(self.texts[idx])
        token_ids = [t for t in token_ids if t < self.vocab_size]
        return torch.tensor(token_ids, dtype=torch.long), self.labels[idx]


def collate_fn(batch):
    input_ids = []
    labels = []
    offsets = [0]
    for tokens, label in batch:
        input_ids.append(tokens)
        labels.append(label)
        offsets.append(offsets[-1] + len(tokens))
    input_ids = torch.cat(input_ids)
    offsets = offsets[:-1]
    return input_ids, torch.tensor(offsets, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for input_ids, offsets, labels in loader:
        input_ids, offsets, labels = input_ids.to(device), offsets.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(input_ids, offsets)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / len(loader), correct / total


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for input_ids, offsets, labels in loader:
            input_ids, offsets, labels = input_ids.to(device), offsets.to(device), labels.to(device)
            outputs = model(input_ids, offsets)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)
    return total_loss / len(loader), correct / total


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tok = get_tokenizer()
    VOCAB_SIZE = 6400

    train_df = load_csv("train")
    val_df = load_csv("val")
    test_df = load_csv("test")

    y_train = torch.tensor([CLASS_TO_IDX[c] for c in train_df["label"]], dtype=torch.long)
    y_val = torch.tensor([CLASS_TO_IDX[c] for c in val_df["label"]], dtype=torch.long)
    y_test = torch.tensor([CLASS_TO_IDX[c] for c in test_df["label"]], dtype=torch.long)

    train_ds = TextClassificationDataset(train_df["text"].tolist(), y_train, tok, VOCAB_SIZE)
    val_ds = TextClassificationDataset(val_df["text"].tolist(), y_val, tok, VOCAB_SIZE)
    test_ds = TextClassificationDataset(test_df["text"].tolist(), y_test, tok, VOCAB_SIZE)

    BATCH_SIZE = 64
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = TextClassifier(vocab_size=VOCAB_SIZE, embed_dim=256, num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    print(f"Model has {sum(p.numel() for p in model.parameters()):,} parameters")

    NUM_EPOCHS = 10
    best_val_acc = 0
    patience = 3
    stale = 0

    mon = ResourceMonitor("pytorch")
    mon.start()

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:2d}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(ARTIFACTS_DIR, "pytorch_model.pt"))
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    mon.stop()
    info = mon.summary()
    print(f"Training time: {info['train_time_s']}s, Peak CPU: {info['peak_cpu_mb']} MB, Peak GPU: {info['peak_gpu_mb']} MB")

    model.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "pytorch_model.pt")))
    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for input_ids, offsets, labels in test_loader:
            input_ids, offsets, labels = input_ids.to(device), offsets.to(device), labels.to(device)
            outputs = model(input_ids, offsets)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    metrics = evaluate(all_labels, all_preds)
    print(f"Test accuracy: {metrics['accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")
    print(metrics["classification_report"])

    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix_torch.png")
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), CLASSES, "PyTorch - Confusion Matrix", cm_path)

    model_size_mb = os.path.getsize(os.path.join(ARTIFACTS_DIR, "pytorch_model.pt")) / 1024 / 1024
    info["model_size_mb"] = round(model_size_mb, 2)
    info["test_accuracy"] = round(metrics["accuracy"], 4)
    info["f1_macro"] = round(metrics["f1_macro"], 4)

    with open(os.path.join(ARTIFACTS_DIR, "pytorch_metrics.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("Metrics saved:", json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
