import sys, os, time, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "common"))

import torch
from torch.utils.data import DataLoader
from model import TextClassifier
from train_torch import TextClassificationDataset, collate_fn
from utils import get_tokenizer, load_csv, CLASSES, CLASS_TO_IDX, ARTIFACTS_DIR
from evaluate import evaluate


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextClassifier(vocab_size=6400, embed_dim=256, num_classes=len(CLASSES)).to(device)
    model.load_state_dict(torch.load(os.path.join(ARTIFACTS_DIR, "pytorch_model.pt"), map_location=device))
    model.eval()

    tok = get_tokenizer()
    test_df = load_csv("test")
    y_test = torch.tensor([CLASS_TO_IDX[c] for c in test_df["label"]], dtype=torch.long)
    test_ds = TextClassificationDataset(test_df["text"].tolist(), y_test, tok, 6400)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    # Batch inference timing
    start = time.perf_counter()
    all_preds = []
    with torch.no_grad():
        for input_ids, offsets, labels in test_loader:
            input_ids, offsets = input_ids.to(device), offsets.to(device)
            outputs = model(input_ids, offsets)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().tolist())
    batch_time = time.perf_counter() - start
    num_samples = len(test_df)
    print(f"Batch inference ({num_samples} samples): {batch_time:.4f}s ({batch_time/num_samples*1000:.2f}ms per sample)")

    # Single inference timing
    single_times = []
    with torch.no_grad():
        for i in range(min(100, len(test_df))):
            tokens = tok.encode(test_df["text"].iloc[i])
            tokens = torch.tensor(tokens, dtype=torch.long).to(device)
            offsets = torch.tensor([0], dtype=torch.long).to(device)
            t0 = time.perf_counter()
            model(tokens, offsets)
            single_times.append(time.perf_counter() - t0)
    avg_single = np.mean(single_times) * 1000
    print(f"Single inference (avg of {len(single_times)}): {avg_single:.4f}ms")

    metrics = evaluate(test_df["label"].tolist(), [CLASSES[p] for p in all_preds])
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()
