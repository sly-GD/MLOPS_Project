import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "common"))

import time
import joblib
import numpy as np
from utils import get_tokenizer, load_csv, CLASSES, ARTIFACTS_DIR
from evaluate import evaluate

def main():
    tok = get_tokenizer()
    test = load_csv("test")

    model = joblib.load(os.path.join(ARTIFACTS_DIR, "sklearn_model.pkl"))
    vectorizer = joblib.load(os.path.join(ARTIFACTS_DIR, "sklearn_vectorizer.pkl"))

    test_texts = [" ".join(tok.tokenize(t)) for t in test["text"]]
    X_test = vectorizer.transform(test_texts)

    # Batch inference timing
    start = time.perf_counter()
    y_pred = model.predict(X_test)
    batch_time = time.perf_counter() - start
    n_samples = X_test.shape[0]
    print(f"Batch inference ({n_samples} samples): {batch_time:.4f}s ({batch_time/n_samples*1000:.2f}ms per sample)")

    # Single inference timing
    single_times = []
    for i in range(min(100, len(test_texts))):
        x = vectorizer.transform([test_texts[i]])
        t0 = time.perf_counter()
        model.predict(x)
        single_times.append(time.perf_counter() - t0)
    avg_single = np.mean(single_times) * 1000
    print(f"Single inference (avg of {len(single_times)}): {avg_single:.4f}ms")

    metrics = evaluate(test["label"].tolist(), y_pred.tolist())
    print(f"Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1_macro']:.4f}")

if __name__ == "__main__":
    main()
