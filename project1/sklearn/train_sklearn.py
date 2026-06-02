import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "common"))

import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from utils import get_tokenizer, load_csv, CLASSES, ARTIFACTS_DIR
from evaluate import evaluate, plot_confusion_matrix
from monitor import ResourceMonitor

def main():
    tok = get_tokenizer()

    train = load_csv("train")
    val = load_csv("val")
    test = load_csv("test")

    print("BPE tokenizing...")
    train_texts = [" ".join(tok.tokenize(t)) for t in train["text"]]
    val_texts = [" ".join(tok.tokenize(t)) for t in val["text"]]
    test_texts = [" ".join(tok.tokenize(t)) for t in test["text"]]

    print("Vectorizing with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, token_pattern=r"(?u)\S+")
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    y_train = train["label"].values
    y_val = val["label"].values
    y_test = test["label"].values

    print(f"X_train: {X_train.shape}, X_val: {X_val.shape}, X_test: {X_test.shape}")

    print("Training LogisticRegression...")
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)

    mon = ResourceMonitor("sklearn")
    mon.start()
    model.fit(X_train, y_train)
    mon.stop()
    info = mon.summary()
    print(f"Training time: {info['train_time_s']}s, Peak CPU: {info['peak_cpu_mb']} MB")

    val_score = model.score(X_val, y_val)
    test_score = model.score(X_test, y_test)
    print(f"Val accuracy: {val_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

    y_pred = model.predict(X_test)
    metrics = evaluate(y_test.tolist(), y_pred.tolist())
    print(f"Test F1 (macro): {metrics['f1_macro']:.4f}")
    print(metrics["classification_report"])

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACTS_DIR, "sklearn_model.pkl")
    vec_path = os.path.join(ARTIFACTS_DIR, "sklearn_vectorizer.pkl")
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vec_path)

    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix_sklearn.png")
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), CLASSES, "sklearn - Confusion Matrix", cm_path)

    model_size_mb = os.path.getsize(model_path) / 1024 / 1024
    info["model_size_mb"] = round(model_size_mb, 2)
    info["val_accuracy"] = round(val_score, 4)
    info["test_accuracy"] = round(test_score, 4)
    info["f1_macro"] = round(metrics["f1_macro"], 4)

    import json
    with open(os.path.join(ARTIFACTS_DIR, "sklearn_metrics.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("Metrics saved:", json.dumps(info, indent=2))

if __name__ == "__main__":
    main()
