import os

PROJECT1_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT1_DIR, "data")
ARTIFACTS_DIR = os.path.join(PROJECT1_DIR, "artifacts")
REPORT_DIR = os.path.join(PROJECT1_DIR, "report")
MINIMIND_DIR = os.path.join(os.path.dirname(PROJECT1_DIR), "mini_deepseek_mind")
MODEL_DIR = os.path.join(MINIMIND_DIR, "model")

CLASSES = ["identity", "tech_discussion", "recommendation", "casual_chat", "domain_knowledge", "capability"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

def get_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    return tok

def load_csv(split="train"):
    import pandas as pd
    path = os.path.join(DATA_DIR, f"{split}.csv")
    return pd.read_csv(path)
