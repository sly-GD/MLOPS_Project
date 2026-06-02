import json
import random
import re
import csv
import os
from collections import defaultdict

random.seed(42)

SRC_PATH = os.path.join(os.path.dirname(__file__), "../minimind_data/sft_t2t_mini.jsonl")
DST_DIR = os.path.join(os.path.dirname(__file__), "data")

CLASS_KEYWORDS = {
    "identity": ["真实来源", "真实身份", "开发背景", "哪家公司", "谁开发", "谁创造", "开发团队", "背景", "版本", "来源", "创始人", "公司"],
    "tech_discussion": ["如何平衡", "怎么处理", "原理", "算法", "为什么", "数学", "训练", "模型", "数据", "参数", "优化", "梯度", "损失", "架构", "推理", "注意力", "transformer", "loss", "embedding", "权重", "收敛", "过拟合", "欠拟合", "泛化", "正则化"],
    "recommendation": ["推荐", "有没有看过", "好电影", "好书", "好听的", "好看的", "推荐一下", "有什么推荐", "安利", "推荐一部", "推荐一本", "推荐一首"],
    "casual_chat": ["今天", "过得怎么样", "在忙什么", "聊聊", "你好吗", "最近", "心情", "天气", "吃饭", "早安", "晚安", "哈哈", "随便聊聊", "聊聊天", "开心"],
    "domain_knowledge": ["医疗", "伦理", "人工智能", "量子", "科学", "基因", "物理", "化学", "生物", "数学题", "公式", "定理", "历史", "政治", "经济", "哲学", "心理", "医学", "疾病", "治疗"],
    "capability": ["能不能", "是否支持", "可以做什么", "多语言", "能力", "功能", "支持", "会什么", "擅长", "能做什么", "什么都会", "会不会", "有多强", "局限性", "限制"],
}

def extract_first_query(line):
    try:
        obj = json.loads(line)
        conversations = obj.get("conversations", [])
        if not conversations:
            return None
        first = conversations[0]
        if first.get("role") == "user":
            content = first.get("content", "").strip()
            if len(content) >= 3:
                return content
        return None
    except (json.JSONDecodeError, KeyError, IndexError):
        return None

def classify(text):
    for cls, keywords in CLASS_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                return cls
    return "other"

def main():
    buckets = defaultdict(list)

    with open(SRC_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            query = extract_first_query(line)
            if query is None:
                continue
            cls = classify(query)
            buckets[cls].append(query)

            if (i + 1) % 100000 == 0:
                print(f"  Processed {i+1} lines...")

    print(f"\nTotal lines processed: {i+1}")
    print("Class distribution before sampling:")
    for cls in sorted(buckets.keys()):
        print(f"  {cls}: {len(buckets[cls])}")

    target_classes = list(CLASS_KEYWORDS.keys())
    samples_per_class = 16666
    sampled = []
    for cls in target_classes:
        pool = buckets.get(cls, [])
        n = min(samples_per_class, len(pool))
        chosen = random.sample(pool, n)
        for text in chosen:
            sampled.append({"text": text, "label": cls})
        print(f"  Sampled {n}/{len(pool)} for {cls}")

    random.shuffle(sampled)
    total = len(sampled)
    train_end = int(total * 0.8)
    val_end = int(total * 0.9)

    train = sampled[:train_end]
    val = sampled[train_end:val_end]
    test = sampled[val_end:]

    print(f"\nTrain: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = os.path.join(DST_DIR, f"{split_name}.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["text", "label"])
            writer.writeheader()
            writer.writerows(split_data)
        print(f"  Saved {path} ({len(split_data)} rows)")

if __name__ == "__main__":
    main()
