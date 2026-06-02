import sys, os, time, json, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(__file__)), "common"))

from pyspark.sql import SparkSession
from pyspark.ml.feature import Tokenizer, HashingTF, IDF, StringIndexer, IndexToString
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import col

from utils import get_tokenizer, load_csv, CLASSES, ARTIFACTS_DIR
from evaluate import evaluate, plot_confusion_matrix

def main():
    spark = SparkSession.builder \
        .appName("TopicClassification") \
        .master("local[*]") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.driver.extraJavaOptions", "-Xss4m") \
        .config("spark.sql.adaptive.enabled", "true") \
        .getOrCreate()

    print("Spark session created.")

    tok = get_tokenizer()

    def add_bpe_column(df, name):
        texts = df["text"].tolist()
        bpe_texts = [" ".join(tok.tokenize(t)) for t in texts]
        df_out = df.copy()
        df_out["bpe_text"] = bpe_texts
        return df_out

    train_pd = add_bpe_column(load_csv("train"), "train")
    val_pd = add_bpe_column(load_csv("val"), "val")
    test_pd = add_bpe_column(load_csv("test"), "test")

    schema = StructType([
        StructField("bpe_text", StringType(), True),
        StructField("label", StringType(), True),
    ])

    train_sdf = spark.createDataFrame(train_pd[["bpe_text", "label"]], schema)
    val_sdf = spark.createDataFrame(val_pd[["bpe_text", "label"]], schema)
    test_sdf = spark.createDataFrame(test_pd[["bpe_text", "label"]], schema)

    print(f"Train: {train_sdf.count()}, Val: {val_sdf.count()}, Test: {test_sdf.count()}")

    tokenizer = Tokenizer(inputCol="bpe_text", outputCol="words")
    hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=5000)
    idf = IDF(inputCol="rawFeatures", outputCol="features")
    labelIndexer = StringIndexer(inputCol="label", outputCol="label_index").setHandleInvalid("keep")
    lr = LogisticRegression(maxIter=100, regParam=0.01, family="multinomial", featuresCol="features", labelCol="label_index")

    labelIndexerModel = labelIndexer.fit(train_sdf)
    labels = labelIndexerModel.labels
    labelConverter = IndexToString(inputCol="prediction", outputCol="predicted_label", labels=labels)

    pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, labelIndexerModel, lr, labelConverter])

    print("Training Spark Pipeline...")
    start_time = time.perf_counter()
    model = pipeline.fit(train_sdf)
    train_time = time.perf_counter() - start_time
    print(f"Training time: {train_time:.2f}s")

    print("Evaluating on test set...")
    predictions = model.transform(test_sdf)
    predictions.select("bpe_text", "label", "predicted_label", "probability").show(5, truncate=50)

    evaluator = MulticlassClassificationEvaluator(
        labelCol="label_index", predictionCol="prediction", metricName="f1"
    )
    f1_spark = evaluator.evaluate(predictions)
    print(f"Spark F1 (macro): {f1_spark:.4f}")

    preds_local = predictions.select("label", "predicted_label").collect()
    y_true = [r["label"] for r in preds_local]
    y_pred = [r["predicted_label"] for r in preds_local]
    metrics = evaluate(y_true, y_pred)
    print(f"Test accuracy: {metrics['accuracy']:.4f}, F1 (macro): {metrics['f1_macro']:.4f}")

    print(metrics["classification_report"])

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    model_path = os.path.join(ARTIFACTS_DIR, "spark_model")
    model.write().overwrite().save(model_path)

    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix_spark.png")
    plot_confusion_matrix(np.array(metrics["confusion_matrix"]), CLASSES, "Spark - Confusion Matrix", cm_path)

    import psutil
    proc = psutil.Process(os.getpid())
    peak_cpu = proc.memory_info().rss / 1024 / 1024

    info = {
        "framework": "spark",
        "train_time_s": round(train_time, 2),
        "peak_cpu_mb": round(peak_cpu, 1),
        "peak_gpu_mb": 0,
        "model_size_mb": round(sum(os.path.getsize(os.path.join(dp, f)) for dp, dn, fn in os.walk(model_path) for f in fn) / 1024 / 1024, 2),
        "test_accuracy": round(metrics["accuracy"], 4),
        "f1_macro": round(metrics["f1_macro"], 4),
    }
    with open(os.path.join(ARTIFACTS_DIR, "spark_metrics.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("Metrics saved:", json.dumps(info, indent=2))

    spark.stop()

if __name__ == "__main__":
    main()
