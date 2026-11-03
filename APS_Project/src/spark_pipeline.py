"""
Big Data Analytics Module using Apache Spark (PySpark)
Processes and analyzes the PhiUSIIL Phishing URL Dataset (~191k+ records, 54 features).
Demonstrates distributed data cleaning, feature profiling, correlation analysis, and Spark vs Pandas benchmarks.
"""

import os
import sys
import time
import json

# Ensure stdout handles UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Ensure PySpark uses the correct Python interpreter on Windows
os.environ["PYSPARK_PYTHON"] = sys.executable
os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, StringType
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "phiusiil_dataset.csv")
OUTPUT_SUMMARY_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "spark_analytics_summary.json")

def get_spark_session():
    """Initializes and returns an optimized local SparkSession."""
    spark = SparkSession.builder \
        .appName("PhiUSIIL-BigData-Analytics") \
        .master("local[*]") \
        .config("spark.driver.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.ui.enabled", "false") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark

def run_spark_pipeline():
    print("=" * 70)
    print("[*] INITIATING APACHE SPARK BIG DATA ANALYTICS PIPELINE")
    print("=" * 70)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    spark = get_spark_session()
    print(f"[*] SparkSession initialized: Spark {spark.version} (Master: {spark.sparkContext.master})")

    # 1. Distributed Ingestion & Schema Profiling
    t0 = time.time()
    raw_df = spark.read \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .option("mode", "DROPMALFORMED") \
        .csv(DATA_PATH)
    raw_df.cache()
    total_records = raw_df.count()
    total_columns = len(raw_df.columns)
    ingest_time = round(time.time() - t0, 3)

    print(f"[+] Loaded {total_records:,} records with {total_columns} attributes in {ingest_time}s")

    # 2. Data Cleaning & Preprocessing
    # Drop identifier columns not used in statistical modeling
    cols_to_drop = [c for c in ['FILENAME', 'Title'] if c in raw_df.columns]
    df_clean = raw_df.drop(*cols_to_drop)

    # Cast numeric columns ensuring no nulls / NaNs in key attributes
    numeric_cols = [c for c in df_clean.columns if c not in ['URL', 'Domain', 'TLD']]
    for c in numeric_cols:
        df_clean = df_clean.withColumn(c, F.col(c).cast(DoubleType()))

    # Filter out rows where label is null
    df_clean = df_clean.filter(F.col("label").isNotNull()).cache()
    valid_count = df_clean.count()
    print(f"[+] Cleaned dataset valid records: {valid_count:,}")

    # 3. Distributed Aggregations & Threat Profiling
    print("[*] Computing distributed threat metrics and feature distributions...")
    label_dist = df_clean.groupBy("label").count().collect()
    # In PhiUSIIL: 1 = Legitimate, 0 = Phishing
    counts = {int(row["label"]): row["count"] for row in label_dist}
    legitimate_count = counts.get(1, 0)
    phishing_count = counts.get(0, 0)

    key_features = [
        "URLLength", "DomainLength", "IsDomainIP", "URLSimilarityIndex",
        "CharContinuationRate", "TLDLegitimateProb", "NoOfSubDomain",
        "SpacialCharRatioInURL", "IsHTTPS", "HasPasswordField",
        "HasExternalFormSubmit", "Bank", "Pay", "Crypto"
    ]
    available_features = [f for f in key_features if f in df_clean.columns]

    # Aggregations by class
    agg_exprs = []
    for f in available_features:
        agg_exprs.append(F.round(F.avg(f), 4).alias(f"{f}_avg"))
        agg_exprs.append(F.round(F.stddev(f), 4).alias(f"{f}_std"))

    feature_stats_df = df_clean.groupBy("label").agg(*agg_exprs).collect()
    
    class_feature_stats = {}
    for row in feature_stats_df:
        lbl_key = "legitimate" if int(row["label"]) == 1 else "phishing"
        class_feature_stats[lbl_key] = {}
        for f in available_features:
            class_feature_stats[lbl_key][f] = {
                "avg": row[f"{f}_avg"],
                "std": row[f"{f}_std"]
            }

    # 4. Distributed Pearson Correlation with Target Label
    print("[*] Computing distributed Pearson correlation against phishing label...")
    corr_results = {}
    for f in available_features:
        try:
            corr_val = df_clean.stat.corr(f, "label")
            corr_results[f] = round(corr_val, 4) if corr_val is not None else 0.0
        except Exception:
            corr_results[f] = 0.0

    # Sort correlation
    sorted_corr = sorted(corr_results.items(), key=lambda x: abs(x[1]), reverse=True)

    # 5. Big Data Processing Benchmark: Spark vs Pandas
    print("[*] Benchmarking Big Data processing performance (Spark vs Pandas)...")
    
    # Spark Benchmark
    t_start = time.time()
    spark_agg = df_clean.groupBy("label").agg(
        F.avg("URLLength"), F.avg("DomainLength"), F.count("IsDomainIP")
    ).collect()
    spark_time = round(time.time() - t_start, 4)

    # Filter benchmark in Spark
    t_start = time.time()
    spark_filtered = df_clean.filter((F.col("URLLength") > 60) & (F.col("NoOfSubDomain") > 2)).count()
    spark_filter_time = round(time.time() - t_start, 4)

    # Pandas Benchmark on the same dataset
    sample_df = pd.read_csv(DATA_PATH, on_bad_lines='skip', nrows=valid_count)
    t_start = time.time()
    pd_agg = sample_df.groupby("label")[["URLLength", "DomainLength", "IsDomainIP"]].mean()
    pandas_time = round(time.time() - t_start, 4)

    t_start = time.time()
    pd_filtered = len(sample_df[(sample_df["URLLength"] > 60) & (sample_df["NoOfSubDomain"] > 2)])
    pandas_filter_time = round(time.time() - t_start, 4)

    benchmark_data = {
        "dataset_rows": valid_count,
        "spark_version": spark.version,
        "spark_partitions": df_clean.rdd.getNumPartitions(),
        "aggregation_task": {
            "spark_time_sec": spark_time,
            "pandas_time_sec": pandas_time,
            "speedup_factor": round(pandas_time / max(spark_time, 0.0001), 2)
        },
        "complex_filter_task": {
            "spark_time_sec": spark_filter_time,
            "pandas_time_sec": pandas_filter_time,
            "filtered_matches": spark_filtered
        }
    }

    # Compile Final Summary Output
    summary_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_records": valid_count,
        "total_features": len(numeric_cols) - 1,
        "classes": {
            "legitimate": {
                "count": legitimate_count,
                "percentage": round(legitimate_count / valid_count * 100, 2)
            },
            "phishing": {
                "count": phishing_count,
                "percentage": round(phishing_count / valid_count * 100, 2)
            }
        },
        "feature_statistics": class_feature_stats,
        "top_correlations": sorted_corr,
        "benchmark": benchmark_data
    }

    with open(OUTPUT_SUMMARY_PATH, "w") as f:
        json.dump(summary_report, f, indent=2)

    print(f"[+] Big Data Analytics successfully exported to {OUTPUT_SUMMARY_PATH}")
    print("=" * 70)
    print(f"Summary: {valid_count:,} records processed | Phishing: {phishing_count:,} ({phishing_count/valid_count*100:.1f}%) | Legitimate: {legitimate_count:,} ({legitimate_count/valid_count*100:.1f}%)")
    print(f"Top 5 correlated features with Phishing: {sorted_corr[:5]}")
    print("=" * 70)

    spark.stop()
    return summary_report

if __name__ == "__main__":
    run_spark_pipeline()
