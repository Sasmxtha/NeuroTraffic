"""
Export combined parquet to TSV for Hadoop MapReduce Streaming input.
Runs in Docker (PySpark).
"""

from pyspark.sql import SparkSession

COMBINED = "hdfs://namenode:9000/data/neurotraffic/combined"
OUTPUT_TSV = "hdfs://namenode:9000/data/neurotraffic/mapreduce_input"

def main():
    spark = (
        SparkSession.builder
        .appName("ExportForMapReduce")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )

    print("Reading combined data...")
    df = spark.read.parquet(COMBINED)
    print(f"  Rows: {df.count():,}")

    print("Writing TSV for MapReduce input...")
    df.write.mode("overwrite").option("sep", "\t").option("header", "true").csv(OUTPUT_TSV)
    print(f"  Written to {OUTPUT_TSV}")

    spark.stop()
    print("DONE")

if __name__ == "__main__":
    main()
