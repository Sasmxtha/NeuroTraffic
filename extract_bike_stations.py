"""
Extract unique bike stations from cleaned Citi Bike data on HDFS.
Runs in Docker (PySpark). Writes CSV to /scripts/bike_stations.csv
which is bind-mounted to E:\Big_Data_Project\ on the host.
"""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

CLEANED_BIKE = "hdfs://namenode:9000/data/neurotraffic/cleaned/citibike"

def main():
    spark = (
        SparkSession.builder
        .appName("ExtractBikeStations")
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.shuffle.partitions", "20")
        .config("spark.driver.maxResultSize", "1g")
        .getOrCreate()
    )

    print("Reading cleaned bike data...")
    bike = spark.read.parquet(CLEANED_BIKE)
    print(f"  Total rows: {bike.count():,}")

    start = bike.select(
        F.col("start_station_id").alias("station_id"),
        F.col("start_station_name").alias("station_name"),
        F.col("start_lat").alias("lat"),
        F.col("start_lng").alias("lng"),
    ).filter(F.col("station_id").isNotNull())

    end = bike.select(
        F.col("end_station_id").alias("station_id"),
        F.col("end_station_name").alias("station_name"),
        F.col("end_lat").alias("lat"),
        F.col("end_lng").alias("lng"),
    ).filter(F.col("station_id").isNotNull())

    stations = start.union(end).groupBy("station_id", "station_name").agg(
        F.avg("lat").alias("lat"),
        F.avg("lng").alias("lng"),
    )

    count = stations.count()
    print(f"  Unique stations: {count:,}")

    stations.coalesce(1).write.mode("overwrite").option("header", "true").csv("/tmp/bike_stations_out")

    import subprocess
    result = subprocess.run(
        ["find", "/tmp/bike_stations_out", "-name", "*.csv"],
        capture_output=True, text=True
    )
    csv_file = result.stdout.strip().split("\n")[0]
    subprocess.run(["cp", csv_file, "/scripts/bike_stations.csv"])
    print(f"  Written to /scripts/bike_stations.csv ({count} rows)")

    spark.stop()
    print("DONE")

if __name__ == "__main__":
    main()
