import argparse
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, FloatType
from src.spark_session import get_spark

def load_ml1m(spark, data_path):
    ratings_path = f"{data_path}/ratings.dat"
    movies_path = f"{data_path}/movies.dat"

    ratings = (
        spark.read.text(ratings_path)
        .withColumn("temp", F.split(F.col("value"), "::"))
        .select(
            F.col("temp").getItem(0).cast(IntegerType()).alias("userId"),
            F.col("temp").getItem(1).cast(IntegerType()).alias("movieId"),
            F.col("temp").getItem(2).cast(FloatType()).alias("rating"),
            F.col("temp").getItem(3).cast(IntegerType()).alias("timestamp")
        )
    )

    movies = (
        spark.read.text(movies_path)
        .withColumn("temp", F.split(F.col("value"), "::"))
        .select(
            F.col("temp").getItem(0).cast(IntegerType()).alias("movieId"),
            F.col("temp").getItem(1).alias("title")
        )
    )

    return ratings, movies


def main(args):
    spark = get_spark("Preprocess ML1M")

    ratings, movies = load_ml1m(spark, args.data_path)

    # Join titles (important for recommend step)
    df = ratings.join(movies, "movieId", "left")

    # Split train/test
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    train.write.mode("overwrite").parquet(f"{args.data_path}/train_1m")
    test.write.mode("overwrite").parquet(f"{args.data_path}/test_1m")

    print("Preprocessing complete.")
    print("Saved:")
    print(f"- {args.data_path}/train_1m")
    print(f"- {args.data_path}/test_1m")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data")
    args = parser.parse_args()
    main(args)
