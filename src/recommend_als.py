import argparse
from pyspark.ml.recommendation import ALSModel
from src.spark_session import get_spark
from pyspark.sql import functions as F

def main(args):
    spark = get_spark("ALSRecommend-ML1M")

    model = ALSModel.load(args.model_path)
    movies = spark.read.text(f"{args.data_path}/movies.dat")

    # Split movies.dat
    movies = movies.withColumn("temp", F.split("value", "::")) \
        .select(
            F.col("temp").getItem(0).cast("int").alias("movieId"),
            F.col("temp").getItem(1).alias("title")
        )

    # Recommend for user
    recs = model.recommendForUserSubset(
        spark.createDataFrame([(args.user_id,)], ["userId"]),
        args.top_n
    ).collect()[0]["recommendations"]

    print(f"\nTop-{args.top_n} ALS Recommendations for User {args.user_id}:\n")
    for row in recs:
        movie = movies.filter(movies.movieId == row.movieId).collect()[0].title
        print(f"{row.movieId:<5} | {movie:<40} | predicted={row.rating:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--model-path", default="models/als_1m")
    parser.add_argument("--user-id", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()
    main(args)
