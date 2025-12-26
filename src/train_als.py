import argparse
import time
from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from src.spark_session import get_spark

def main(args):
    spark = get_spark("TrainALS-ML1M")

    train = spark.read.parquet(f"{args.data_path}/train_1m")

    print("Training ALS model...")
    start = time.time()

    als = ALS(
        userCol="userId",
        itemCol="movieId",
        ratingCol="rating",
        rank=args.rank,
        maxIter=args.maxIter,
        regParam=args.reg,
        coldStartStrategy="drop"
    )

    model = als.fit(train)
    train_time = time.time() - start

    print(f"ALS model trained in {train_time:.2f} seconds")

    model.save(args.model_out)
    print(f"Model saved to {args.model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--model-out", default="models/als_1m")
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--maxIter", type=int, default=10)
    args = parser.parse_args()
    main(args)
