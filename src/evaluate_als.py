import argparse
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALSModel
from src.spark_session import get_spark

def main(args):
    spark = get_spark("EvaluateALS-ML1M")

    test = spark.read.parquet(f"{args.data_path}/test_1m")
    model = ALSModel.load(args.model_path)

    preds = model.transform(test)

    rmse_eval = RegressionEvaluator(metricName="rmse", labelCol="rating", predictionCol="prediction")
    mae_eval = RegressionEvaluator(metricName="mae", labelCol="rating", predictionCol="prediction")

    rmse = rmse_eval.evaluate(preds)
    mae = mae_eval.evaluate(preds)

    print("\n=== ALS Evaluation (MovieLens 1M) ===")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE : {mae:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--model-path", default="models/als_1m")
    args = parser.parse_args()
    main(args)
