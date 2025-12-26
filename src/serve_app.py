from fastapi import FastAPI
from pydantic import BaseModel
import argparse
from src.spark_session import get_spark
from pyspark.ml.recommendation import ALSModel
from src.preprocessing import load_movielens_100k
from pyspark.sql.functions import col

app = FastAPI()
model = None
movies_df = None

class Query(BaseModel):
    user_id: int
    top_n: int = 10

@app.on_event("startup")
def startup_event():
    global model, movies_df
    import os
    model_path = os.environ.get("MODEL_PATH", "models/als_model")
    data_dir = os.environ.get("DATA_DIR", "data")
    spark = get_spark("ServeApp")
    model = ALSModel.load(model_path)
    ratings = load_movielens_100k(spark, data_dir)
    movies_df = ratings.select("movieId","title").dropDuplicates(['movieId']).cache()

@app.post("/recommend")
def recommend(q: Query):
    global model, movies_df
    spark = get_spark()
    user_df = spark.createDataFrame([(q.user_id,)], ["userId"])    
    recs = model.recommendForUserSubset(user_df, q.top_n).collect()
    if not recs:
        return {"recommendations": []}
    movie_ids = [r.movieId for r in recs[0][1]]
    out = movies_df.filter(col("movieId").isin(movie_ids)).toPandas().to_dict(orient='records')
    return {"recommendations": out}

if __name__ == "__main__":
    import uvicorn, os
    os.environ.setdefault('MODEL_PATH', 'models/als_model')
    os.environ.setdefault('DATA_DIR', 'data')
    uvicorn.run(app, host='0.0.0.0', port=8000)