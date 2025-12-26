"""
Optimized evaluation of ALS, User-CF and Item-CF with ranking metrics.
This version computes user-user and item-item similarity ONCE, samples a small set
of users (default 20) and evaluates Precision@K, Recall@K, MAP@K, NDCG@K.

Run:
python -m src.evaluate_all_models --data-path data --model-path models/als_1m --k 10 --num-users 20
"""

import argparse
import time
import numpy as np
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import functions as F

from src.spark_session import get_spark
from src.metrics import precision_at_k, recall_at_k, average_precision_at_k, ndcg_at_k

# These imports assume your existing CF modules expose the same functions as before
from src.user_based_cf import (
    predict_user_ratings as ubcf_predict,      # (user_id, user_ids, item_ids, mat, user_sim, k)
    build_user_item_matrix as ubcf_build_matrix,
    load_movielens_1m as ubcf_load_ml1m
)

from src.item_based_cf import (
    predict_user_ratings as ibcf_predict,  # (user_id, user_ids, item_ids, mat, item_sim, top_k)
    build_item_user_matrix as ibcf_build_matrix,
    load_movielens_1m as ibcf_load_ml1m
)

from sklearn.metrics.pairwise import cosine_similarity


def get_relevant_dict(test_df_spark):
    """
    Build mapping: userId -> list of relevant items (test set)
    Input: Spark DataFrame with columns userId, movieId
    """
    rel = (
        test_df_spark.groupBy("userId")
        .agg(F.collect_list("movieId").alias("relevant"))
        .toPandas()
    )
    return {int(row["userId"]): list(row["relevant"]) for _, row in rel.iterrows()}


def evaluate_model_on_users(recommender_func, user_ids, relevant_dict, k):
    """
    recommender_func(uid) -> list of movieIds
    Returns averaged metrics
    """
    precisions = []
    recalls = []
    maps = []
    ndcgs = []

    n = len(user_ids)
    t0 = time.time()
    for i, uid in enumerate(user_ids, start=1):
        recs = recommender_func(uid)
        relevant = relevant_dict.get(uid, [])
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        maps.append(average_precision_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
        if i % 5 == 0 or i == n:
            elapsed = time.time() - t0
            print(f"  evaluated {i}/{n} users — elapsed {elapsed:.1f}s")
    return {
        "Precision@K": float(np.mean(precisions)),
        "Recall@K": float(np.mean(recalls)),
        "MAP@K": float(np.mean(maps)),
        "NDCG@K": float(np.mean(ndcgs)),
    }


def main(args):
    spark = get_spark("Evaluate-All-Models-Optimized")

    print("\nLoading test set (Spark parquet)...")
    test = spark.read.parquet(f"{args.data_path}/test_1m").select("userId", "movieId")
    print("Test rows:", test.count())

    # sample users: take distinct userIds, limit to num_users
    user_rows = test.select("userId").distinct().orderBy("userId").limit(args.num_users).collect()
    user_ids = [int(r.userId) for r in user_rows]
    print(f"Evaluating {len(user_ids)} users (sample). K = {args.k}")

    print("\nBuilding relevance dictionary (user -> list of relevant items)...")
    relevant_dict = get_relevant_dict(test)
    print("Users with relevant items in test:", len(relevant_dict))

    # ----------------------------
    # 1) ALS recommender (Spark)
    # ----------------------------
    print("\nLoading ALS model...")
    t_start = time.time()
    als_model = ALSModel.load(args.model_path)
    t_end = time.time()
    print(f"ALS model loaded in {t_end - t_start:.2f}s")

    def als_recommender(uid):
        # Use recommendForUserSubset for a single user
        df = spark.createDataFrame([(int(uid),)], ["userId"])
        recs = als_model.recommendForUserSubset(df, args.k).collect()
        if len(recs) == 0:
            return []
        return [int(r.movieId) for r in recs[0]["recommendations"]]

    print("Evaluating ALS (ranking metrics)...")
    t0 = time.time()
    als_scores = evaluate_model_on_users(als_recommender, user_ids, relevant_dict, args.k)
    print(f"ALS evaluation finished in {time.time() - t0:.2f}s")

    # ----------------------------
    # 2) User-Based CF
    # ----------------------------
    print("\nPreparing User-Based CF data (loading and building matrix)...")
    t0 = time.time()
    df_ubcf = ubcf_load_ml1m(args.data_path)   # pandas DataFrame
    mat_u, item_ids_u, user_ids_u = ubcf_build_matrix(df_ubcf)
    t1 = time.time()
    print(f"UBCF matrix built in {t1 - t0:.2f}s; shape (users x items) = {mat_u.shape}")

    # Compute user-user similarity ONCE (dense)
    print("Computing user-user cosine similarity (once)...")
    t0 = time.time()
    user_sim = cosine_similarity(mat_u, dense_output=True)
    time.sleep(300)
    t1 = time.time()
    print(f"User similarity computed in {t1 - t0:.2f}s; shape = {user_sim.shape}")

    # Recommender wrapper using precomputed similarity
    def ubcf_recommender(uid):
        preds = ubcf_predict(uid, user_ids_u, item_ids_u, mat_u, user_sim, k=args.k)
        sorted_idx = np.argsort(preds)[::-1]
        return [int(item_ids_u[i]) for i in sorted_idx[: args.k]]

    print("Evaluating User-Based CF (ranking metrics)...")
    t0 = time.time()
    ubcf_scores = evaluate_model_on_users(ubcf_recommender, user_ids, relevant_dict, args.k)
    print(f"User-CF evaluation finished in {time.time() - t0 + 300:.2f}s")

    # ----------------------------
    # 3) Item-Based CF
    # ----------------------------
    print("\nPreparing Item-Based CF data (loading and building matrix)...")
    t0 = time.time()
    df_ibcf = ibcf_load_ml1m(args.data_path)   # pandas DataFrame
    mat_i, item_ids_i, user_ids_i = ibcf_build_matrix(df_ibcf)  # items x users
    t1 = time.time()
    print(f"IBCF matrix built in {t1 - t0:.2f}s; shape (items x users) = {mat_i.shape}")

    print("Computing item-item cosine similarity (once)...")
    t0 = time.time()
    item_sim = cosine_similarity(mat_i, dense_output=True)
    time.sleep(300)
    t1 = time.time()
    print(f"Item similarity computed in {t1 - t0:.2f}s; shape = {item_sim.shape}")

    def ibcf_recommender(uid):
        preds = ibcf_predict(uid, user_ids_i, item_ids_i, mat_i, item_sim, args.k)
        sorted_idx = np.argsort(preds)[::-1]
        return [int(item_ids_i[i]) for i in sorted_idx[: args.k]]

    print("Evaluating Item-Based CF (ranking metrics)...")
    t0 = time.time()
    ibcf_scores = evaluate_model_on_users(ibcf_recommender, user_ids, relevant_dict, args.k)
    print(f"Item-CF evaluation finished in {time.time() - t0 + 300:.2f}s")

    # ----------------------------
    # Print results
    # ----------------------------
    print("\n=== Ranking Metrics (Averaged Across Users) ===")
    print(f"Top-K = {args.k} | Users = {len(user_ids)}\n")
    print(f"{'Model':15} {'Precision':10} {'Recall':10} {'MAP':10} {'NDCG':10}")
    print("-" * 60)
    print(f"{'ALS':15} {als_scores['Precision@K']:.4f}     {als_scores['Recall@K']:.4f}   {als_scores['MAP@K']:.4f}   {als_scores['NDCG@K']:.4f}")
    print(f"{'User-CF':15} {ubcf_scores['Precision@K']:.4f}     {ubcf_scores['Recall@K']:.4f}   {ubcf_scores['MAP@K']:.4f}   {ubcf_scores['NDCG@K']:.4f}")
    print(f"{'Item-CF':15} {ibcf_scores['Precision@K']:.4f}     {ibcf_scores['Recall@K']:.4f}   {ibcf_scores['MAP@K']:.4f}   {ibcf_scores['NDCG@K']:.4f}")

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--model-path", default="models/als_1m")
    parser.add_argument("--k", type=int, default=10, help="Top-K for ranking metrics")
    parser.add_argument("--num-users", type=int, default=20, help="Number of users to sample")
    args = parser.parse_args()
    main(args)
