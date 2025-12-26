"""
User-based Collaborative Filtering (MovieLens 1M version)

Usage (from project root):
    python -m src.user_based_cf --data-path data --user-id 100 --top-n 10 --k 20
"""

import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time
import os


# ----------------------------------------------------
# Load MovieLens 1M
# ----------------------------------------------------
def load_movielens_1m(data_path: str) -> pd.DataFrame:
    ratings_path = os.path.join(data_path, "ratings.dat")
    movies_path = os.path.join(data_path, "movies.dat")

    if not (os.path.exists(ratings_path) and os.path.exists(movies_path)):
        raise FileNotFoundError(
            f"MovieLens 1M files not found in {data_path}. Expect ratings.dat and movies.dat."
        )

    ratings = pd.read_csv(
        ratings_path,
        sep="::",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python",
    )

    movies = pd.read_csv(
        movies_path,
        sep="::",
        names=["movieId", "title", "genres"],
        engine="python",
        encoding="iso-8859-1"
    )

    df = ratings.merge(movies[["movieId", "title"]], on="movieId", how="left")

    print("\nLoaded MovieLens 1M dataset:")
    print(f"  - {df.shape[0]:,} ratings")
    print(f"  - {df.userId.nunique():,} users")
    print(f"  - {df.movieId.nunique():,} items")

    return df


# ----------------------------------------------------
# Matrix builder (user × item)
# ----------------------------------------------------
def build_user_item_matrix(df: pd.DataFrame):
    """
    Build user × item sparse matrix (CSR format)
    """
    user_ids = np.sort(df.userId.unique())
    item_ids = np.sort(df.movieId.unique())

    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {m: i for i, m in enumerate(item_ids)}

    rows = df.userId.map(user_index).to_numpy()
    cols = df.movieId.map(item_index).to_numpy()
    vals = df.rating.astype(np.float32).to_numpy()

    mat = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(user_ids), len(item_ids)),
        dtype=np.float32,
    )

    return mat, user_ids, item_ids


# ----------------------------------------------------
# User similarity
# ----------------------------------------------------
def compute_user_similarity(mat: csr_matrix):
    """
    Compute user-user cosine similarity.
    WARNING: produces dense matrix user × user.
    OK for ML-1M (~6000 users).
    """
    print("\nComputing user-user cosine similarity...")
    t0 = time.time()

    sim = cosine_similarity(mat, dense_output=True)

    t1 = time.time()

    print(f"User similarity matrix shape: {sim.shape}")
    print(f"Similarity calculation time: {t1 - t0:.2f} seconds")

    return sim, t1 - t0


# ----------------------------------------------------
# Predict user ratings using user-based CF
# ----------------------------------------------------
def predict_user_ratings(target_user, user_ids, item_ids, mat, sim, k=20):
    """
    Predict ratings for a target user using top-k similar users.
    """
    # Find target user index
    idx = np.where(user_ids == target_user)[0]
    if len(idx) == 0:
        raise ValueError(f"User {target_user} not found.")
    uidx = int(idx[0])

    # Similarity vector for target user
    sim_vec = sim[uidx]

    # Exclude self
    sim_vec[uidx] = 0

    # Find top-k similar users
    topk_idx = np.argpartition(-sim_vec, k)[:k]
    topk_sims = sim_vec[topk_idx]

    # Get their ratings matrix: top-k users × items
    topk_ratings = mat[topk_idx].toarray()

    # Weighted sum
    num = np.dot(topk_sims, topk_ratings)
    den = np.abs(topk_sims).sum()

    # Avoid divide by zero
    preds = num / den if den > 0 else np.zeros_like(num)

    return preds


# ----------------------------------------------------
# Recommendation extraction
# ----------------------------------------------------
def top_n_recommendations(user_id, preds, item_ids, df, n=10):
    rated = set(df[df.userId == user_id].movieId.unique())

    sorted_idx = np.argsort(preds)[::-1]

    out = []
    for idx in sorted_idx:
        mid = int(item_ids[idx])
        if mid in rated:
            continue
        title = df[df.movieId == mid]["title"].iloc[0]
        out.append((mid, title, float(preds[idx])))
        if len(out) == n:
            break

    return out


def print_user_history(user_id, df, top_n=10):
    print(f"\nTop rated movies by user {user_id}:")
    history = df[df.userId == user_id].sort_values(by="rating", ascending=False).head(top_n)
    for _, row in history.iterrows():
        print(f"  {row.movieId:<5} | {row.title[:40]:40} | rating={row.rating}")


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------
def main(args):
    print("\nLoading ML-1M...")
    df = load_movielens_1m(args.data_path)

    print("\nBuilding user-item matrix...")
    t0 = time.time()
    mat, user_ids, item_ids = build_user_item_matrix(df)
    t1 = time.time()

    print(f"Matrix built in {t1 - t0:.2f}s")
    print(f"Matrix shape: {mat.shape}  (users × items)")

    sim, sim_time = compute_user_similarity(mat)

    print(f"\nPredicting for user {args.user_id}...")
    t2 = time.time()
    preds = predict_user_ratings(args.user_id, user_ids, item_ids, mat, sim, k=args.k)
    t3 = time.time()

    print(f"Prediction time: {t3 - t2:.2f} seconds")

    print_user_history(args.user_id, df, top_n=10)

    print(f"\nTop-{args.top_n} User-Based CF recommendations:")
    recs = top_n_recommendations(args.user_id, preds, item_ids, df, n=args.top_n)
    for mid, title, score in recs:
        print(f"  {mid:<5} | {title[:40]:40} | predicted={score:.4f}")

    print("\n=== TIMING SUMMARY ===")
    print(f"Matrix build:       {t1 - t0:.2f} s")
    print(f"Similarity compute: {sim_time:.2f} s")
    print(f"Prediction:         {t3 - t2:.2f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default="data")
    parser.add_argument("--user-id", type=int, default=100)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--k", type=int, default=20)
    args = parser.parse_args()
    main(args)
