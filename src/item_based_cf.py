"""
Item-based Collaborative Filtering (MovieLens 1M version)

Usage:
    python -m src.item_based_cf --data-path data --user-id 100 --top-n 10 --k 20
"""

import argparse
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import time
import os


def load_movielens_1m(data_path: str) -> pd.DataFrame:
    """
    Loads MovieLens 1M dataset:
    - ratings.dat   => userId::movieId::rating::timestamp
    - movies.dat    => movieId::title::genres
    """
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
        encoding="iso-8859-1",
    )

    df = ratings.merge(movies[["movieId", "title"]], on="movieId", how="left")

    print("Loaded MovieLens 1M dataset.")
    print(f"- {df.shape[0]:,} ratings")
    print(f"- {df['userId'].nunique():,} users")
    print(f"- {df['movieId'].nunique():,} movies")

    return df


def build_item_user_matrix(df: pd.DataFrame):
    """
    Build an items × users sparse matrix (csr_matrix) for 1M dataset.
    """
    user_ids = np.sort(df.userId.unique())
    item_ids = np.sort(df.movieId.unique())

    user_index = {u: i for i, u in enumerate(user_ids)}
    item_index = {m: i for i, m in enumerate(item_ids)}

    rows = df.movieId.map(item_index).to_numpy()
    cols = df.userId.map(user_index).to_numpy()
    vals = df.rating.astype(np.float32).to_numpy()

    mat = csr_matrix(
        (vals, (rows, cols)), shape=(len(item_ids), len(user_ids)), dtype=np.float32
    )

    return mat, item_ids, user_ids


def compute_item_similarity(mat: csr_matrix):
    """
    Compute item-item cosine similarity.
    WARNING: Produces a dense item×item matrix. OK for ML-1M (~4000 items).
    """
    print("\nComputing item-item cosine similarity...")
    t0 = time.time()

    sim = cosine_similarity(mat, dense_output=True)

    t1 = time.time()
    print(f"Similarity matrix shape: {sim.shape}")
    print(f"Similarity computation time: {t1 - t0:.2f} seconds")

    return sim, t1 - t0


def predict_user_ratings(user_id, user_ids, item_ids, mat, sim, k=20):
    """
    Predict ratings using item-based CF with top-k neighbors per item.
    """
    # Get user column index
    user_idx = np.where(user_ids == user_id)[0]
    if len(user_idx) == 0:
        raise ValueError(f"User {user_id} not found.")
    uidx = int(user_idx[0])

    user_ratings = mat[:, uidx].toarray().reshape(-1)
    n_items = mat.shape[0]

    preds = np.zeros(n_items, dtype=np.float32)

    # Precompute top-k neighbors for efficiency
    top_k_idx = np.argpartition(-sim, k, axis=1)[:, :k]

    for i in range(n_items):
        neigh = top_k_idx[i]
        sims = sim[i, neigh]
        ratings = user_ratings[neigh]

        mask = ratings > 0
        if not mask.any():
            preds[i] = 0
        else:
            preds[i] = np.dot(sims[mask], ratings[mask]) / (np.abs(sims[mask]).sum())

    return preds


def top_n_recommendations(user_id, preds, item_ids, df, n=10):
    """Return top-N recommended movie titles."""
    rated_movies = set(df[df.userId == user_id].movieId.unique())

    sorted_idx = np.argsort(preds)[::-1]
    out = []

    for idx in sorted_idx:
        movie = int(item_ids[idx])
        if movie in rated_movies:
            continue
        title = df[df.movieId == movie]["title"].iloc[0]
        out.append((movie, title, float(preds[idx])))
        if len(out) == n:
            break

    return out


def print_user_history(user_id, df, top_n=10):
    print(f"\nHistory of user {user_id}:")
    history = df[df.userId == user_id].sort_values(by="rating", ascending=False).head(top_n)
    for _, row in history.iterrows():
        print(f"  {row.movieId:<5} | {row.title[:40]:40} | rating={row.rating}")







def main(args):
    print("Loading ML-1M dataset...")
    df = load_movielens_1m(args.data_path)

    print("\nBuilding item-user matrix...")
    t0 = time.time()
    mat, item_ids, user_ids = build_item_user_matrix(df)
    t1 = time.time()
    print(f"Matrix built in {t1 - t0:.2f} seconds")
    print(f"Matrix shape: {mat.shape}")

    sim, sim_time = compute_item_similarity(mat)

    print(f"\nPredicting ratings for user {args.user_id}...")
    t2 = time.time()
    preds = predict_user_ratings(args.user_id, user_ids, item_ids, mat, sim, k=args.k)
    t3 = time.time()

    print(f"Prediction time: {t3 - t2:.2f} seconds")

    print_user_history(args.user_id, df, top_n=10)

    print(f"\nTop-{args.top_n} recommendations:")
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




