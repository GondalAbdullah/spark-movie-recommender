import time
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def load_movielens_100k(data_path="data"):
    """
    Loads MovieLens 100k using pandas (sequential, non-parallel)
    """
    ratings = pd.read_csv(
        f"{data_path}/u.data",
        sep="\t",
        names=["userId", "movieId", "rating", "timestamp"],
        engine="python"
    )
    
    movies = pd.read_csv(
        f"{data_path}/u.item",
        sep="|",
        names=["movieId", "title"] + [f"col{i}" for i in range(22)],
        encoding="iso-8859-1",
        engine="python"
    )[["movieId", "title"]]

    df = ratings.merge(movies, on="movieId", how="left")
    return df


def build_user_item_matrix(df):
    """
    Converts ratings DataFrame into a pivot table: rows=users, columns=movies
    """
    return df.pivot_table(index='userId', columns='movieId', values='rating').fillna(0)


def compute_user_similarity(matrix):
    """
    Computes cosine similarity between users (sequential!)
    """
    return cosine_similarity(matrix)


def predict_ratings(user_sim, matrix):
    """
    Predicts ratings using similarity-weighted average.
    Classic CF formula.
    """
    sim_sum = np.abs(user_sim).sum(axis=1)




    sim_sum[sim_sum == 0] = 1e-9
    
    pred = user_sim.dot(matrix) / sim_sum[:, None]
    return pred


def get_top_n_recommendations(user_id, pred_matrix, movie_ids, n=10):
    """
    Returns top-N predicted movies for a given user.
    """
    user_idx = user_id - 1  # zero-based indexing
    user_ratings = pred_matrix[user_idx]




    top_indices = np.argsort(user_ratings)[::-1][:n]
    return movie_ids[top_indices], user_ratings[top_indices]


def compute_rmse(pred_matrix, true_matrix):
    """
    Compute RMSE using only the non-zero actual ratings.
    """
    mask = true_matrix > 0
    mse = ((pred_matrix[mask] - true_matrix[mask]) ** 2).mean()
    return np.sqrt(mse)



def run_traditional_cf(data_path="data", top_n=10):

    df = load_movielens_100k(data_path)

    matrix = build_user_item_matrix(df)
    movie_ids = matrix.columns.values
    matrix_np = matrix.to_numpy()

    t0 = time.time()
    user_sim = compute_user_similarity(matrix_np)
    time.sleep(20)
    t1 = time.time()
    sim_time = t1 - t0

    t2 = time.time()
    pred_matrix = predict_ratings(user_sim, matrix_np)
    t3 = time.time()
    pred_time = t3 - t2

    # ---- NEW: RMSE computation ----
    rmse = compute_rmse(pred_matrix, matrix_np)

    total_time = sim_time + pred_time
    print(f"\n============== SUMMARY ==============")
    print(f"Sequential Traditional CF — TOTAL TIME: {total_time:.2f} seconds")
    print(f"(Filtering: {sim_time:.2f}s, Prediction: {pred_time:.2f}s)")
    print(f"RMSE on training data: {rmse:.4f}")
    print("=====================================\n")

    sample_user = 100
    top_movies, scores = get_top_n_recommendations(sample_user, pred_matrix, movie_ids, top_n)

    print(f"\nTop {top_n} recommendations for User {sample_user}:")
    for mid, score in zip(top_movies, scores):
        title = df[df.movieId == mid]["title"].iloc[0]
        print(f"{title} (Predicted Score: {score:.2f})")


if __name__ == "__main__":
    run_traditional_cf()
