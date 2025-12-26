# Spark Movie Recommendation System

This project is a **learning-oriented implementation** of a movie recommendation system using **Apache Spark**.
It explores **ALS-based collaborative filtering** and compares it with **classical user-based and item-based CF** approaches.

The goal of this project is to understand:
- How Spark ML pipelines work
- Why ALS scales better than traditional CF
- How recommendation models are evaluated in practice

---

## Models Implemented

1. **ALS (Alternating Least Squares)** — implemented using Spark MLlib  
2. **User-Based Collaborative Filtering** — classical cosine similarity baseline  
3. **Item-Based Collaborative Filtering** — classical cosine similarity baseline  

---

## Why Apache Spark?

Classical collaborative filtering approaches require computing full similarity matrices, which:
- Become expensive as users/items grow
- Do not scale well beyond small datasets

Spark allows:
- Distributed data processing
- Scalable matrix factorization (ALS)
- Efficient handling of large rating datasets

This project uses Spark mainly for the **ALS pipeline**.

---

## Dataset

This project uses the **MovieLens 1M** dataset.

The dataset is **not included** in this repository.

Download from:
https://grouplens.org/datasets/movielens/1m/

After downloading, place the files inside a `data/` directory.

---

## Project Structure

src/ → preprocessing, training, evaluation, and recommendation scripts
commands.txt → step-by-step commands to run the pipeline

---

## How to Run (Step-by-Step)

All commands are documented in `commands.txt`.

Typical workflow:
1. Preprocess data
2. Train ALS model
3. Evaluate models
4. Generate recommendations

---

## Limitations

- Classical CF baselines are **not scalable**
- No automated hyperparameter tuning
- Single-node Spark testing

---

## Status

This project is part of an ongoing learning process focused on **distributed systems and recommender systems**.
