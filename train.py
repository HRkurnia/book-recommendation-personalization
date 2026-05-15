"""
train.py — ALS + LightFM training script for AWS SageMaker.

Usage (SageMaker):
    Uploaded as source_dir in SKLearn/Framework estimator.
    SageMaker sets env vars: SM_MODEL_DIR, SM_CHANNEL_TRAINING, SM_OUTPUT_DATA_DIR

Usage (local):
    python train.py --data-dir ./data --model-dir ./model_output --model-type als
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ============================================================
# Data Processing
# ============================================================
EVENT_WEIGHTS = {'page_view': 1, 'wishlist': 3, 'add_to_cart': 5, 'purchase': 12}


def load_and_prepare(data_dir):
    """Load CSVs and build weighted interactions."""
    log = pd.read_csv(os.path.join(data_dir, 'interactions.csv'))
    log['weight'] = log['event_type'].map(EVENT_WEIGHTS).fillna(1).astype(float)
    interactions = log.groupby(['user_id', 'item_id'])['weight'].sum().reset_index()

    items_df = None
    items_path = os.path.join(data_dir, 'items.csv')
    if os.path.exists(items_path):
        items_df = pd.read_csv(items_path)

    users_df = None
    users_path = os.path.join(data_dir, 'users.csv')
    if os.path.exists(users_path):
        users_df = pd.read_csv(users_path)

    return interactions, items_df, users_df


def apply_kcore(df, k):
    """Iterative k-core filtering."""
    for _ in range(30):
        prev = len(df)
        uc = df.groupby('user_id').size()
        df = df[df['user_id'].isin(uc[uc >= k].index)]
        ic = df.groupby('item_id').size()
        df = df[df['item_id'].isin(ic[ic >= k].index)]
        if len(df) == prev:
            break
    return df


def build_sparse_matrix(df, users, items):
    """Build CSR matrix from interactions dataframe."""
    u2idx = {u: i for i, u in enumerate(users)}
    i2idx = {s: i for i, s in enumerate(items)}
    row = df['user_id'].map(u2idx).values
    col = df['item_id'].map(i2idx).values
    val = df['weight'].values.astype(np.float32)
    R = csr_matrix((val, (row, col)), shape=(len(users), len(items)))
    return R, u2idx, i2idx


# ============================================================
# Model Training
# ============================================================
def train_als(R, factors=150, iterations=15, regularization=0.1):
    """Train ALS model using implicit library."""
    from implicit.als import AlternatingLeastSquares
    model = AlternatingLeastSquares(
        factors=factors, iterations=iterations,
        regularization=regularization, random_state=42
    )
    model.fit(R)
    return model


def train_lightfm(interactions_df, items_df, users_df, users, items,
                  components=64, epochs=30, lr=0.05):
    """Train LightFM hybrid model."""
    from lightfm import LightFM
    from lightfm.data import Dataset

    # Prepare features
    item_features_labels = []
    if items_df is not None and 'category' in items_df.columns:
        item_features_labels = [f"cat:{c}" for c in items_df['category'].dropna().unique()]

    user_features_labels = []
    if users_df is not None and 'city' in users_df.columns:
        top_cities = users_df['city'].value_counts().head(20).index.tolist()
        user_features_labels = [f"city:{c}" for c in top_cities] + ['city:other']

    ds = Dataset()
    ds.fit(users=users, items=items,
           user_features=user_features_labels or None,
           item_features=item_features_labels or None)

    lfm_interactions, lfm_weights = ds.build_interactions(
        ((r['user_id'], r['item_id'], r['weight']) for _, r in interactions_df.iterrows())
    )

    # Build feature matrices
    uf_sparse, if_sparse = None, None
    if user_features_labels and users_df is not None:
        city_map = users_df.drop_duplicates('user_id').set_index('user_id')['city'].to_dict()
        top_set = set(top_cities)
        uf_list = [(u, [f"city:{city_map[u] if city_map.get(u) in top_set else 'other'}"])
                   if u in city_map else (u, []) for u in users]
        uf_sparse = ds.build_user_features(uf_list, normalize=False)

    if item_features_labels and items_df is not None:
        cat_map = items_df.drop_duplicates('item_id').set_index('item_id')['category'].to_dict()
        if_list = [(i, [f"cat:{cat_map[i]}"] if i in cat_map and pd.notna(cat_map[i]) else [])
                   for i in items]
        if_sparse = ds.build_item_features(if_list, normalize=False)

    model = LightFM(loss='warp', no_components=components,
                    learning_rate=lr, user_alpha=1e-6, item_alpha=1e-6)
    model.fit(lfm_interactions, user_features=uf_sparse, item_features=if_sparse,
              sample_weight=lfm_weights, epochs=epochs, num_threads=4)

    return model, ds, uf_sparse, if_sparse


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-type', type=str, default='als', choices=['als', 'lightfm'])
    parser.add_argument('--factors', type=int, default=150)
    parser.add_argument('--iterations', type=int, default=15)
    parser.add_argument('--regularization', type=float, default=0.1)
    parser.add_argument('--components', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--kcore', type=int, default=7)
    # SageMaker env vars
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model_output'))
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', './data'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Training {args.model_type.upper()} model")
    print(f"Data dir: {args.data_dir}")
    print(f"Model dir: {args.model_dir}")

    # Load data
    t0 = time.time()
    interactions, items_df, users_df = load_and_prepare(args.data_dir)
    print(f"Loaded {len(interactions):,} interactions in {time.time()-t0:.1f}s")

    # K-core filter
    interactions = apply_kcore(interactions, args.kcore)
    users = sorted(interactions['user_id'].unique())
    items = sorted(interactions['item_id'].unique())
    n_users, n_items = len(users), len(items)
    density = len(interactions) / (n_users * n_items) * 100
    print(f"After k-core={args.kcore}: {n_users:,} users, {n_items:,} items, density={density:.3f}%")

    # Train
    t0 = time.time()
    if args.model_type == 'als':
        R, u2idx, i2idx = build_sparse_matrix(interactions, users, items)
        model = train_als(R, args.factors, args.iterations, args.regularization)
        # Save
        artifacts = {
            'model': model, 'users': users, 'items': items,
            'u2idx': u2idx, 'i2idx': i2idx, 'R': R,
            'model_type': 'als',
            'hyperparameters': {'factors': args.factors, 'iterations': args.iterations,
                                'regularization': args.regularization, 'kcore': args.kcore}
        }
    else:
        model, ds, uf_sparse, if_sparse = train_lightfm(
            interactions, items_df, users_df, users, items,
            args.components, args.epochs, args.learning_rate
        )
        R, u2idx, i2idx = build_sparse_matrix(interactions, users, items)
        artifacts = {
            'model': model, 'dataset': ds, 'user_features': uf_sparse,
            'item_features': if_sparse, 'users': users, 'items': items,
            'u2idx': u2idx, 'i2idx': i2idx, 'R': R,
            'model_type': 'lightfm',
            'hyperparameters': {'components': args.components, 'epochs': args.epochs,
                                'learning_rate': args.learning_rate, 'kcore': args.kcore}
        }

    train_time = time.time() - t0
    print(f"Trained in {train_time:.1f}s")

    # Save model
    model_path = os.path.join(args.model_dir, 'model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(artifacts, f)
    print(f"Model saved to {model_path}")

    # Save metrics
    metrics = {
        'model_type': args.model_type,
        'n_users': n_users, 'n_items': n_items,
        'density_pct': round(density, 4),
        'train_time_sec': round(train_time, 1),
        'n_interactions': len(interactions),
    }
    metrics_path = os.path.join(args.output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {metrics_path}")


if __name__ == '__main__':
    main()
