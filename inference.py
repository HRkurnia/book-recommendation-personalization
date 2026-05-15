"""
inference.py — SageMaker inference handler for ALS/LightFM models.

SageMaker calls: model_fn, input_fn, predict_fn, output_fn
"""
import json
import os
import pickle

import numpy as np


def model_fn(model_dir):
    """Load model artifacts from model_dir."""
    model_path = os.path.join(model_dir, 'model.pkl')
    with open(model_path, 'rb') as f:
        artifacts = pickle.load(f)
    return artifacts


def input_fn(request_body, request_content_type):
    """Parse input: expects JSON with user_id and optional N."""
    if request_content_type == 'application/json':
        data = json.loads(request_body)
        return data
    raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, artifacts):
    """Generate top-N recommendations for a user."""
    user_id = input_data.get('user_id')
    n = input_data.get('n', 10)
    model_type = artifacts['model_type']
    model = artifacts['model']
    users = artifacts['users']
    items = artifacts['items']
    u2idx = artifacts['u2idx']
    R = artifacts['R']

    if user_id not in u2idx:
        return {'user_id': user_id, 'recommendations': [], 'error': 'user_not_found'}

    uidx = u2idx[user_id]

    if model_type == 'als':
        ids, scores = model.recommend(uidx, R[uidx], N=n, filter_already_liked_items=True)
        recs = [{'item_id': items[i], 'score': float(s)} for i, s in zip(ids, scores)]
    else:
        # LightFM
        ds = artifacts['dataset']
        uf = artifacts['user_features']
        itf = artifacts['item_features']
        uid_map, _, iid_map, _ = ds.mapping()
        rev_iid = {v: k for k, v in iid_map.items()}

        lfm_u = uid_map[user_id]
        scores = model.predict(lfm_u, np.arange(len(iid_map)),
                               user_features=uf, item_features=itf)
        # Exclude train items
        train_items = set(R[uidx].indices)
        train_lfm = {iid_map[items[i]] for i in train_items if items[i] in iid_map}
        ranking = np.argsort(-scores)
        top_items = [i for i in ranking if i not in train_lfm][:n]
        recs = [{'item_id': rev_iid[i], 'score': float(scores[i])} for i in top_items]

    return {'user_id': user_id, 'recommendations': recs}


def output_fn(prediction, accept):
    """Serialize output as JSON."""
    return json.dumps(prediction), 'application/json'
