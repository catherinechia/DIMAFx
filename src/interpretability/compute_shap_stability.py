import os
import sys 
import argparse
import pickle

import numpy as np
from itertools import combinations

#sys.path.append('../')
from utils.general_utils import set_seed, save_pkl

def jaccard_similarity(set_a, set_b):
    """ Compute the Jaccard similarity between two sets. """
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 1.0

def top_k_features(shap_values, feature_names, k=10):
    """Get top-k features by mean absolute SHAP value."""
    # shap_values: [N_samples, N_features, ...] -> average over all axes except features
    reduce_axes = tuple(ax for ax in range(shap_values.ndim) if ax != 1)
    mean_abs_shap = np.mean(np.absolute(shap_values), axis=reduce_axes)
    top_k_idx = np.argsort(mean_abs_shap)[::-1][:k]
    return set(feature_names[i] for i in top_k_idx)

def jaccard_stability(shap_values, feature_names, k=10, n_bootstrap=100):
    """ Bootstrap test set, compute top-k features each time, report pairwise Jaccard stability.  """
    n_samples = shap_values.shape[0]
    feature_names = np.array(feature_names)
    top_k_sets = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        idx = np.random.choice(n_samples, size=n_samples, replace=True)
        shap_boot = shap_values[idx]
        top_k = top_k_features(shap_boot, feature_names, k=k)
        top_k_sets.append(top_k)
    # Pairwise Jaccard
    scores = [
        jaccard_similarity(a, b)
        for a, b in combinations(top_k_sets, 2)
    ]
    mean_j = np.mean(scores)
    std_j = np.std(scores)
    print(f"Top-{k} Jaccard Stability: {mean_j:.3f} ± {std_j:.3f} (n_bootstrap={n_bootstrap})")
    results_j = {"mean": mean_j, "std": std_j}
    return results_j

def get_shap_dict(dir, shap_ref_distr):
    shap_results_fold_dir = os.path.join(dir, f'shap_all_test_{shap_ref_distr}.pkl')
    shap_dict = pickle.load(open(shap_results_fold_dir, 'rb'))     
    shap_values = np.sum(shap_dict['shap values'], axis=2)
    feature_names = list(shap_dict['Feature names'])
    return shap_values, feature_names

def compute_shap_stability(dir_res, folds, repr_type, bootstrap_samples=100):
    # For each fold
    for i in range(folds):
        results = {}
        fold_dir = os.path.join(dir_res, f'Fold_{i}/post_training/shap/{repr_type}')
        for k in [5, 10, 20, 30]:
            print(f"Evaluating stability for top-{k} features...")
            shap_values, feature_names = get_shap_dict(fold_dir, args.shap_refdist_n)
            results_k =jaccard_stability(shap_values, feature_names, k=k, n_bootstrap=bootstrap_samples)
            results[f"Top_{k}"] = results_k
        save_pkl(fold_dir, f'shap_stability_{repr_type}.pkl', results)

def main(args):
    set_seed(args.seed)
    # Check stability for unimodal representations
    compute_shap_stability(args.dir_res, args.folds, 'modal', args.bootstrap_samples)
    # Check stability for unimodal representations
    compute_shap_stability(args.dir_res, args.folds, 'post_attn', args.bootstrap_samples)

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dir_res', type=str, default='Results main directory')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--bootstrap_samples', type=int, default=100)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--shap_refdist_n', type=int, default=512)
    args = parser.parse_args()
    
    main(args)