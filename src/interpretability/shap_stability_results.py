import pickle
import numpy as np

dir_res = '/gpfs/work4/0/prjs1086/dimafx/exp/brca_reproduce_n16/brca_reproduce_n16'
folds=5

# Info: the "modal" SHAP values (16 WSI prototypes + 50 Hallmark pathways = 66 features). Resampled 100 times. All 66 features ranked by |SHAP| mean, top 5 kept. Jaccard similarity calculated for 100 top 5 (4950 pairs comparisons)
print("Unimodal")
results = {}
for k in [5, 10, 20, 30]:
    means = []
    results = f"& $k  = {k}$ "
    for i in range(folds):
        path_stab_modal_pkl = os.path.join(dir_res, f'Fold_{i}/post_training/shap/modal', 'shap_stability_modal.pkl')
        shap_stab_dict = pickle.load(open(path_stab_modal_pkl, 'rb'))    
        results += f"& {shap_stab_dict[f'Top_{k}']['mean']:.3f}±{shap_stab_dict[f'Top_{k}']['std']:.3f} "
        means.append(shap_stab_dict[f'Top_{k}']['mean'])
    mean_over_folds = np.mean(means)
    std_over_folds = np.std(means)
    results += f"& {mean_over_folds:.3f}±{std_over_folds:.3f} \\\\"
    print(results)

#########################
print("Multimodal")
results = {}
for k in [5, 10, 20, 30]:
    means = []
    results = f"& $k  = {k}$ "
    for i in range(folds):
        path_stab_multimodal_pkl = os.path.join(dir_res, f'Fold_{i}/post_training/shap/post_attn', 'shap_stability_post_attn.pkl')
        shap_stab_dict = pickle.load(open(path_stab_multimodal_pkl, 'rb'))   
        results += f"& {shap_stab_dict[f'Top_{k}']['mean']:.3f}±{shap_stab_dict[f'Top_{k}']['std']:.3f} "
        means.append(shap_stab_dict[f'Top_{k}']['mean'])
    mean_over_folds = np.mean(means)
    std_over_folds = np.std(means)
    results += f"& {mean_over_folds:.3f}±{std_over_folds:.3f} \\\\"
    print(results)