import sys
import pickle
import shap
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#sys.path.append('../')
from utils.shap_utils import shap_dotplot_by_risk

dir_res = '/gpfs/work4/0/prjs1086/dimafx/exp/dsquame_classification_patchdictionary_n37/dsquame_classification_patchdictionary_n37'
dir_main = '/gpfs/work4/0/prjs1086/dimafx'
path_rna_gene_sets = "/gpfs/work4/0/prjs1086/dimafx/data/rna/dsquame/dsquame_gene_sets.csv"
fold = 2
nr_ref_samples = 512
nr_proto = 37

dir_analysis_out = "/gpfs/work4/0/prjs1086/dimafx/analysis"


# Insert your own WSI annotations here
# wsi_annotations = ['Tumor', '--', 'Tissue edge: stroma > tumor', 'Tumor', 'Adipose > stroma', '--', 'Stroma > adipose', 'Adipose > stroma', 'Tumor: solid pattern, abundant cytoplasm', 'Tumor >> adipose', 'Stroma > tumor > immune cells', 'Tumor > stroma > immune cells', 'Normal ducts/lobules', 'Stroma > adipose', 'Edge stainings, blood vessels', 'Stroma >  immune cells']

path_concepts = "/gpfs/work4/0/prjs1086/cchia/P8/data/patch_dictionary_v2/3_tessapath_dev_v2/uni-cls/concepts.csv"
df_concepts= pd.read_csv(path_concepts)

# Name 16 W0-W15 clusters
#l_wsi_annotations = [f'W{i}' for i in range(16)]
l_wsi_annotations = df_concepts["concept"].tolist()


###############
# Loading pathways names and initializing cluster identities
hallmarks = pd.read_csv(path_rna_gene_sets)
# DEBUG for mm_dataset.py# Remove first column
# hallmarks = hallmarks.iloc[:, 1:]
# If column name sttarts with "HALLMARK", remove it. Otherwise keep whole cell value
hallmarks = np.array([' '.join(x[9:].split('_')) if x.startswith('HALLMARK') else x for x in hallmarks.columns])
hallmarks_ids = np.array([f'R{i}' for i, item in enumerate (hallmarks)])
hallmarks_names = np.array([f'R{i}: {item}' for i, item in enumerate (hallmarks)])

cluster_ids = np.array([f'W{i}' for i, item in enumerate (l_wsi_annotations)])
cluster_names = np.array([f'W{i}: {item}' for i, item in enumerate (l_wsi_annotations)])

all_feats_names = [f"{name}" for name in cluster_names] + [f"{name}" for name in hallmarks_names]
all_short_feat_names = np.concatenate((hallmarks_ids, cluster_ids))


# Obtain shap values
path_shap_test_modal_pkl = os.path.join(dir_res, f'Fold_{fold}/post_training/shap/modal', f'shap_all_test_{nr_ref_samples}.pkl')
shap_dict = pickle.load(open(path_shap_test_modal_pkl, 'rb')) #(210, 65, 256), samples/features/dim
shap_values = np.asarray(shap_dict['shap values'])
# Classifier output gives one SHAP set per class: (samples, features, dim, classes).
# With softmax, P(class 1) = sigmoid(z1 - z0), so explain the log-odds z1 - z0. SHAP is additive over outputs.
assert shap_values.ndim == 4 and shap_values.shape[-1] == 2, f"Expected binary classifier SHAP values, got shape {shap_values.shape}"
shap_logodds = shap_values[..., 1] - shap_values[..., 0]  # (samples, features, dim)
all_feats_shap = np.sum(shap_logodds, axis=2)  # (samples, features)
assert all_feats_shap.shape[1] == len(all_feats_names), (all_feats_shap.shape, len(all_feats_names))

# Obtain predicted probabilities, (samples, classes)
data_results_path = os.path.join(dir_res, f'Fold_{fold}/post_training','predictions_test.pkl')
scores_results = pickle.load(open(data_results_path, 'rb'))


#Plot shap plot vs predicted log-odds
dir_out_shap_vs_risk = os.path.join(dir_analysis_out, 'dsquame_classification_patchdictionary_n37', 'unimodal_shap_vs_probabilities')
os.makedirs(dir_out_shap_vs_risk, exist_ok=True)
path_out_shap_vs_risk_score = os.path.join(dir_out_shap_vs_risk, f'shap_vs_risk_fold_{fold}.png')

# Color by the explained output: predicted log-odds of class 1, log(p1 / p0) = z1 - z0 (0 = undecided)
probs = np.clip(np.asarray(scores_results['Probabilities']), 1e-12, 1.0)
pred_logodds = np.log(probs[:, 1]) - np.log(probs[:, 0])

shap_dotplot_by_risk(all_feats_shap, all_feats_names, pred_logodds, cmap="coolwarm",
                     save_path=path_out_shap_vs_risk_score, log2=False, cbar_title='Log-odds\nclass 1')

# Plot SHAP of single sample
def get_plot_single_sample(sample, all_feats_shap, samples, all_feats_names, path_out_shap_vs_risk_score, max_display=15):
    """ Plot the log-odds SHAP values (summed over the embedding dim) of a single sample."""
    sample_index = list(samples).index(sample)
    shap_of_one_sample = all_feats_shap[sample_index:sample_index + 1]
    # Plot shap
    sample_expl = shap.Explanation(values=shap_of_one_sample, feature_names=all_feats_names)
    shap.plots.bar(sample_expl[0], max_display=max_display, show=False)
    if path_out_shap_vs_risk_score is not None:
        plt.savefig(path_out_shap_vs_risk_score, dpi=500, bbox_inches='tight')
    plt.show()
    
# We can also plot the local multimodal feature importance plot of one sample
# Take a test id of the specified fold
id = '3_P298_T0_S1_B1_V1_L1' #case
path_out_shap_vs_risk_score = os.path.join(dir_out_shap_vs_risk, f'shap_vs_risk_fold_{fold}_case_sample{id}.png')

id = '3_P403_T2_S1_B1_V1_L1' #control
path_out_shap_vs_risk_score = os.path.join(dir_out_shap_vs_risk, f'shap_vs_risk_fold_{fold}_control_sample{id}.png')


# Delete previous plt object
plt.clf()

get_plot_single_sample(id, all_feats_shap, shap_dict['Samples'], all_feats_names, path_out_shap_vs_risk_score, max_display=15)