import sys
import pickle
import shap
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#sys.path.append('../')
from utils.shap_utils import shap_dotplot_by_risk

dir_res = '/gpfs/work4/0/prjs1086/dimafx/exp/brca_reproduce_n16/brca_reproduce_n16'
dir_main = '/gpfs/work4/0/prjs1086/dimafx'
path_rna_gene_sets = os.path.join(dir_main, 'data/rna/brca', 'hallmarks_signatures.csv')
fold = 2
nr_ref_samples = 512
nr_proto = 16

dir_analysis_out = "/gpfs/work4/0/prjs1086/dimafx/analysis"


# Insert your own WSI annotations here
# wsi_annotations = ['Tumor', '--', 'Tissue edge: stroma > tumor', 'Tumor', 'Adipose > stroma', '--', 'Stroma > adipose', 'Adipose > stroma', 'Tumor: solid pattern, abundant cytoplasm', 'Tumor >> adipose', 'Stroma > tumor > immune cells', 'Tumor > stroma > immune cells', 'Normal ducts/lobules', 'Stroma > adipose', 'Edge stainings, blood vessels', 'Stroma >  immune cells']

# Name 16 W0-W15 clusters
l_wsi_annotations = [f'W{i}' for i in range(16)]


###############
# Loading pathways names and initializing cluster identities
hallmarks = pd.read_csv(path_rna_gene_sets)
# DEBUG for mm_dataset.py# Remove first column
# hallmarks = hallmarks.iloc[:, 1:]
hallmarks = np.array([' '.join(x[9:].split('_')) for x in hallmarks.columns])
hallmarks_ids = np.array([f'R{i}' for i, item in enumerate (hallmarks)])
hallmarks_names = np.array([f'R{i}: {item}' for i, item in enumerate (hallmarks)])

cluster_ids = np.array([f'W{i}' for i, item in enumerate (l_wsi_annotations)])
cluster_names = np.array([f'W{i}: {item}' for i, item in enumerate (l_wsi_annotations)])

all_feats_names = [f"{name}" for name in cluster_names] + [f"{name}" for name in hallmarks_names]
all_short_feat_names = np.concatenate((hallmarks_ids, cluster_ids))


# Obtain shap values
path_shap_test_modal_pkl = os.path.join(dir_res, f'Fold_{fold}/post_training/shap/modal', f'shap_all_test_{nr_ref_samples}.pkl')
shap_dict = pickle.load(open(path_shap_test_modal_pkl, 'rb')) #(210, 65, 256), samples/features/dim
shap_values = shap_dict['shap values']
all_feats_shap = np.sum(shap_values, axis=2)

# Obtain risks
# cox_distcor. risk = exp(logits), the Cox hazard ratio relative to the baseline hazard.
# censorship = 1 = censored: no disease-specific death was observed during follow-up, 0 event observed.
data_results_path = os.path.join(dir_res, f'Fold_{fold}/post_training','predicted_risk_scores_test.pkl')
scores_results = pickle.load(open(data_results_path, 'rb'))   


#Plot shap plot vs risk score
dir_out_shap_vs_risk = os.path.join(dir_analysis_out, 'unimodal_shap_vs_risk')
os.makedirs(dir_out_shap_vs_risk, exist_ok=True)
path_out_shap_vs_risk_score = os.path.join(dir_out_shap_vs_risk, f'shap_vs_risk_fold_{fold}.png')
shap_dotplot_by_risk(all_feats_shap, all_feats_names, scores_results['Risk scores'], path_out_shap_vs_risk_score, cmap="coolwarm")

# Plot SHAP of single sample
def get_plot_single_sample(sample, shap_dict, all_feats_names,path_out_shap_vs_risk_score, max_display=15):
    """ Plot SHAP values for a single sample."""
    # Get shap values of this sample
    shap_values = shap_dict['shap values']
    print(shap_dict['Samples'])
    sample_index = shap_dict['Samples'].index(sample)
    shap_of_one_sample = np.expand_dims(np.sum(shap_values[sample_index], axis=-1),axis=0)
    # Plot shap
    sample_expl = shap.Explanation(values=shap_of_one_sample, feature_names=all_feats_names)
    shap.plots.bar(sample_expl[0], max_display=max_display)
    if path_out_shap_vs_risk_score is not None:
        plt.savefig(path_out_shap_vs_risk_score, dpi=500, bbox_inches='tight')
    plt.show()
    
# We can also plot the local multimodal feature importance plot of one sample
# Take a test id of the specified fold
id = 'TCGA-3C-AALI-01Z-00-DX2.CF4496E0-AB52-4F3E-BDF5-C34833B91B7C'
path_out_shap_vs_risk_score = os.path.join(dir_out_shap_vs_risk, f'shap_vs_risk_fold_{fold}_sample{id}.png')

# Delete previous plt object
plt.clf()

get_plot_single_sample(id, shap_dict, all_feats_names, path_out_shap_vs_risk_score, max_display=15)