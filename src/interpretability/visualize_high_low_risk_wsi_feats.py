import os
import sys
import math
import torch
import argparse
import pickle

import numpy as np
from tqdm import tqdm
import openslide
import h5py
import matplotlib.pyplot as plt

sys.path.append('../')
from embeddings.embeddings import get_mixture_params
from utils.visualization_utils import get_panther_encoder, find_patch_size
from utils.general_utils import set_seed


def visualize_pt_sample(result_dir, split_folder, slide_id, id, pt, type, shap_val, nr_pt):
    """Visualize prototypes using patches of a single wsi."""
    # input paths
    slide_fpath = f'../data/data_files/tcga_{type}/wsi/images/{slide_id}.svs'
    h5_feats_fpath = f'../data/data_files/tcga_{type}/wsi/extracted_res0_5_patch256_uni/feats_h5/{slide_id}.h5'

    # Get WSI and feats
    wsi = openslide.open_slide(slide_fpath)
    h5 = h5py.File(h5_feats_fpath, 'r')
    feats = torch.Tensor(h5['features'][:]).unsqueeze(0)

    # Get PANTHER model and the wsi's to obtain the embeddings
    panther_encoder = get_panther_encoder(split_folder=split_folder, nr_prototypes=nr_pt)

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    # Get proportions of each mixture component
    with torch.inference_mode():
        out, qqs = panther_encoder.representation(feats).values()
        pis, mus, vars = get_mixture_params(out, p=nr_pt)
        pis = pis[0].detach().cpu().numpy()
        qq = qqs[0,:,:,0].cpu().numpy()

        # Show closest patces for each prototype
        top_indices = np.argsort(qq[:, pt])[-5:][::-1]  # top 5, best first
        coords = h5['coords']
        main_patch_size = 0
        for i, top_index in enumerate(top_indices):
            ax = axes[i]
            if qq[top_index, pt] < 0.00001:
                ax.imshow(np.ones((256, 256, 3)))  # blank white image
            else:
                next_coords = coords[top_index+1:]
                prev_coords = coords[:top_index]
                coords_patch = coords[top_index]
                patch_size = find_patch_size(next_coords, prev_coords, coords_patch)
                
                if main_patch_size > 0 and patch_size != main_patch_size:
                    print("WARNING: patch size is different from the main patch size, something is wrong..")
                    print(f"ID: {id}, PATCHSIZE main: {main_patch_size}, current: {patch_size}")
                else:
                    main_patch_size = patch_size
                patch = wsi.read_region(
                    (coords[top_index][0], coords[top_index][1]),
                    level=0,
                    size=(main_patch_size, main_patch_size)
                ).convert("RGB")   
                ax.imshow(patch)

            ax.axis("off")
            if i == 0:
                if pis[pt] < 0.0005:
                    ax.text(-0.05, 0.5, f"$\\mathbf{{W({pt})}}$, c<0.001", va='center', ha='right', rotation=90, fontsize=9, transform=ax.transAxes)
                else:
                    ax.text(-0.05, 0.5, f"$\\mathbf{{W({pt})}}$, c={pis[pt]:.3f}", va='center', ha='right', rotation=90, fontsize=9, transform=ax.transAxes)
        
    # Save and close the plot
    os.makedirs(result_dir, exist_ok=True)
    result_file = os.path.join(result_dir, f'{id}_shap_{shap_val:.3f}.png')
    plt.tight_layout()
    plt.savefig(result_file, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def find_samples(feat_num, shap_dict, nr_samples):  
    shap_values = shap_dict['shap values']
    all_feats_shap = np.sum(shap_values, axis=2)
    feat_shap = all_feats_shap[:, feat_num]
    samples = shap_dict['Samples']

    zipped = [(i, name, shap) for i, (name, shap) in enumerate(zip(samples, feat_shap))]
    # Sort by shap value, high to low
    zipped_sorted = sorted(zipped, key=lambda x: x[2], reverse=True)

    # Sort by shap value, low to high
    zipped_sorted_neg = sorted(zipped, key=lambda x: x[2], reverse=False)

    chosen_samples = []
    # Print top results
    for i, (id_, name, shap) in enumerate(zipped_sorted):
        chosen_samples.append((id_, name, shap))
        if i+1 == nr_samples:
            break

    # Print top results
    for i, (id_, name, shap) in enumerate(zipped_sorted_neg):
        chosen_samples.append((id_, name, shap))
        if i+1 == nr_samples:
            break

    # Sort by absolute shap value, lowest (as intermediate risk)
    zipped_sorted_abs_neg = sorted(zipped, key=lambda x: abs(x[2]), reverse=False)
    # Print top results
    for i, (id_, name, shap) in enumerate(zipped_sorted_abs_neg):
        chosen_samples.append((id_, name, shap))
        if i+1 == 4:
            break

    return chosen_samples

def find_case_shap(feat_num, shap_dict, cases):  
    shap_values = shap_dict['shap values']
    all_feats_shap = np.sum(shap_values, axis=2)
    feat_shap = all_feats_shap[:, feat_num]
    samples = shap_dict['Samples']

    chosen_samples = []
    for i in cases:
        name = samples[i]
        shap_val = feat_shap[i]
        chosen_samples.append((i, name, shap_val))

    return chosen_samples


def visualize_pt(feat_num, result_dir, split_folder, data_type, nr_pt, nr_samples=5, shap_dict=None, items=None):
    result_dir = os.path.join(result_dir, f'W{feat_num}')

    if items is None:
        chosen_samples = find_samples(feat_num, shap_dict, nr_samples)
    
    else:
        print("Using provided items for visualization..")
        chosen_samples = find_case_shap(feat_num, shap_dict, items)

    for id, slide_id, shap_val in chosen_samples:
        print(f"for {id} slide: {slide_id}, shap value: {shap_val:.3f}")
        visualize_pt_sample(result_dir, split_folder, slide_id, id, feat_num, data_type, shap_val, nr_pt)


    
def main(args):
    split_folder = f"../data/data_files/tcga_{args.data_type}/splits/{args.fold}"
    shap_dir = f"../results/dss_survival_{args.data_type}/{args.exp}/Fold_{args.fold}/post_training/shap/modal/shap_all_test_{args.shap_refdist_n}.pkl"
    result_dir = f'wsi_representations_vis/tcga_{args.data_type}/fold_{args.fold}/risk/{args.exp}/shap_{args.shap_refdist_n}'
    os.makedirs(result_dir, exist_ok=True)

    set_seed(args.seed)

    shap_dict = pickle.load(open(shap_dir, 'rb'))     


    # If items == None, the code will find the samples to visualize based on risk
    for feat in [8, 3, 0, 9, 10]:
        print(f"Visualizing feat {feat}")
        # Visualize prototypes with one patch per sample for the test set
        visualize_pt(feat, result_dir, split_folder, args.data_type, nr_pt=args.nr_pt, nr_samples=args.nr_samples, shap_dict=shap_dict, items=None)
    
    # W8
    visualize_pt(8, result_dir, split_folder, args.data_type, nr_pt=args.nr_pt, shap_dict=shap_dict, items=[76, 52, 168, 132])

    # W0
    visualize_pt(0, result_dir, split_folder, args.data_type, nr_pt=args.nr_pt, shap_dict=shap_dict, items=[37, 179, 38])

    # W7
    visualize_pt(7, result_dir, split_folder, args.data_type, nr_pt=args.nr_pt, shap_dict=shap_dict, items=[15, 13, 58, 195])

    # W4
    visualize_pt(4, result_dir, split_folder, args.data_type, nr_pt=args.nr_pt, shap_dict=shap_dict, items=[15, 13, 58, 195])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_type', type=str, default='brca')
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--nr_pt', type=int, default=16)
    parser.add_argument('--shap_refdist_n', type=int, default=512)
    parser.add_argument('--exp', type=str, default='DIMAFx')
    parser.add_argument('--nr_samples', type=int, default=8)
    parser.add_argument('--wsi_feats', type=str, default='extracted_res0_5_patch256_uni', help='manually specify the wsi feat types')

    args = parser.parse_args()
    
    main(args)


    
