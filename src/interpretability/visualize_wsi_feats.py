import os
import sys
import math
import torch
import argparse

import numpy as np
from tqdm import tqdm


sys.path.append('../')
from embeddings.embeddings import get_mixture_params
from utils.visualization_utils import get_panther_encoder, get_mixture_plot_figure, get_dataset, plot_pt
from utils.general_utils import set_seed


def find_grid_shape(k):
    max_patches = 150
    top_k = min(k, max_patches)

    # Find the largest integer x such that x * x <= top_k
    grid_size = math.floor(math.sqrt(top_k))
    return grid_size

def find_top_k(slide_best_patches, threshold):
    # Find top k patches that have a ll > threshold

    for i, (slide_idx, total_patch_idx, ll) in enumerate(slide_best_patches):

        if ll < threshold:
            print(f"LL is lower than the threshold at number {i}, with a ll of {ll}.")
            return i
    
    # ll is never lower than threshold
    print("Lowest ll is: ", min(slide_best_patches, key=lambda x: x[2]))
    return i

def visualize_pt_per_sample(type, fold, mode, split_folder, results_dir, wsi_feats, th, nr_prototypes, id_col=None):
    """Visualize the closest patches to each prototype for a specific fold."""
    slides_fpath = f'../data/data_files/tcga_{type}/wsi/images'
    h5_feats_fpath = f"../data/data_files/tcga_{type}/wsi/{wsi_feats}/feats_h5"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = os.path.join(results_dir, mode)
    os.makedirs(results_dir, exist_ok=True)

    all_qq, all_patch_lens, all_ids, all_case_ids, all_mus = [], [], [], [], []

    # Get PANTHER model and the wsi's to obtain the embeddings
    panther_encoder = get_panther_encoder(split_folder=split_folder, nr_prototypes=nr_prototypes)
    dataset = get_dataset(mode, fold, type, wsi_feats, id_col=id_col)
   
    # Loop over the dataset
    with torch.inference_mode():
        for idx in tqdm(range(len(dataset))):
            batch = dataset.__getitem__(idx)
            data, slide_id = batch['img'].unsqueeze(dim=0), batch['slide_id']
            data = data.to(device)

            with torch.no_grad():
                # Obtain slide embeddings (GMM parameters)
                out, qqs = panther_encoder.representation(data).values()
                # Obtain the posterior probabilities of each patch given each prototype
                qq = qqs[0,:,:,0].cpu().numpy()
                mus, pis, sigmas = get_mixture_params(out.detach().cpu(), p=nr_prototypes)
                # We obtain the mixture probabilities
                mus = mus[0].detach().cpu().numpy()

            # Save distribution importances of each slide
            all_mus.append(mus)
            
            all_qq.append(qq)
            all_ids.append(slide_id)
            all_case_ids.append(batch['case_id'])
            all_patch_lens.append(qq.shape[0])

    
    all_qq = np.vstack(all_qq)
    all_ids = np.array(all_ids)
    all_patch_lens = np.array(all_patch_lens)

    all_mus = np.array(all_mus)
    all_mus_mean = np.mean(all_mus, axis=0)

    # Visualize overall prototype assignment

    fig_path = os.path.join(results_dir, f'pt_assignment.pdf')
    get_mixture_plot_figure(all_mus_mean, plot_path=fig_path)

    # Visualize prototypes
    for i in range(nr_prototypes):
        print(f"Prototype {i}")
        slide_best_patches = []  # Track best patch per slide for this prototype

        # Group patches by slide and find the best patch for each slide
        start_idx = 0
        all_persons = []
        for slide_idx, num_patches in enumerate(all_patch_lens):
            person_id = all_case_ids[slide_idx]
            if person_id in all_persons:
                end_idx = start_idx + num_patches
                start_idx = end_idx
                continue

            all_persons.append(person_id)
            end_idx = start_idx + num_patches
            slide_qq = all_qq[start_idx:end_idx, i]
            best_patch_idx = np.argmax(slide_qq)
            best_patch_likelihood = slide_qq[best_patch_idx]
            slide_best_patches.append((all_ids[slide_idx], start_idx + best_patch_idx, best_patch_likelihood))
            start_idx = end_idx

        # Sort all slides by their best patch likelihood for this prototype
        slide_best_patches.sort(key=lambda x: x[2], reverse=True)

        # find how many unique patches belong to this prototype
        top_k = find_top_k(slide_best_patches, th)

        # Find grid shape based on top_k. Max k (nr of patches) = 150
        grid_shape = find_grid_shape(top_k)
        top_k = grid_shape * grid_shape
        if top_k < 3:
            print(f"No patches represent prototype {i}, we continue to the next pt")
            continue

        # Get the patches that represent this prototype
        top_patches = slide_best_patches[:top_k]
        grid_shape = find_grid_shape(top_k)

        # # Extract indices and likelihoods for visualization
        top_slides = [entry[0] for entry in top_patches]
        top_indices = [entry[1] for entry in top_patches]
        top_likelihoods = [entry[2] for entry in top_patches]

        # # Visualize those patches
        result_img_pt_name = os.path.join(results_dir, f'W_{i}.png')
        plot_pt(
            result_img_pt_name, h5_feats_fpath, slides_fpath, 
            all_ids, all_patch_lens, top_indices, grd_shape=grid_shape
        )


def main(args):
    set_seed(args.seed)
    split_folder = f"../data/data_files/tcga_{args.data_type}/splits/{args.fold}"
    result_dir = f'wsi_representations_vis/tcga_{args.data_type}/pt_{args.nr_prototypes}/{args.fold}'
    os.makedirs(result_dir, exist_ok=True)

    # Visualize prototypes with one patch per sample for the train set
    visualize_pt_per_sample(args.data_type, args.fold, 'train', split_folder, result_dir, args.wsi_feats, args.threshold, args.nr_prototypes, id_col=args.id_col)

    # Visualize prototypes with one patch per sample for the test set
    visualize_pt_per_sample(args.data_type, args.fold, 'test', split_folder, result_dir, args.wsi_feats, args.threshold, args.nr_prototypes, id_col=args.id_col)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--data_type', type=str, default='brca')
    parser.add_argument('--nr_prototypes', type=int, default=16)
    parser.add_argument('--fold', type=int, default=2)
    parser.add_argument('--wsi_feats', type=str, default='extracted_res0_5_patch256_uni', help='manually specify the wsi feat types')
    parser.add_argument('--threshold', type=float, default=0.9)
    parser.add_argument('--id_col', type=str, default=None, help='Slide id column in the splits files, matching the names of the WSI feature files. Default: slide_id (TCGA). For dsquame: study_number')
    args = parser.parse_args()
    
    main(args)


    
