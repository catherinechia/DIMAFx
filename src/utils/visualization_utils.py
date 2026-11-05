
import os
import h5py
import torch
import openslide

import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde, spearmanr, pearsonr, kendalltau

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


from models.PANTHER import PANTHER
from embeddings.embeddings import get_prototypes
from data.WSI_dataset import WSIDataset


def get_dataset(mode, fold, type, wsi_feat_type, id_col=None):
    """ Obtain dataset for visualization. """
    data_source = f'../data/data_files/tcga_{type}/'
    dataset = WSIDataset(data_source, wsi_feat_type, mode, fold, id_col=id_col)
    return dataset

def find_fold(slide_id, type, mode, id_col=None):
    """ Find fold where slide is in the test or train (mode) set. """
    # Slide id column, default is slide_id (TCGA)
    slide_col = id_col if id_col else 'slide_id'
    data_folder = f"../data/data_files/tcga_{type}"
    for i in range(5):
        fold_test_file = os.path.join(data_folder, f'splits/{i}')
        test_pd = pd.read_csv(os.path.join(fold_test_file, f"{mode}_filtered.csv"))
        if slide_id in test_pd[slide_col].values:
            return fold_test_file
    
    raise ValueError("Slide is not in the dataset")

def get_panther_encoder(split_folder, nr_prototypes=16):
    """ Load the PANTHER model for obtaining WSI embeddings. Needed for visualization of WSI prototypes. """
    # Define args
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class Args:
        in_dim = 1024
        n_proto = nr_prototypes
        em_iter = 1
        tau = 0.001
        ot_eps = 0.1
        fix_proto = True

    args = Args()
    
    prototypes = get_prototypes(args.n_proto, args.in_dim, split_folder, f'prototypes/prototypes_{nr_prototypes}_type_faiss_init_3_nr_100000.pkl')

    model = PANTHER(args, prototypes, device).to(device)
    model.eval()
    return model

def get_mixture_plot_figure(mixtures, plot_path=None):
    """Create a barplot for the mixture coefficients Pi_c."""
    labels = [f'W{i}' for i in range(len(mixtures))]

    # Create dataframe for plotting
    mixtures_df = pd.DataFrame(mixtures, index=labels).T

    # Plot
    fig, ax = plt.subplots()
    sns.barplot(mixtures_df, color='navy', ax=ax)

    # Spine visibility
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    # Labels and ticks
    ax.set_ylabel(r'c', fontsize=15)
    max_val = np.max(mixtures)  
    ylim = np.ceil(max_val * 10) / 10.0  
    ax.set_ylim([0, ylim + 0.02])
    yticks = np.arange(0, ylim + 0.02, 0.1)
    ax.set_yticks(yticks)
    plt.xticks(rotation=45)

    # Save plot if path is provided
    if plot_path:
        fig.savefig(plot_path, format='pdf', dpi=300)

    plt.close(fig)
    return fig


def find_patch_size(next_coords, prev_coords, coords_patch):
    """Find the patch size by comparing the coords with the next patch"""
    y_ps = 0
    for y_cord, x_cord in next_coords:
        if y_ps > 0:
            break

        if y_cord > coords_patch[0]:
            y_ps = y_cord-coords_patch[0]

    # If the patch is at the last edge
    if y_ps == 0:
        for y_cord, x_cord in prev_coords[::-1]:
            if y_ps > 0:
                break

            if y_cord < coords_patch[0]:
                y_ps = coords_patch[0] - y_cord
    
    assert y_ps > 0, "patch size is 0. Somthing is going wrong!"

    # a patch is always square
    return y_ps
    
def find_patch(all_ids, all_patch_lens, idx, h5_feats_fpath):
    """Find the coordinates and patch size of the patch you want to visualize."""
    counter = 0
    for i, patch_len in enumerate(all_patch_lens):
        # Find the slide the specific patch belongs to
        if idx < (counter + patch_len):
            slide_id = all_ids[i]
            len_slide = patch_len
            patch_i = idx - counter
            break
        else:
            counter += patch_len

    # Open the slide
    h5_feats_fpath_slide = os.path.join(h5_feats_fpath, f'{slide_id}.h5')
    h5_file = h5py.File(h5_feats_fpath_slide, 'r')

    # Get coords of the specific patch
    coords = h5_file['coords']
    coords_patch = coords[patch_i]

    # Get the patchsize of this patch
    patch_size = find_patch_size(coords[patch_i+1:], coords[:patch_i], coords_patch)
    return slide_id, coords_patch, patch_size


def plot_pt(result_file, h5_feats_fpath, slides_fpath, all_ids, all_patch_lens, indices, grd_shape=5):
    """Visualize the given patches to a specified prototype. """
    grid_shape = (grd_shape, grd_shape) 
    fig, axes = plt.subplots(*grid_shape, figsize=(3.5*grd_shape, 3.8*grd_shape))  # Adjust size as needed
    axes = axes.flatten() 

    # We visualize x patches per prototype
    for i, (idx, ax) in enumerate(zip(indices, axes)):
        slide = None
        try:
            # Find slide and patch information
            slide_id, h5_coord, patch_size = find_patch(all_ids, all_patch_lens, idx, h5_feats_fpath)
            slide_path = os.path.join(slides_fpath, f'{slide_id}.svs')

            # Open slide and extract patch
            slide = openslide.OpenSlide(slide_path)
            patch = slide.read_region(
                (h5_coord[0], h5_coord[1]), 
                level=0, 
                size=(patch_size, patch_size)
            ).convert("RGB")
            
            # Display patch in subplot
            ax.imshow(patch)
            ax.axis("off")
        except Exception as e:
            print(f"Error processing patch {i}: {e}")
        finally:
            if slide is not None:
                slide.close()
        ax.set_title(f"Patch {i}", fontsize=18)
    
    # Save and close the plot
    plt.tight_layout()
    plt.savefig(result_file, format='png', dpi=300, bbox_inches='tight')
    plt.close()


def get_data_pathways(pathways, test_data, hallmarks, rna_data, shap_dict, id_col=None):
    """ Get the mean pathway expression of all samples per pathway together with the predicted risk score (group level). """
    # Slide and patient id columns, default is slide_id and case_id (TCGA)
    slide_col = id_col if id_col else 'slide_id'
    case_col = id_col if id_col else 'case_id'

    shap_values = shap_dict['shap values']
    shap_values_rna = np.sum(shap_values[:, 16:, :], axis=2)

    plot_data = []
    for i, (sample, slide) in enumerate(zip(test_data[case_col].values, test_data[slide_col].values)):
     
        case_id_index = list(shap_dict['Samples']).index(slide)
        assert shap_dict['Samples'][case_id_index] == slide, "Correct case ID not found in results."
        color_value_case = shap_values_rna[case_id_index]

        for pathway in pathways:
            # Get the RNA data for the specific case
            rna_data_case = rna_data[rna_data['Unnamed: 0'] == sample]
    
            pathway_name = hallmarks.columns.tolist()[pathway]
            pathway_name = ' '.join(pathway_name[9:].split('_'))

            genes = hallmarks[hallmarks.columns[pathway]].values

            genes = rna_data_case.columns.intersection(genes)
            all_expression_data = rna_data_case[genes]
            # Calculate the mean expression of the pathway (for this sample)
            mean_expression = np.mean(all_expression_data)


            color_value_case_item = color_value_case[pathway]
            
            plot_data.append({
                "Sample": sample,
                "Slide": slide,
                "Pathway": f" R{pathway}: {pathway_name}",
                "Mean expression": mean_expression,
                "Color value": color_value_case_item
            })

    plot_data_group = pd.DataFrame(plot_data)
    return plot_data_group

def pathway_swarm_plot(plot_data, color_name="SHAP value", x_name="Mean expression", x_label="Mean normalized pathway expression", cmap = "RdBu_r"):
    """ Plot the mean pathway expression of all samples per pathway together with the predicted risk score (group level). """
    
    # Normalize the color scale, indiciated by the predicted risk score, around zero.
    col_val = plot_data["Color value"]
    cbar_name = color_name
    
    max_val = np.max(np.abs(col_val))
    norm = TwoSlopeNorm(vmin=-max_val, vcenter=0, vmax=max_val)
    cmap = plt.get_cmap(cmap)


    # Compute Spearman correlation per pathway
    pathways = plot_data["Pathway"].unique()
    
    results = []
    for pathway in pathways:
        subset = plot_data[plot_data["Pathway"] == pathway]
        x = subset[x_name]
        y = subset["Color value"]
        
        sp_r, sp_p = spearmanr(x, y)
        pe_r, pe_p = pearsonr(x, y)
        kt_r, kt_p = kendalltau(x, y)
        
        results.append({
            "Pathway": pathway,
            "Spearman r": round(sp_r, 3),
            "Spearman p": sp_p,
            "Pearson r": round(pe_r, 3),
            "Pearson p": pe_p,
            "Kendall tau": round(kt_r, 3),
            "Kendall p": kt_p,
        })
    
    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(13,6)) # 13,6

    # Create the swarm plot
    sns.swarmplot(
        data=plot_data,
        x=x_name,
        y="Pathway",
        hue=col_val,
        hue_norm=norm,
        palette=cmap,
        legend=False,
        size=4.8  # default is 5
    )

    # add colorbar manually
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, orientation='vertical', shrink=0.2, aspect=4, pad=0)
    cbar.set_label(cbar_name, fontsize=8)

    # Plot params
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.tick_params(axis="x", bottom=True)
    ax.tick_params(axis="y", left=True) 
    ax.set_ylabel("")
    ax.set_xlabel(x_label)

    plt.show()

    return df


def get_data_ridge(pathway_nr, case, shap_feats, all_pathways, all_rna_data, tr_cases):
    """Get data for ridge plot."""
    rna_data_case = all_rna_data[all_rna_data['Unnamed: 0'] == case]
    rna_data_train = all_rna_data[all_rna_data['Unnamed: 0'].isin(tr_cases)]

    

    #Find pathway name
    pathway_name = all_pathways.columns.tolist()[pathway_nr]
    pathway_name = ' '.join(pathway_name[9:].split('_'))

    # Get the gene expresssion distribution of this pathway and this sample
    genes = all_pathways[all_pathways.columns[pathway_nr]].values
    case_ex_genes = rna_data_case.columns.intersection(genes)
    case_expression_data = rna_data_case[case_ex_genes].values[0]

    # Get the mean gene expresssion distribution of this pathway  (over all train cases)
    all_ex_genes = rna_data_train.columns.intersection(genes)
    all_expression_data = rna_data_train[all_ex_genes].values
    # Calculate the mean expression (over the samples) for the pathway
    mean_expression = np.mean(all_expression_data, axis=0)

    # Number of genes should be the same in this same pathway
    if len(mean_expression) is not len(case_expression_data):
        print(f"Warning: Length mismatch for pathway {pathway_name}. Expected {len(mean_expression)}, got {len(case_expression_data)}.")
        return

    data_pt = {
        'name': f" R{pathway_nr}: {pathway_name}",
        'shap': shap_feats[pathway_nr],
        'genes': np.array(case_ex_genes).flatten(),
        'values': np.array(case_expression_data),
        'mean_values': np.array(mean_expression),
    }
    
    return data_pt


def plot_ridge_pathways(pathway_data, offset=0, ylim=[0, 0.5], xlim=[-7, 12]):
    """ PLot the pathway distribution of a single sample together with the background (train) distribution. """
    fig, ax = plt.subplots(figsize=(8, 6))

    values = np.array(pathway_data['values'])
    mean_values = np.array(pathway_data['mean_values'])

    # KDE
    kde = gaussian_kde(values)
    x_range = np.linspace(values.min(), values.max(), 500)
    y = kde(x_range)

    kde_mean = gaussian_kde(mean_values)
    x_range_mean = np.linspace(mean_values.min(), mean_values.max(), 500)
    y_mean = kde_mean(x_range_mean)

    # Full-width horizontal baseline
    ax.hlines(offset, -7, 12, color='black', lw=1, zorder=0)

    # Mean ridge
    ax.fill_between(x_range_mean, offset, y_mean + offset, color='grey', alpha=0.4)
    ax.plot(x_range_mean, y_mean + offset, color='grey', lw=1)

    # Fill ridge
    ax.fill_between(x_range, offset, y + offset, color='green', alpha=0.4)
    ax.plot(x_range, y + offset, color='green', lw=1)


    # Y-tick
    ax.set_yticks([])
    ax.set_xlabel("Gene expression")

    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlim(xlim[0], xlim[1])

    # Remove spines for cleaner style
    for spine in ['top', 'right', 'left']:
        ax.spines[spine].set_visible(False)

    plt.title(pathway_data['name'])

    plt.tight_layout()

    plt.show()


def find_col_row(cluster_ids, hallmarks_ids, attn_type):
    """ Returns the corresponding row and column names. """
    if attn_type == "self_attn_wsi":
        return cluster_ids, cluster_ids
    if attn_type == "self_attn_rna":
        return hallmarks_ids, hallmarks_ids
    if attn_type == "cross_attn_rna_wsi":
        return cluster_ids, hallmarks_ids
    if attn_type == "cross_attn_wsi_rna":
        return hallmarks_ids, cluster_ids
    
    raise ValueError(" Attention type is not valid...")


def visualize_int_row(data_attn, attn_type, cluster_ids, hallmarks_ids, search_name, color='royalblue', topk=3, figsize=(7,6)):
    """ PLots the most important interactiong generating a given multimodal feature. """
    attn_matrix_mean = np.mean(data_attn[attn_type], axis=0)
    attn_matrix_std = np.std(data_attn[attn_type], axis=0)
    row_names, column_names = find_col_row(cluster_ids, hallmarks_ids, attn_type)
    df_mean = pd.DataFrame(attn_matrix_mean, index=row_names, columns=column_names)
    df_std = pd.DataFrame(attn_matrix_std, index=row_names, columns=column_names)

    row_mean, feat_name = get_row_feat(df_mean, search_name)
    row_std, feat_name = get_row_feat(df_std, search_name)
    
    # Select top 5 by mean
    topk_mean = row_mean.nlargest(topk)
    topk_mean_stds = row_std[topk_mean.index]

    # Select top 5 by std
    topk_std = row_std.nlargest(topk)
    topk_std_means = row_mean[topk_std.index]

    # Reverse order for horizontal bar plots (highest on top)
    topk_mean = topk_mean[::-1]
    topk_mean_stds = topk_mean_stds[topk_mean.index]

    topk_std = topk_std[::-1]
    topk_std_means = topk_std_means[topk_std.index]

    # Create subplots
    fig, axes = plt.subplots(1, 1, figsize=figsize)
    plt.subplots_adjust(left=0.5, right=0.95, top=0.9, bottom=0.15)

    # Plot 1: ordered by mean
    axes.barh(topk_mean.index, topk_mean.values, xerr=topk_mean_stds.values, color=color, capsize=topk)
    axes.set_xlabel("Mean attention score")
    axes.set_title(f"{feat_name}") # \nTop interactions by mean")

    plt.show()

def get_row_feat(df, search_name):
    """ Helper function for the visualize int row function. """
    for idx, (name, value) in enumerate(df.iterrows()):
        if name.startswith(search_name):
            selected_row = value
            return selected_row, name
        

def visualize_interaction(row_val, col_val, color_val, row_name, col_name, color_name, title):
    """ Visualize an interaction between two multimodal features. """
    plt.figure(figsize=(8, 6))
    plt.scatter(col_val, row_val, c=color_val, cmap='viridis', s=30)
    plt.title(title)
    plt.xlabel(col_name, fontsize=10)
    plt.ylabel(row_name, fontsize=10)
    # Let matplotlib autoscale based on data first
    plt.autoscale()

    cbar = plt.colorbar(orientation='vertical', fraction=0.05, aspect=5)

    # Add a horizontal label on top of the vertical colorbar
    cbar.ax.set_title(color_name, pad=10, fontsize=10)  # horizontal title above the colorbar


    plt.show()


