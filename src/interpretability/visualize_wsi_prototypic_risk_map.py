import os
import json
import pickle
import numpy as np, torch, h5py, openslide
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm

from utils.visualization_utils import get_panther_encoder, find_patch_size
from utils.general_utils import load_pkl, save_pkl
from models.PANTHER import PANTHER

def strip_slide_ext(slide_id):
    """ Strip a trailing WSI image extension (e.g. '.svs'), as slide2vec keeps it in its sample ids. """
    root, ext = os.path.splitext(slide_id)
    return root if ext.lower() in l_wsi_extensions else slide_id

def load_patch_feats(path_feat, path_coords=None):
    """
    Load the patch features, level 0 (x, y) coordinates and level 0 patch size of a slide.
        - 'h5' : CLAM/TRIDENT '<slide_id>.h5' with 'features' and 'coords' datasets
        - 'pt' : slide2vec '<sample_id>.pt' (n_tiles, feat_dim) tensor, with '<sample_id>.coordinates.npz' (keys: tile_index, x, y, tissue_fractions)
                 and optional '<sample_id>.coordinates.meta.json' next to it (or at path_coords)
    """
    # Get extension of the feature file from path_feat
    ext_feat = os.path.basename(path_feat).split('.')[-1].lower()
    if ext_feat == 'pt' and path_coords is None:
        path_coords = f"{path_feat[:-len('.pt')]}.coordinates.npz"
    ext_coords = os.path.basename(path_coords).split('.')[-1].lower() if path_coords is not None else None
    if ext_feat == 'h5':
        with h5py.File(path_feat, 'r') as h5:
            feats = torch.Tensor(h5['features'][:])
            coords = h5['coords'][:]   # (x, y) at level 0
        patch_size = find_patch_size(coords[1:], coords[:0], coords[0])
    elif ext_feat == 'pt' and ext_coords == 'npz':
        feats = torch.load(path_feat, map_location='cpu', weights_only=True).float()
        stem = path_feat[:-len('.pt')]
        path_coords = path_coords if path_coords is not None else f'{stem}.coordinates.npz'
        coords_npz = np.load(path_coords)
        coords = np.stack([coords_npz['x'], coords_npz['y']], axis=1)
        assert len(coords) == len(feats), f'{len(feats)} features but {len(coords)} coordinates in {path_coords}.'
        # Patch size at level 0 from the metadata, else the smallest step between tile x coordinates
        meta_path = path_coords.replace('.coordinates.npz', '.coordinates.meta.json')
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
            assert meta['artifact']['coordinate_space'] == 'level0_px', f"Coordinates in {meta_path} are not in level 0 pixels."
            patch_size = meta['tiling']['tile_size_lv0']
        else:
            patch_size = int(np.diff(np.unique(coords[:, 0])).min())
    else:
        raise ValueError(f"feat_format must be 'h5' or 'pt', got '{ext_feat}'")
    if len(feats.shape) > 2:
        feats = feats.squeeze(0)
    return feats, coords, patch_size

def get_prototypes(n_proto, embed_dim, split_folder, proto_file):
    """"
        Load, check and return the specified prototypes
    """
    assert os.path.exists(os.path.join(split_folder, proto_file)), "{} does not exist!".format(os.path.join(split_folder, proto_file))
    if proto_file.endswith('pkl'):
        prototypes = load_pkl(split_folder, proto_file)['prototypes'].squeeze()
    else:
        print("File format for given prototypes is not supported. Abborting...")
    assert (n_proto == prototypes.shape[0]) and (embed_dim == prototypes.shape[1]),\
        "Prototype dimensions do not match! Params: ({}, {}) Suplied: ({}, {})".format(n_proto,
                                                                                        embed_dim,
                                                                                        prototypes.shape[0],
                                                                                        prototypes.shape[1])
    return prototypes

def overlay_prototypes(slide_filename, path_wsi, path_feat, 
                       dir_split, prototype_filename, args_proto,
                       path_shap_test_modal_pkl, nr_proto=16, type='brca', thumb=2000, pts=None, path_out_fig=None, path_coords=None):
    """
    Overlay prototype assignment and prototype SHAP (risk) on the WSI thumbnail. pts limits the risk map to some prototypes.
        - feat_format : 'h5' (CLAM) or 'pt' (slide2vec) patch features (str)
        - path_feat   : Path of the feature file. Default for 'h5' is the CLAM UNI features of the dataset, required for 'pt' (str)
        - path_coords : For 'pt' only, path of the '.coordinates.npz' file. Defaults to the one next to path_feat (str)
        - path_wsi  : Path of the WSI. Defaults to the '.svs' image of the dataset (str)
        - pts : prototype indices to include in the risk map. Default None = all prototypes (list of int)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slide_filename_without_ext = os.path.splitext(slide_filename)[0]
    slide = openslide.OpenSlide(path_wsi)
    feats, coords, patch_size = load_patch_feats(path_feat, path_coords)
    feats = feats.unsqueeze(0).to(device)
    # Posterior of each patch for each prototype
    in_dim = args_proto.in_dim
    n_proto = args_proto.n_proto
    prototypes = get_prototypes(n_proto, in_dim, dir_split, prototype_filename)
    encoder = PANTHER(args_proto, in_dim, prototypes, device).to(device)
    encoder.eval()
    with torch.inference_mode():
        _, qqs = encoder.representation(feats).values()
    qq = qqs[0, :, :, 0].cpu().numpy()   # (n_patches, nr_proto)
    # SHAP of the WSI prototypes of this slide (modal order: wsi_pt_0..nr_proto-1 first)
    shap_dict = pickle.load(open(path_shap_test_modal_pkl, 'rb'))
    shap_pt = shap_dict['shap values'][list(shap_dict['Samples']).index(slide_filename_without_ext)].sum(-1)[:nr_proto]
    keep = np.arange(nr_proto) if pts is None else np.array(pts)
    risk = qq[:, keep] @ shap_pt[keep]
    assign = qq.argmax(1)
        # Rasterise the patch values onto the thumbnail grid
    img = slide.get_thumbnail((thumb, thumb))
    ds = slide.dimensions[0] / img.size[0]
    w = max(1, int(round(patch_size / ds)))
    risk_map = np.full((img.size[1], img.size[0]), np.nan)
    assign_map = np.full_like(risk_map, np.nan)
    for (x, y), r, a in zip(coords, risk, assign):
        x0, y0 = int(x / ds), int(y / ds)
        risk_map[y0:y0 + w, x0:x0 + w] = r
        assign_map[y0:y0 + w, x0:x0 + w] = a
    #Plot
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    for ax in axes:
        ax.imshow(img)
        ax.axis('off')
    # One discrete color per prototype. tab20 comes in dark/light pairs, so take the dark ones first to keep neighbouring prototypes distinct
    if nr_proto <= 20:
        tab20 = plt.get_cmap('tab20').colors
        cmap_a = ListedColormap((tab20[::2] + tab20[1::2])[:nr_proto])
    else:
        cmap_a = plt.get_cmap('turbo', nr_proto)
    norm_a = BoundaryNorm(np.arange(-0.5, nr_proto), nr_proto)
    im_a = axes[1].imshow(assign_map, cmap=cmap_a, norm=norm_a, alpha=0.6, interpolation='nearest')
    lim = np.nanmax(np.abs(risk_map))
    im_r = axes[2].imshow(risk_map, cmap='RdBu_r', norm=TwoSlopeNorm(0, -lim, lim), alpha=0.6, interpolation='nearest')
    axes[0].set_title(slide_filename_without_ext[:12])
    axes[1].set_title('Most likely prototype')
    axes[2].set_title('Prototype SHAP per patch (red = higher risk)')
    fig.colorbar(im_a, ax=axes[1], ticks=range(nr_proto), fraction=0.03)
    fig.colorbar(im_r, ax=axes[2], fraction=0.03)
    plt.tight_layout()
    if path_out_fig is not None:
        plt.savefig(path_out_fig, dpi=500, bbox_inches='tight')
    plt.show()
    slide.close()
