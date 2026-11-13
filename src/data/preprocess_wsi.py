"""
This is the main file to create the WSI prototypes.

Prototypes are either clustered from the train patches of each fold (--mode),
or taken as-is from a tessapath patch dictionary (--patch_dict_dir).
"""
import argparse
import json
import os
import re
import numpy as np
import pandas as pd
from torch import seed
from torch.utils.data import DataLoader

from WSI_dataset import WSIClusterDataset
from utils.general_utils import set_seed, save_pkl
from embeddings.prototype import cluster

# import torch.multiprocessing as mp
# mp.set_sharing_strategy('file_system')

def create_dataloader(args, split_dir):
    """ Create the dataloader for the clustering """
    # Get the train samples and wsi features path
    split_file = os.path.join(split_dir, f"train_filtered.csv")
    wsi_path = args.wsi_dir #os.path.join(args.data_source, args.wsi_dir)
    coord_dir = args.coord_dir
    bool_strip_slide_ext = args.bool_strip_slide_ext
    bool_return_coords = args.bool_return_coords
    batch_size = args.batch_size
    #
    # Slide id column, default is slide_id (TCGA)
    id_col = args.id_col if args.id_col else 'slide_id'
    #
    # Get the dataset and dataloader
    dataset = WSIClusterDataset(wsi_path, split_file, slide_col=id_col, coord_dir=coord_dir, bool_strip_slide_ext=bool_strip_slide_ext, bool_return_coords=bool_return_coords)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=args.n_workers)
    return dataloader

def save_prototypes(args, pt_weights, prototype_dir):
    """ Save the generated prototypes in a file """
    name = f"prototypes_{args.n_proto}_type_{args.mode}_init_{args.n_init}_nr_{args.n_proto_patches}.pkl"
    output_dir = os.path.join(prototype_dir)
    os.makedirs(output_dir, exist_ok=True)
    save_pkl(output_dir, name, {'prototypes': pt_weights})
    return


def _canon_model(name):
    """ Normalise a foundation model name, e.g. 'conch-v1.5' and 'conchv15' -> 'conchv15' """
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def load_patch_dictionary(dict_dir, fm_type, in_dim):
    """ Load and check a tessapath patch dictionary (see tessapath.dictionary.patch_dictionary) """
    with open(os.path.join(dict_dir, "manifest.json")) as f:
        manifest = json.load(f)
    assert manifest.get('kind') == 'patch_dictionary', f"{dict_dir} is not a tessapath patch dictionary (kind={manifest.get('kind')})"
    #
    # Component table, ordered by row in prototypes.npy
    concepts = pd.read_csv(os.path.join(dict_dir, "concepts.csv")).sort_values('component_index').reset_index(drop=True)
    prototypes = np.load(os.path.join(dict_dir, "prototypes.npy")).astype(np.float32)
    variances_file = os.path.join(dict_dir, "variances.npy")
    variances = np.load(variances_file).astype(np.float32) if os.path.exists(variances_file) else None
    #
    n_proto, feature_dim = prototypes.shape
    assert len(concepts) == n_proto, f"{len(concepts)} concepts but {n_proto} prototypes in {dict_dir}"
    assert feature_dim == in_dim, f"Dictionary feature dim {feature_dim} != in_dim {in_dim} ({dict_dir})"
    assert variances is None or variances.shape == prototypes.shape, f"variances {variances.shape} do not match prototypes {prototypes.shape}"
    model = manifest.get('model')
    assert model is None or _canon_model(model) == _canon_model(fm_type), \
        f"Dictionary was extracted with model '{model}' but fm_type is '{fm_type}'"
    #
    # Mixture weights from the number of member patches per component (as tessapath's gmm_init)
    counts = concepts['n_patches'].to_numpy(dtype=np.float64)
    weights = counts / counts.sum() if counts.sum() > 0 else np.full(n_proto, 1 / n_proto)
    #
    print(f"Loaded patch dictionary {dict_dir}: {n_proto} components x {feature_dim} dims "
          f"(model={model}, token={manifest.get('token')}, target_mpp={manifest.get('target_mpp')})")
    return {'prototypes': prototypes,
            'variances': variances,
            'weights': weights.astype(np.float32),
            'concepts': concepts['concept'].astype(str).tolist(),
            'component_ids': concepts['component_id'].tolist(),
            'source': {'type': 'patch_dictionary',
                       'path': os.path.abspath(dict_dir),
                       **{k: manifest.get(k) for k in ('model', 'token', 'target_mpp', 'patch_size', 'tessapath_version', 'created')}}}


def save_dictionary_prototypes(args, dictionary, prototype_dir):
    """ Save the patch dictionary prototypes (+ variances, weights, concepts) in a file """
    dict_name = os.path.basename(os.path.normpath(args.patch_dict_dir))
    name = f"prototypes_{len(dictionary['concepts'])}_type_dictionary_{dict_name}.pkl"
    save_pkl(prototype_dir, name, dictionary)
    return name


def main(args):
    """ Generate prototypes for the different folds """
    set_seed(args.seed)
    #
    # A patch dictionary gives fixed prototypes, identical for every fold
    dictionary = load_patch_dictionary(args.patch_dict_dir, args.fm_type, args.in_dim) if args.patch_dict_dir else None
    if dictionary is not None and args.n_proto != len(dictionary['concepts']):
        print(f"Ignoring --n_proto {args.n_proto}: the dictionary has {len(dictionary['concepts'])} components")
    #
    for i in range(args.folds):
        print("Generating prototypes for fold ", i)
        prototype_dir = os.path.join(args.data_source, "prototypes", args.fm_type, f'splits/{i}')
        if dictionary is not None:
            name = save_dictionary_prototypes(args, dictionary, prototype_dir)
            print(f"Saved the dictionary prototypes for fold {i}: {os.path.join(prototype_dir, name)}")
            continue
        split_dir = os.path.join(args.splits_dir, f'splits/{i}')
        dataloader = create_dataloader(args, split_dir)
        pt_weights = cluster(dataloader, 
                            n_proto=args.n_proto,
                            n_iter=args.n_iter,
                            n_init=args.n_init,
                            feature_dim=args.in_dim,
                            mode=args.mode,
                            n_proto_patches=args.n_proto_patches)
        #
        save_prototypes(args, pt_weights, prototype_dir)
        print(f"Saved the prototypes for fold {i}!")
    #
    print("DONE")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create WSI prototypes')

    # General args
    parser.add_argument('--seed', type=int, default=1, help='Random seed for reproducible experiment')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--n_workers', type=int, default=2)

    # Clustering args
    parser.add_argument('--n_proto', type=int, default=16, help='Number of prototypes')
    parser.add_argument('--n_proto_patches', type=int, default=100000,help='Number of patches per prototype to use. Total patches = n_proto * n_proto_patches')
    parser.add_argument('--n_init', type=int, default=3, help='Number of different KMeans initialization (for FAISS)')
    parser.add_argument('--n_iter', type=int, default=50, help='Number of iterations for Kmeans clustering')
    parser.add_argument('--mode', type=str, choices=['kmeans', 'faiss'], default='faiss', help='Clustering mode')
    parser.add_argument('--patch_dict_dir', type=str, default=None, help='Optional tessapath patch dictionary folder (manifest.json, prototypes.npy, concepts.csv, ...). If given, its prototypes are used for every fold instead of clustering; the clustering args are ignored and n_proto = number of dictionary components')

    # Data args
    parser.add_argument('--splits_dir', type=str, default='doc/splits/tcga_brca/', help='The splits directory')
    parser.add_argument('--id_col', type=str, default=None, help='Slide id column in the splits files, matching the names of the WSI feature files. Default: slide_id (TCGA). For dsquame: study_number')
    parser.add_argument('--data_source', type=str, default='data/data_files/tcga_brca/', help='The data source')
    parser.add_argument('--fm_type', type=str, default='uni', choices=['uni', 'mstar', 'conchv15', 'virchow2'], help='foundation model type for WSI patch embeddings')
    parser.add_argument('--wsi_dir', type=str, default='wsi/extracted_res0_5_patch256_uni/feats_h5/', help='The WSI features directory')
    parser.add_argument('--coord_dir', type=str, default=None, help='Optional directory for coordinates of the patches (slide2vec)')
    parser.add_argument('--bool_strip_slide_ext', type=bool, default=False, help='Whether to strip the slide extension from the slide name, "true" if used; drop variable if not used')
    parser.add_argument('--bool_return_coords', type=bool, default=False, help='Whether to return the coordinates of the patches (slide2vec), "true" if used; drop variable if not used')
    parser.add_argument('--in_dim', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the dataloader. Currently only batch_size=1 is supported') #TODO: Implement collate function

    args = parser.parse_args()
    main(args)


