"""
This is the main file to create the WSI prototypes.
"""
import argparse
import os
from torch import seed
from torch.utils.data import DataLoader

from data.WSI_dataset import WSIClusterDataset
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
    # Get the dataset and dataloader
    dataset = WSIClusterDataset(wsi_path, split_file, coord_dir=coord_dir, bool_strip_slide_ext=bool_strip_slide_ext, bool_return_coords=bool_return_coords)
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=args.n_workers)
    return dataloader

def save_prototypes(args, pt_weights, prototype_dir):
    """ Save the generated prototypes in a file """
    name = f"prototypes_{args.n_proto}_type_{args.mode}_init_{args.n_init}_nr_{args.n_proto_patches}.pkl"
    output_dir = os.path.join(prototype_dir)
    os.makedirs(output_dir, exist_ok=True)
    save_pkl(output_dir, name, {'prototypes': pt_weights})
    return


def main(args):
    """ Generate prototypes for the different folds """
    set_seed(args.seed)
    #
    for i in range(args.folds):
        print("Generating prototypes for fold ", i)
        split_dir = os.path.join(args.splits_dir, f'splits/{i}')
        prototype_dir = os.path.join(args.data_source, "prototypes", f'splits/{i}')
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

    # Data args
    parser.add_argument('--data_source', type=str, default='data/data_files/tcga_brca/', help='The data source')
    parser.add_argument('--wsi_dir', type=str, default='wsi/extracted_res0_5_patch256_uni/feats_h5/', help='The WSI features directory')
    parser.add_argument('--coord_dir', type=str, default=None, help='Optional directory for coordinates of the patches (slide2vec)')
    parser.add_argument('--bool_strip_slide_ext', type=bool, default=False, help='Whether to strip the slide extension from the slide name')
    parser.add_argument('--bool_return_coords', type=bool, default=False, help='Whether to return the coordinates of the patches (slide2vec)')
    parser.add_argument('--in_dim', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for the dataloader. Currently only batch_size=1 is supported') #TODO: Implement collate function

    args = parser.parse_args()
    main(args)


