"""
Adapted from https://github.com/mahmoodlab/MMP/blob/main/src/utils/proto_utils.py
"""
import torch
import time
import numpy as np
from tqdm import tqdm

import faiss
from sklearn.cluster import KMeans

def kmeans_clustering(patches, n_proto, n_iter):
    """ Find cluster centers of the data (Prototypes) using KMeans (cpu). """
    print("\nUsing Kmeans for clustering...")
    print(f"\n\tNum of clusters {n_proto}, num of iter {n_iter}")
    #
    # Cluster the data
    kmeans = KMeans(n_clusters=n_proto, max_iter=n_iter)
    kmeans.fit(patches.cpu())
    #
    # Get prototypes
    weight = kmeans.cluster_centers_[np.newaxis, ...]
    return weight

def faiss_clustering(patches, n_proto, n_iter, n_init, n_proto_patches):
    """ Find cluster centers of the data (Prototypes) using Faiss (gpu). """
    numOfGPUs = torch.cuda.device_count()
    print(f"\nUsing Faiss Kmeans for clustering with {numOfGPUs} GPUs...")
    print(f"\tNum of clusters {n_proto}, num of iter {n_iter}")
    #
    # Cluster the data
    kmeans = faiss.Kmeans(patches.shape[1], 
                            n_proto, 
                            niter=n_iter, 
                            nredo=n_init,
                            verbose=True, 
                            max_points_per_centroid=n_proto_patches,
                            gpu=numOfGPUs)
    #
    kmeans.train(patches.numpy())
    #
    # Get prototypes
    weight = kmeans.centroids[np.newaxis, ...]
    return weight

def get_patches(dataloader, n_proto, n_proto_patches, feature_dim):
    """ Obtain the patch features used for clustering. """
    #
    n_total = n_proto * n_proto_patches
    n_patches_per_batch = (n_total + len(dataloader) - 1) // len(dataloader)
    print(f"Sampling maximum of {n_proto * n_proto_patches} patches: {n_patches_per_batch} each from {len(dataloader)}.")
    patches = torch.Tensor(n_total, feature_dim)
    #
    # Go over all slides
    n_patches = 0
    for batch in tqdm(dataloader):
        if n_patches >= n_total:
            continue
        #
        # Select the patches randomly from each slide
        with torch.no_grad():
            indices = torch.randperm(batch.shape[1])[:n_patches_per_batch]
            selected_patches = batch[0][indices]
        #
        # Check if we reached n_total 
        size = selected_patches.size(0)
        if n_patches + size > n_total:
            size = n_total - n_patches
            selected_patches = selected_patches[:size]
        #
        # Store selected patches
        patches[n_patches: n_patches + size] = selected_patches
        n_patches += size
    #
    return patches, n_patches

def cluster(dataloader, n_proto, n_iter, n_init, feature_dim, mode, n_proto_patches):
    """ This function clusters the data and finds the prototypes. """
    #
    # Sample the patches to cluster from each train slide
    patches, n_patches = get_patches(dataloader, n_proto, n_proto_patches, feature_dim)
    print(f"\nTotal of {n_patches} patches aggregated")
    #
    # Find cluster centers according to clustering mode
    s = time.time()
    if mode == 'kmeans':
        weight = kmeans_clustering(patches, n_proto, n_iter)
    elif mode == 'faiss':
        assert torch.cuda.is_available(), f"FAISS requires access to GPU. Please enable use_cuda"
        weight = faiss_clustering(patches, n_proto, n_iter, n_init, n_proto_patches)
    #
    else:
        raise NotImplementedError(f"Clustering not implemented for {mode}!")
    #
    e = time.time()
    print(f"\nClustering took {e-s} seconds!")
    #
    # Return prototypes / cluster centers
    return weight