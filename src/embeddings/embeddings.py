import os
import torch
from utils.general_utils import load_pkl, save_pkl
from models.PANTHER import PANTHER

# Create a dictionary for foundation model and their respective embedding dimensions
dict_fm = {
    'uni': 1024,
    'mstar': 1024,
    'conchv15':768,
    'virchow2': 1280, #CLS or patch tokens only
}



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



def create_slide_embeddings(args, in_dim_fm, prototypes, dataloader):
    """
        Generate the slide embeddings in an unsupervised way using PANTHER (https://github.com/mahmoodlab/PANTHER)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PANTHER(args, in_dim_fm, prototypes, device).to(device)

    # Generate the slide embeddings
    X, y = model.predict(dataloader)
    dataloader.dataset.X, dataloader.dataset.y = X, y
    embeddings = {'X': X, 'y': y}
    return embeddings

def get_mixture_params(X, p):
    """
        Get the mixture probabilities/mean/covariance
    """
    d = (X.shape[1] - p) // (2 * p)
    prob = X[:, : p]
    mean = X[:, p: p * (1 + d)].reshape(-1, p, d)
    cov = X[:, p * (1 + d):].reshape(-1, p, d)
    return prob, mean, cov


def reshape_embeddings(dataloader, n_proto):
    """
        Reshape the embeddings to be concatenated features of the prob and mean per prototype
    """
    new_in_dim = dataloader.dataset.X.shape[-1]
    prob, mean, cov = get_mixture_params(dataloader.dataset.X, n_proto)

    # Embeddings consist of importance and mean per mixture
    dataloader.dataset.X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean)], dim=-1)
    in_dim = (new_in_dim // n_proto) - cov.shape[-1]

    return dataloader, in_dim


def get_slide_embeddings(args, mode, dataloader):
    """
        Obtain slide embeddings
    """
    in_dim_fm = dict_fm[args.fm_type]
    # Preparing file path for saving embeddings
    print('\nConstructing unsupervised slide embedding...', end=' ')
    emb_slide_filename = f"{mode}_{args.fm_type}_embeddings_wsi_proto_{args.n_proto}_em_{args.em_iter}_tau_{args.tau}.pkl"
    #embedding_dir = os.path.join(dataloader.dataset.split_dir, 'embeddings_DIMAFx')
    split_folder_dir = os.path.join(args.proto_splits_dir, f'{dataloader.dataset.fold}')
    emb_slide_path = os.path.join(split_folder_dir, emb_slide_filename)
    
    if os.path.isfile(emb_slide_path):
        # Load existing embeddings if already created
        embeddings = load_pkl(split_folder_dir, emb_slide_filename)
        print(f'\n\tEmbedding already exists! Loading', end=' ')
    else:
        # Else, create them using PANTHER
        #split_folder = dataloader.dataset.get_split_folder()
        prototypes = get_prototypes(args.n_proto, in_dim_fm, split_folder_dir, args.proto_file)
        embeddings = create_slide_embeddings(args, in_dim_fm, prototypes, dataloader)

        # Save the embeddings for efficiency
        #os.makedirs(embedding_dir, exist_ok=True)
        save_pkl(split_folder_dir, emb_slide_filename, embeddings)
    
    dataloader.dataset.X, dataloader.dataset.Y = embeddings['X'], embeddings['y']
    assert dataloader.dataset.X is not None
    
    dataloader, in_dim = reshape_embeddings(dataloader, args.n_proto)

    return dataloader, in_dim

def prepare_embeddings(args, mode, dataloader):
    """
        Prepare unimodal embeddings for both the wsi and rna-seq data.
    """
    data_info = {}

    # Obtain the wsi embeddings
    dataloader, in_dim_wsi = get_slide_embeddings(args, mode, dataloader)
    data_info['Dim wsi'] = in_dim_wsi
    
    # The pathway embeddings are made on the fly
    omic_dim = dataloader.dataset.df_rna.shape[1] 
    pathway_sizes = dataloader.dataset.pathway_sizes
    
    data_info['Pathway sizes'] = pathway_sizes
    data_info['Dim rna'] = omic_dim

    return dataloader, data_info
