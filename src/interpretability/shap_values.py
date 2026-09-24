
import torch
import os
import shap
import sys
import numpy as np
from torch.utils.data import DataLoader

from embeddings.embeddings import get_mixture_params
from models.DIMAFx import SHAP_DIMAFx
from data.mm_survival_dataset import MMSurvivalDataset
from utils.general_utils import load_pkl, save_pkl

def get_dataset(args, mode, fold):
    # Obtain dataset for SHAP
    dataset = MMSurvivalDataset(args, mode, fold)

    embeddings_name = f"{mode}_{args.fm_type}_embeddings_wsi_proto_{args.n_proto}_em_{args.em_iter}_tau_{args.tau}.pkl"
    embedding_dir = os.path.join(args.proto_splits_dir, f'{fold}')
    
    try:
        embeddings = load_pkl(embedding_dir, embeddings_name)
    except:
        print("Something is wrong; there are no embeddings for the defined WSI's \n Train the model before you compute SHAP..")

    dataset.X, dataset.Y = embeddings['X'], embeddings['y']

    in_dim_old = dataset.X.shape[-1]
    prob, mean, cov = get_mixture_params(dataset.X, args.n_proto)
    dataset.X = torch.cat([torch.Tensor(prob).unsqueeze(dim=-1), torch.Tensor(mean)], dim=-1)
    in_dim = (in_dim_old // args.n_proto) - cov.shape[-1]

    return dataset, in_dim

def prepare_data_shap_pre_attn(data, model, num_w, num_proto_wsi=16):
    """ Function that prepares the data to obtain shap values of the unimodal embeddings before fusion. """
    dataloader = DataLoader(data, batch_size=model.batch_size, shuffle=False, num_workers=num_w)
    preproc_dataset = model.prep_data_pre_attn(dataloader)
    feature_names = [f"wsi_pt_{i}" for i in range(num_proto_wsi)] + [f"rna_pt_{i}" for i in range(50)]
    samples = []
    for i in range(data.__len__()):
        case_id = data.data_df.loc[i]['slide_id']
        samples.append(case_id)

    return preproc_dataset, feature_names, samples

def prepare_data_shap_post_attn(data, model, num_w, num_proto_wsi=16):
    """ Function that prepares the data to obtain shap values of ebeddings after fusion. """
    dataloader = DataLoader(data, batch_size=model.batch_size, shuffle=False, num_workers=num_w)
    preproc_dataset = model.prep_data_post_attn(dataloader)
    feature_names = [f"rna_specific_{i}" for i in range(50)] + [f"wsi_rna_{i}" for i in range(50)] + [f"rna_wsi_{i}" for i in range(num_proto_wsi)] + [f"wsi_specific_{i}" for i in range(num_proto_wsi)]
    samples = []
    for i in range(data.__len__()):
        case_id = data.data_df.loc[i]['slide_id']
        samples.append(case_id)

    return preproc_dataset, feature_names, samples

def prepare_data_shap_post_attn_av(data, model, num_w):
    """ Function that prepares the data to obtain shap values of ebeddings after fusion. """
    dataloader = DataLoader(data, batch_size=model.batch_size, shuffle=False, num_workers=num_w)
    preproc_dataset = model.prep_data_post_attn_av(dataloader)
    feature_names = ["wsi_rna_zhg", "rna_wsi_zgh", "rna_specific_zgg", "wsi_specific_zhh"]
    samples = []
    for i in range(data.__len__()):
        case_id = data.data_df.loc[i]['slide_id']
        samples.append(case_id)

    return preproc_dataset, feature_names, samples

def survival_shap(args, fold, post_attn='modal', background_seed=None):
    """ Obtain shap values for trained survival prediction model. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir_fold =  os.path.join(args.result_dir, f"Fold_{fold}/")
    pretrained_model_path = os.path.join(results_dir_fold, "model_checkpoint.pth")
    resuls_dir_shap = os.path.join(results_dir_fold, f'post_training/shap/{post_attn}')
    os.makedirs(resuls_dir_shap, exist_ok=True)

    # Create unimodal representations
    train_data, wsi_dim  = get_dataset(args, 'train', fold)
    test_data, wsi_dim  = get_dataset(args, 'test', fold)

    # Obtain wrapper for SHAP 
    model = SHAP_DIMAFx(rna_dims=train_data.pathway_sizes,
                       histo_dim=wsi_dim,
                       bs=args.shap_bs,
                       device=device,
                       post_attn=post_attn,
                       single_out_dim=256,
                       aggr_post_embed=args.aggr_post_embed,
                       wsi_representation_type=args.wsi_repr,
                       num_proto_wsi=args.n_proto)
    
    model.to(device)
    model.from_pretrained(pretrained_model_path)

    # Obtain input data for the shap_module
    if post_attn == 'modal':  
        train_data, feature_names, samples_train = prepare_data_shap_pre_attn(train_data, model, args.num_workers, num_proto_wsi=args.n_proto)
        test_data, feature_names_test, samples_test = prepare_data_shap_pre_attn(test_data, model, args.num_workers, num_proto_wsi=args.n_proto)
        assert feature_names_test == feature_names
    elif post_attn == 'post_attn':  
        train_data, feature_names, samples_train = prepare_data_shap_post_attn(train_data, model, args.num_workers, num_proto_wsi=args.n_proto)
        test_data, feature_names_test, samples_test = prepare_data_shap_post_attn(test_data, model, args.num_workers, num_proto_wsi=args.n_proto)
        assert feature_names_test == feature_names
    elif post_attn == 'post_attn_av':  
        train_data, feature_names, samples_train = prepare_data_shap_post_attn_av(train_data, model, args.num_workers)
        test_data, feature_names_test, samples_test = prepare_data_shap_post_attn_av(test_data, model, args.num_workers)
        assert feature_names_test == feature_names
    else:
        sys.exit("SHAP mode is not implemented, abborting....")

    if background_seed is not None:
        mask = shap.sample(train_data.to(device), args.shap_refdist_n, random_state=background_seed)
    else:
        mask = shap.sample(train_data.to(device), args.shap_refdist_n)

    test_data = test_data.to(device)
    
    # Compute values with the given explainer
    if args.explainer == 'shap':
        print("Computing SHAP values....")
        explainer = shap.DeepExplainer(model, mask)
        shap_values = explainer.shap_values(test_data, check_additivity=False)
    elif args.explainer == 'eg':
        print("Computing Expected Gradients values....")
        explainer = shap.GradientExplainer(model, mask)
        shap_values = explainer.shap_values(test_data)
    else:
        sys.exit("Unspecified explainer! Abborting..")
    
    shap_values_sq = np.squeeze(shap_values) # [B, feats, 2049]

    print("Save and Visualize shap values.") 

    # Save shap values
    print("Saving shap values..")
    if background_seed is not None:
        name = f'{args.explainer}_all_test_seed_{background_seed}'
    else:
        name = f'{args.explainer}_all_test_{args.shap_refdist_n}'

    save_pkl(resuls_dir_shap, f"{name}.pkl", {'shap values': shap_values_sq, 'Feature names': feature_names, "Samples": samples_test})
