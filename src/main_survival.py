import argparse
import sys
import os 
from torch.utils.data import DataLoader

from data.mm_survival_dataset import MMSurvivalDataset
from utils.general_utils import set_seed, save_json, save_exp_settings
from survival.train import survival_train
from survival.test import survival_test
from interpretability.shap_values import survival_shap


def create_dataloader(args, fold, mode="train", type='dl'):
    """ Obtain the dataset and dataloader. """
    dataset = MMSurvivalDataset(args, mode, fold)
    #
    print(f"Dataset for fold {fold} is constructed and checked!")
    print(f'Split: {fold}, n: {len(dataset)}')
    #
    if type == 'dl':
        shuffle_mode = mode == "train"
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle_mode, num_workers=args.num_workers)
        return dataloader
    elif type == 'data_info':
        all_censorships, all_event_times = dataset.get_all_labels()
        return {'censorship': all_censorships, 'time':all_event_times}
    else: 
        return dataset

def k_fold_shap(args):
    """ Obtain shap values for k-folds. """
    for i in range(args.folds):
        print(f"Fold {i}")
        # Pre attention SHAP (modality experiment of 2 representations)
        survival_shap(args, i, 'modal')
        # Post attention SHAP (shared/specific experiment of 4 representations)
        survival_shap(args, i, 'post_attn')
        # Post attention and aggregatopm SHAP (shared/specific experiment of 4 representations)
        survival_shap(args, i, 'post_attn_av')

def k_fold_shap_stable(args):
    """ Obtain shap values for k-folds. """
    for seed in [1, 2, 3, 4, 5]:
        for i in range(args.folds):
            print(f"Fold {i}")
            # Pre attention SHAP (modality experiment of 2 representations)
            survival_shap(args, i, 'modal', background_seed=seed)
            # Post attention SHAP (shared/specific experiment of 4 representations)
            survival_shap(args, i, 'post_attn', background_seed=seed)
            # Post attention and aggregatopm SHAP (shared/specific experiment of 4 representations)
            survival_shap(args, i, 'post_attn_av', background_seed=seed)

def k_fold_test(args):
    """ K-fold cross-validation - Test only. """
    final_res = {}
    for i in range(args.folds):
        print("\n \n Testing fold ", i)
        # Get test dataloader & train data info
        train_data_info = create_dataloader(args, fold=i, mode="train", type='data_info')
        test_dl = create_dataloader(args, fold=i, mode='test')
        #
        # Get test results
        results = survival_test(args, test_dl, survival_info_train=train_data_info, fold=i)
        final_res[f'Fold{i}'] = results
    #
    save_json(args.result_dir, 'Results.json', final_res)

def k_fold_train(args):
    """ K-fold cross-validation - Train only. """
    save_exp_settings(args)
    for i in range(args.folds):
        # Get train dataloader
        train_dl = create_dataloader(args, fold=i, mode="train")
        #
        # Train
        survival_train(args, i, train_dl)

def k_fold_train_test(args):
    """ K-fold cross-validation - Train and Test. """
    final_res = {}
    save_exp_settings(args)
    for i in range(args.folds):
        # Get train and test dataloaders
        train_dl = create_dataloader(args, fold=i, mode="train")
        test_dl = create_dataloader(args, fold=i, mode="test")
        #
        # Train and test
        results = survival_train(args, i, train_dl, test_dl)
        final_res[f'Fold{i}'] = results
    #
    save_json(args.result_dir, 'Results.json', final_res)


def main(args):
    """ K-fold cross-validation for Survival Prediction """
    set_seed(args.seed)
    #
    # Result and log dirs
    args.result_dir = os.path.join(args.result_dir, args.exp_code)
    args.log_dir = os.path.join(args.result_dir, 'logs', args.exp_code)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    #
    # If not using NLL loss, set n_label_bins to 0
    if args.loss_fn != 'nll':
        args.n_label_bins = 0
    #
    # Run the specified mode
    if args.mode == "train_test":
        k_fold_train_test(args)
    elif args.mode == "train":
        k_fold_train(args)
    elif args.mode == "test":
        k_fold_test(args)
    elif args.mode == "shap":
        k_fold_shap(args)
    elif args.mode == "shap_stable":
        k_fold_shap_stable(args)
    else:
        sys.exit("Unspecified mode! Abborting..")
    #
    print("\n\nFINISHED!\n\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Survival Prediction with DIMAFx')

    # General args 
    parser.add_argument('--seed', type=int, default=1, help='random seed for reproducible experiment')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)


    # Learning args
    parser.add_argument('--max_epochs', type=int, default=30, help='maximum number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--wd', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--lr_scheduler', type=str, choices=['cosine', 'linear', 'constant'], default='cosine')
    parser.add_argument('--warmup_steps', type=int, default=-1, help='warmup iterations')
    parser.add_argument('--warmup_epochs', type=int, default=1, help='warmup epochs')

    # Model args
    parser.add_argument('--aggr_post_embed', type=str, default='weighted_mean', choices=['mean', 'weighted_mean'])
    parser.add_argument('--wsi_repr', type=str, default='importance', choices=['normal', 'importance'])

    # PANTHER args
    parser.add_argument('--ot_eps', default=0.1, type=float, help='Strength for entropic constraint regularization for OT')
    parser.add_argument('--em_iter', type=int, default=1)
    parser.add_argument('--tau', type=float, default=0.001)
    parser.add_argument('--fix_proto', type=bool, default=True)
    parser.add_argument('--n_proto', type=int, default=16)
    parser.add_argument('--proto_splits_dir', type=str, default='data/prototypes/', help='path to the prototype splits directory')
    parser.add_argument('--proto_file', type=str, default="prototypes/prototypes_16_type_faiss_init_3_nr_100000.pkl", help="Path to prototypes")

    # Loss args
    parser.add_argument('--w_dis', type=float, default=7.0)
    parser.add_argument('--w_surv', type=float, default=1.0)
    parser.add_argument('--n_label_bins', type=int, default=4, help='number of bins for event time discretization')
    parser.add_argument('--loss_fn', type=str, default='cox_distcor', choices=['nll', 'cox', 'cox_orthogonal', 'cox_distcor', 'cox_hsic', 'nll_orthogonal', 'nll_distcor' 'nll_hsic'], help='which loss function to use')
    parser.add_argument('--nll_alpha', type=float, default=0.5, help='Balance between censored / uncensored loss (in NLL)')


    # Experiment args / label args
    #parser.add_argument('--task', type=str, default='dss_survival_brca')
    parser.add_argument('--exp_code', type=str, default='test', help='experiment code for saving results')
    parser.add_argument('--target_col', type=str, default='dss_survival_days')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--mode', type=str, default='train_test', choices=['test', 'train_test', 'train', 'shap', 'shap_stable']) 


    # dataset args
    # SPLITS
    parser.add_argument('--splits_dir', type=str, default='doc/splits/tcga_brca/', help='path to the splits directory')
    # CD
    parser.add_argument('--data_filter_type', type=str, default='filtered', help='manually specify the data filter type (filtered / unfiltered) or with over/undersampling')
    # WSI
    parser.add_argument('--fm_type', type=str, default='uni', choices=['uni', 'mstar', 'conchv15', 'virchow2'], help='foundation model type for WSI patch embeddings')
    parser.add_argument('--data_source', type=str, default='data/data_files/tcga_brca/', help='wsi source')
    parser.add_argument('--feat_wsi_dir', type=str, help='path to the WSI feature directory', default='extracted_res0_5_patch256_uni')

    # RNA
    parser.add_argument('--omics_type', type=str, default='rna_data', help='for naming the output files')
    parser.add_argument('--omics_dir', type=str, help='dir to the preprocessed omics data')
    parser.add_argument('--feat_omics_path', type=str, default='data/rna/BRCA_proc/rna_data.csv', help='path to the preprocessed omics data')
    parser.add_argument('--signatures_omics_path', type=str, default='data/rna/hallmarks_signatures.csv', help='path to the hallmark signatures')

    # logging args
    parser.add_argument('--result_dir', default='results',help='results directory')
    parser.add_argument('--return_attn', action='store_true', default=False)

    # SHAP args
    parser.add_argument('--shap_refdist_n', type=int, default=512) # BRCA: 512, BLCA: 256, LUAD: 320, KIRC: 192
    parser.add_argument('--shap_bs', type=int, default=64)
    parser.add_argument('--explainer', type=str, default='shap', choices=['eg', 'shap'])


    args = parser.parse_args()
    main(args)
