import argparse
import sys
import os
from torch.utils.data import DataLoader

from data.mm_classification_dataset import MMClassificationDataset
from utils.general_utils import set_seed, save_json, save_exp_settings
from classification.train import classification_train


def create_dataloader(args, fold, mode="train", type='dl'):
    """ Obtain the dataset and dataloader. """
    dataset = MMClassificationDataset(args, mode, fold)
    #
    print(f"Dataset for fold {fold} is constructed and checked!")
    print(f'Split: {fold}, n: {len(dataset)}')
    #
    if type == 'dl':
        shuffle_mode = mode == "train"
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle_mode, num_workers=args.num_workers)
        return dataloader
    elif type == 'data_info':
        return {'labels': dataset.get_all_labels()}
    else:
        return dataset

def k_fold_train(args):
    """ K-fold cross-validation - Train only. """
    save_exp_settings(args)
    for i in range(args.folds):
        # Get train dataloader
        train_dl = create_dataloader(args, fold=i, mode="train")
        #
        # Train
        classification_train(args, i, train_dl)

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
        results = classification_train(args, i, train_dl, test_dl)
        final_res[f'Fold{i}'] = results
    #
    save_json(args.result_dir, 'Results.json', final_res)


def main(args):
    """ K-fold cross-validation for Classification """
    set_seed(args.seed)
    #
    # Result and log dirs
    args.result_dir = os.path.join(args.result_dir, args.exp_code)
    args.log_dir = os.path.join(args.result_dir, 'logs', args.exp_code)
    os.makedirs(args.result_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    #
    # Run the specified mode
    if args.mode == "train_test":
        k_fold_train_test(args)
    elif args.mode == "train":
        k_fold_train(args)
    else:
        sys.exit("Mode not (yet) supported for classification! Abborting..")
    #
    print("\n\nFINISHED!\n\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classification with DIMAFx')

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
    parser.add_argument('--loss_fn', type=str, default='ce', choices=['ce', 'svm'], help='which loss function to use. "svm" requires the topk package (SmoothTop1SVM)')

    # Experiment args / label args
    parser.add_argument('--exp_code', type=str, default='test', help='experiment code for saving results')
    parser.add_argument('--label_col', type=str, default='label', help='name of the categorical label column in the splits csv (0-indexed class integers)')
    parser.add_argument('--n_classes', type=int, default=2, help='number of classes for classification')
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--mode', type=str, default='train_test', choices=['train_test', 'train'])


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
    parser.add_argument('--result_dir', default='results', help='results directory')
    parser.add_argument('--return_attn', action='store_true', default=False)

    args = parser.parse_args()
    main(args)
