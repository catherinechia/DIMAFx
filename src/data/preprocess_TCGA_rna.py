import pandas as pd 
import argparse
import os

def check_files(args): 
    """ Check which patients have both RNA, WSI, and clinical data in all folds. """
    #
    splits_dir = args.splits_dir
    #
    # From splits_dir, get number of folds
    l_folds = [f for f in os.listdir(splits_dir) if os.path.isdir(os.path.join(splits_dir, f))]
    l_folds.sort()
    #
    cases_from_folds  = []
    #
    for i in l_folds:
        #df2 = pd.read_csv(f'data_files/tcga_{data_type}/splits/{i}/test_filtered.csv', delimiter=',')
        #df3 = pd.read_csv(f'data_files/tcga_{data_type}/splits/{i}/train_filtered.csv', delimiter=',')
        df2 = pd.read_csv(f'{splits_dir}/{i}/test_filtered.csv', delimiter=',')
        df3 = pd.read_csv(f'{splits_dir}/{i}/train_filtered.csv', delimiter=',')
        #
        cases_to_keep = list(df2[df2.columns[0]].values.flatten())
        cases_splits3 = list(df3[df3.columns[0]].values.flatten())
        cases_to_keep.extend(cases_splits3)
        cases_from_folds.append(set(cases_to_keep))
    #
    assert all(s == cases_from_folds[0] for s in cases_from_folds), "Not all sets are the same"
    #
    # Return a list of patients to keep
    return cases_to_keep


def preprocess_data(df_raw, cases_to_keep):
    """ Preprocess the raw data and save it in the correct format. """
    #
    df_raw_T = df_raw.set_index('sample')
    df_raw_T = df_raw_T.transpose()
    df_raw_T = df_raw_T.sort_index(axis=0)
    df_raw_T.index.name = None
    df_raw_T = df_raw_T.reset_index()
    df_raw_T.columns.name = None
    #
    df_raw_T = df_raw_T.rename(columns={'index': 'sample'})
    #
    # Drop the samples from the normal tissue
    df_filtered = df_raw_T[~df_raw_T['sample'].str.endswith('-11')].reset_index(drop=True)
    #
    # Keep only the samples from the primary tissue
    df_filtered['sample'] = df_filtered['sample'].str.replace(r'-01', '', regex=True)
    #
    # Keep only the samples that have clinical, rna and wsi data
    df_filtered_complete = df_filtered[df_filtered['sample'].isin(cases_to_keep)].reset_index(drop=True)
    df_filtered_complete = df_filtered_complete.rename(columns={'sample': 'Unnamed: 0'})
    #
    return df_filtered_complete

def main(args):
    # Perform data preprocessing steps here
    #df_raw = pd.read_csv(f'data_files/tcga_{args.data}/rna/HiSeqV2_PANCAN_{args.data.upper()}', delimiter='\t')
    df_raw = pd.read_csv(args.data_path, delimiter='\t')
    #
    # Check if the data is loaded correctly
    cases_to_keep = check_files(args)
    #
    # Preprocess data
    df_filtered_complete = preprocess_data(df_raw, cases_to_keep)
    #
    # Save the preprocessed data
    #df_filtered_complete.to_csv(f'data_files/tcga_{args.data}/rna/{args.name}.csv')
    path_out = os.path.join(args.data_dir, f'{args.name}.csv')
    df_filtered_complete.to_csv(path_out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess RNA-seq data')
    #
    parser.add_argument('--splits_dir', type=str, default='doc/splits/tcga_brca/', help='path to the splits directory')
    parser.add_argument('--data_path', help='Path to normalized RNA-seq data. Rows are genes, columns are samples. First column is the gene names (colname: sample)')
    parser.add_argument('--data_dir', type=str, default='data/rna/BRCA_proc', help='Path to the directory containing the RNA-seq data')
    parser.add_argument('--data', default='brca', choices=['blca', 'brca', 'kirc', 'luad'], help='Data cohort to use')
    parser.add_argument('--name', default='test', help='Resulting file name')
    #
    args = parser.parse_args()
    main(args)

# #Debug
# args.splits_dir = '/gpfs/work4/0/prjs1086/dimafx/doc/splits/tcga_brca/splits'
# args.data_path = '/gpfs/work4/0/prjs1086/dimafx/data/rna/BRCA_proc/HiSeqV2_PANCAN_BRCA'
# args.data_dir = '/gpfs/work4/0/prjs1086/dimafx/data/rna/BRCA_proc'
# args.data = 'brca'
# args.name = 'rna_data'

