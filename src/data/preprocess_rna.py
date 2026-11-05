import pandas as pd
import numpy as np
import argparse
import os

# Max value expected for log2-transformed expression. Files with larger values are treated as linear (e.g. TPM).
LOG2_MAX = 30
# Ensembl gene id without version suffix, e.g. ENSG00000000003
ENSEMBL_GENE_ID = r'^ENS[A-Z]*G\d{11}$'


def check_files(args):
    """ Get the patients in the train and test splits of all folds. """
    #
    splits_dir = args.splits_dir
    #
    # From splits_dir, get number of folds
    l_folds = [f for f in os.listdir(splits_dir) if os.path.isdir(os.path.join(splits_dir, f))]
    l_folds.sort()
    #
    cases_from_folds = []
    #
    for i in l_folds:
        df_test = pd.read_csv(os.path.join(splits_dir, i, 'test_filtered.csv'), delimiter=',')
        df_train = pd.read_csv(os.path.join(splits_dir, i, 'train_filtered.csv'), delimiter=',')
        #
        # Patient id column, default is the first column (TCGA: case_id)
        id_col = args.id_col if args.id_col else df_test.columns[0]
        cases_from_folds.append(set(df_test[id_col]) | set(df_train[id_col]))
    #
    # TCGA folds all contain the same patients. dsquame folds can differ slightly in their train sets, so take the union.
    if args.data_source == 'tcga':
        assert all(s == cases_from_folds[0] for s in cases_from_folds), "Not all sets are the same"
    #
    # Return a list of patients to keep
    return sorted(set.union(*cases_from_folds))


def read_rna(path):
    """ Read RNA-seq data with genes as rows and samples as columns. The delimiter (tab or comma) is taken from the header. """
    #
    with open(path) as f:
        header = f.readline()
    delimiter = '\t' if '\t' in header else ','
    #
    return pd.read_csv(path, delimiter=delimiter, index_col=0)


def preprocess_tcga(df_raw, cases_to_keep):
    """ Preprocess the raw TCGA data. Rows of the result are samples, columns are genes. """
    #
    df_raw_T = df_raw.transpose().sort_index(axis=0)
    #
    # Drop the samples from the normal tissue
    df_filtered = df_raw_T[~df_raw_T.index.str.endswith('-11')]
    #
    # Keep only the samples from the primary tissue
    df_filtered.index = df_filtered.index.str.replace(r'-01', '', regex=True)
    #
    # Keep only the samples that have clinical, rna and wsi data
    df_filtered_complete = df_filtered[df_filtered.index.isin(cases_to_keep)]
    #
    return df_filtered_complete


def clean_gene_ids(df):
    """ Drop the PAR_Y copies and strip the version suffix of the Ensembl gene ids (ENSG00000000003.15 -> ENSG00000000003). """
    #
    df = df[~df.index.str.endswith('_PAR_Y')]
    df.index = df.index.str.replace(r'\.\d+$', '', regex=True)
    assert not df.index.duplicated().any(), "Duplicate gene ids after stripping the version suffix"
    #
    return df


def to_log2(df, name):
    """ Apply log2(x + 1) if the data is linear (e.g. TPM), so all files are on the same log2 scale. """
    #
    max_value = df.values.max()
    if max_value > LOG2_MAX:
        print(f'{name}: max value {max_value:.1f}, treating as linear and applying log2(x + 1)')
        df = np.log2(df + 1)
    else:
        print(f'{name}: max value {max_value:.1f}, treating as log2 already')
    #
    return df


def has_ensembl_ids(df):
    """ Check if the gene columns are Ensembl gene ids instead of gene symbols. """
    return df.columns.str.match(ENSEMBL_GENE_ID).all()


def map_gene_symbols(df, ensembl_release):
    """ Rename Ensembl gene ids to gene symbols with pyensembl, as used in the hallmark signatures. Duplicate symbols are averaged. """
    # Imported here, so pyensembl is only needed for data with Ensembl gene ids
    from pyensembl import EnsemblRelease
    #
    # Download and index the annotation of this release if not cached yet
    genome = EnsemblRelease(ensembl_release)
    print(genome)
    genome.download()
    genome.index()
    #
    # Genes missing from the release or without a name (novel genes) are dropped
    known_ids = set(genome.gene_ids())
    gene_map = {g: genome.gene_name_of_gene_id(g) for g in df.columns if g in known_ids}
    gene_map = {g: name for g, name in gene_map.items() if name}
    #
    mapped = df.columns.isin(list(gene_map))
    print(f'Gene symbols (Ensembl {ensembl_release}): {mapped.sum()} of {len(mapped)} genes mapped, dropping {(~mapped).sum()} unmapped genes')
    df = df.loc[:, mapped]
    df.columns = [gene_map[g] for g in df.columns]
    #
    n_dup = df.columns.duplicated().sum()
    if n_dup > 0:
        print(f'Gene symbols: averaging {n_dup} genes with a duplicate symbol')
        df = df.T.groupby(level=0, sort=False).mean().T
    #
    return df


def preprocess_dsquame(df_train_raw, df_test_raw, cases_to_keep):
    """ Preprocess and combine the dsquame train and test data. Rows of the result are samples, columns are genes. """
    #
    df_train = to_log2(clean_gene_ids(df_train_raw), 'train')
    df_test = to_log2(clean_gene_ids(df_test_raw), 'test')
    #
    # The train data is already filtered on genes. Keep these genes, as far as they are present in the test data.
    genes = df_train.index[df_train.index.isin(df_test.index)]
    missing_genes = df_train.index[~df_train.index.isin(df_test.index)]
    print(f'Genes: keeping {len(genes)}, dropping {len(missing_genes)} train genes missing from test: {list(missing_genes)}')
    #
    overlap = df_train.columns.intersection(df_test.columns)
    assert len(overlap) == 0, f"Samples present in both train and test data: {list(overlap)}"
    #
    df_all = pd.concat([df_train.loc[genes], df_test.loc[genes]], axis=1).transpose().sort_index(axis=0)
    #
    # Keep only the samples in the splits
    missing_cases = sorted(set(cases_to_keep) - set(df_all.index))
    print(f'Samples: {len(cases_to_keep) - len(missing_cases)} of {len(cases_to_keep)} split samples have RNA. Missing: {missing_cases}')
    df_filtered_complete = df_all[df_all.index.isin(cases_to_keep)]
    #
    return df_filtered_complete


def main(args):
    # Get the patients to keep
    cases_to_keep = check_files(args)
    #
    # Load and preprocess data
    if args.data_source == 'tcga':
        df_filtered_complete = preprocess_tcga(read_rna(args.data_path), cases_to_keep)
    else:
        assert args.test_data_path, "--test_data_path is required for dsquame"
        df_filtered_complete = preprocess_dsquame(read_rna(args.data_path), read_rna(args.test_data_path), cases_to_keep)
    #
    # The hallmark signatures use gene symbols, so convert Ensembl gene ids if present
    if has_ensembl_ids(df_filtered_complete):
        df_filtered_complete = map_gene_symbols(df_filtered_complete, args.ensembl_release)
    #
    # Save the preprocessed data. First column holds the sample ids.
    df_filtered_complete.index.name = 'Unnamed: 0'
    df_filtered_complete.columns.name = None
    os.makedirs(args.data_dir, exist_ok=True)
    path_out = os.path.join(args.data_dir, f'{args.name}.csv')
    df_filtered_complete.to_csv(path_out)
    print(f'Saved {df_filtered_complete.shape[0]} samples x {df_filtered_complete.shape[1]} genes to {path_out}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Preprocess RNA-seq data of TCGA or dsquame')
    #
    parser.add_argument('--data_source', default='tcga', choices=['tcga', 'dsquame'], help='Data source of the RNA-seq data')
    parser.add_argument('--splits_dir', type=str, default='doc/splits/tcga_brca/', help='path to the splits directory')
    parser.add_argument('--id_col', type=str, default=None, help='Patient id column in the splits files, matching the sample names in the RNA-seq data. Default: first column (TCGA). For dsquame: study_number')
    parser.add_argument('--data_path', help='Path to normalized RNA-seq data (dsquame: train data). Rows are genes, columns are samples. First column is the gene names')
    parser.add_argument('--test_data_path', default=None, help='dsquame only: path to the test RNA-seq data, same layout as --data_path')
    parser.add_argument('--ensembl_release', type=int, default=104, help='Ensembl release used by pyensembl to convert Ensembl gene ids to gene symbols, if the data has Ensembl gene ids')
    parser.add_argument('--data_dir', type=str, default='data/rna/BRCA_proc', help='Path to the output directory')
    parser.add_argument('--name', default='test', help='Resulting file name')
    #
    args = parser.parse_args()
    main(args)

# #Debug dsquame
# args.data_source = 'dsquame'
# args.splits_dir = '_data/dsquame__dev_risk__val_random'
# args.id_col = 'study_number'
# args.data_path = '_data/rna_dis_risk_studynum.tsv'
# args.test_data_path = '_data/rna_val_random_studynum.tsv'
# args.data_dir = '_data/rna_proc'
# args.name = 'rna_data'
