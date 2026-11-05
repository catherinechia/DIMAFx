import os
import torch
import pandas as pd
import h5py
import numpy as np

from torch.utils.data import Dataset
from utils.data_utils import compute_discretization, pd_diff, overlap_col_df, _series_intersection
from utils.general_utils import save_pkl, load_pkl
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

l_wsi_extensions = ('.svs', '.ndpi', '.tif', '.tiff', '.mrxs', '.scn', '.vms', '.vmu', '.svslide', '.bif', '.qptiff', '.dcm')

class MMDatasetBase(Dataset):
    """
    Shared base class for Multi Modal Dataset using RNA-seq data and WSI.
    Sets up and checks the clinical, WSI and RNA data of a split and builds the RNA/WSI part of each sample. Task-specific datasets such as MMSurvivalDataset and MMClassificationDataset only define how labels are set up, checked and returned.
    """
    def __init__(self,
                 args,
                 mode,
                 fold):
        """
        Args:
            - args          : All the arguments given by the user (Obj)
            - mode          : Specifies if we are in 'train' or 'test' mode (str)
            - fold          : Specifies the fold number (int)
        """

        self.mode = mode
        self.fold = fold

        # Data args
        self.data_source = args.data_source
        self.split_dir = os.path.join(args.splits_dir, f'{self.fold}/')
        self.duplicates = False
        if 'oversampled' in args.data_filter_type:
            self.duplicates = True

        # Clinical data
        self.data_file = self.get_data_file_name(args)
        self.clinical_data_path = args.clinical_data_path

        # Id args
        # Slide and patient id columns, default is slide_id and case_id (TCGA). For dsquame: study_number for both
        id_col = getattr(args, 'id_col', None)
        self.slide_col = id_col if id_col else 'slide_id'
        self.case_col = id_col if id_col else 'case_id'

        # WSI args
        self.feat_wsi_dir = args.feat_wsi_dir
        self.X = None
        self.y = None

        # RNA args
        self.omics_dir = args.omics_dir
        self.omics_type = args.omics_type
        self.feat_omics_path = args.feat_omics_path
        self.signatures_omics_path = args.signatures_omics_path
        self.scaler = None

        # Label args
        self.init_label_args(args)

        # Setup and check clinical data
        self.init_df()
        # Setup and check WSI data
        self.init_df_wsi()
        # Setup and check RNA data
        self.init_df_rna()
        # Setup task-specific labels, now that data_df contains the final samples of this split
        self.init_labels()
        
    ############################################################
    # placeholders. if a new subclass is written, these functions need to be implemented
    def init_label_args(self, args):
        """ Store the label args of the task. """
        raise NotImplementedError

    def check_labels(self):
        """ Check the label values in the clinical data of this split. """
        raise NotImplementedError

    def init_labels(self):
        """ Optional label setup after all data is loaded (e.g. discrete time labels). """
        pass

    def get_label_dict(self, idx):
        """ Get the label entries of a sample, as tensors keyed by name. """
        raise NotImplementedError

    def get_all_labels(self):
        """ Get the labels of all samples. """
        raise NotImplementedError

    # ----- Shared setup -----

    def get_data_file_name(self, args):
        """ Name of the split file of this mode, depending on the data filter type. """
        if args.data_filter_type == 'none':
            return f'{self.mode}.csv'
        if self.mode == 'test' and 'censor' in args.data_filter_type:
            return f'{self.mode}_filtered.csv'
        return f'{self.mode}_{args.data_filter_type}.csv'

    def check_unique_per_slide(self, col, name):
        """ Check that each slide has only one unique value in col. """
        num_unique = self.data_df.groupby(self.slide_col)[col].unique().apply(len)
        assert (num_unique == 1).all(), f'Each {self.slide_col} must have only one unique {name} value.'

    def check_data_file(self):
        """ Check the clinical dataset. """
        self.check_labels()

        if not self.duplicates:
            # Should be no duplicates in splits file
            assert len(list(self.data_df[self.slide_col].astype(str).unique())) == len(list(self.data_df[self.slide_col].astype(str))), 'There are duplicates in the given splits file.'

    def check_wsi_files(self, feats_wsi_df):
        """ Check that the wsi files are complete and that there are no duplicates. """
        # Should be no missing files
        missing_feats_in_split = pd_diff(self.data_df[self.slide_col], feats_wsi_df[self.slide_col])
        assert len(missing_feats_in_split) == 0, f'Missing Features in Split:\n{missing_feats_in_split}.'

        # Should be no duplicates
        duplicates = feats_wsi_df[self.slide_col].duplicated()
        assert duplicates.sum() == 0, f'Features duplicated in data source(s):{feats_wsi_df[duplicates].to_string()}.'

    def check_rna_files(self):
        """ Check that the rna files have no duplicates. """
        # For RNA we have a sample per person. For WSI we can have multiple slides per person.
        assert not self.df_rna[self.case_col].duplicated().any(), "There are duplicates in the rna data."

    def init_df(self):
        """ Set up clinical data of this split. """
        # Split file only determines which slide ids belong to this fold/mode
        split_file = os.path.join(self.split_dir, self.data_file)
        split_df = pd.read_csv(split_file)
        split_df[self.slide_col] = split_df[self.slide_col].astype(str)

        # Clinical data (e.g. target_col, censorship_col) is looked up for those slide ids from the general clinical data file
        clinical_df = pd.read_csv(self.clinical_data_path)
        clinical_df[self.slide_col] = clinical_df[self.slide_col].astype(str)

        self.data_df = split_df[[self.slide_col]].merge(clinical_df, how='inner', on=self.slide_col, validate='m:1')
        self.check_data_file()

    def init_df_wsi(self):
        """ Set up WSI data of this split. """
        self.data_df[self.slide_col] = self.data_df[self.slide_col].astype(str)

        # Store feature paths. If extensions are present in the slide ids, we can strip them off to match the slide ids in the splits file. Use l_wsi_extensions
        feats_wsi_df = pd.DataFrame([(e.path, os.path.splitext(e.name)[0]) for e in os.scandir(self.feat_wsi_dir)], columns=['fpath', self.slide_col]).reset_index(drop=True)
        feats_wsi_df[self.slide_col] = feats_wsi_df[self.slide_col].apply(lambda x: os.path.splitext(x)[0] if os.path.splitext(x)[1].lower() in l_wsi_extensions else x)
        self.check_wsi_files(feats_wsi_df)

        if self.duplicates:
            # All slide ids to feature paths should have a one-to-one mapping. Raises ValueError if not.
            self.data_df = self.data_df.merge(feats_wsi_df, how='inner', on=self.slide_col)
        else:
            self.data_df = self.data_df.merge(feats_wsi_df, how='inner', on=self.slide_col, validate='1:1')
        self.data_df = self.data_df[list(self.data_df.columns[-1:]) + list(self.data_df.columns[:-1])]

    def init_df_rna(self):
        """ Set up RNA data of this split. """
        if os.path.isfile(self.feat_omics_path):
            self.df_rna = pd.read_csv(self.feat_omics_path, engine='python')
            # Rename first column with any name to the patient id column
            self.df_rna = self.df_rna.rename(columns={self.df_rna.columns[0]: self.case_col})
        else:
            raise FileNotFoundError(f"{self.feat_omics_path} not found!")

        # Check for duplicates
        self.check_rna_files()

        # Keep only the patients that are in this split and have WSI data
        case_ids_overlap = overlap_col_df(self.df_rna, self.data_df, self.case_col)
        sample_list = sorted(case_ids_overlap)

        self.data_df = self.data_df[self.data_df[self.case_col].isin(sample_list)].reset_index(drop=True)
        self.df_rna = self.df_rna[self.df_rna[self.case_col].isin(sample_list)].reset_index(drop=True)
        self.df_rna = self.df_rna.set_index(self.case_col)

        # Set up hallmark pathways
        self.setup_rna_pathways()

        # Initialize and apply scaler for RNA data
        self.setup_scaler()
        self.apply_scaler()

    def setup_scaler(self):
        """ Fit or load scaler for RNA data. """
        out_dir = os.path.join(self.omics_dir, 'scalers', 'splits', f'{self.fold}')
        if self.mode == 'train':
            # Fit the scaler on the training data
            self.scaler = StandardScaler().fit(self.df_rna)
            save_pkl(out_dir, f'{self.omics_type}_scaler_fold_{self.fold}.pkl', self.scaler)
        else:
            try:
                # Read the scaler from the pickle file
                print("Load scaler")
                self.scaler = load_pkl(out_dir, f'{self.omics_type}_scaler_fold_{self.fold}.pkl')
            except FileNotFoundError:
                print(f"Cannot access the scaler from training. Make sure '{out_dir}scaler_fold_{self.fold}.pkl' exists.")
                self.scaler = None

    def apply_scaler(self):
        """ Apply fitted scaler to RNA data. """
        assert not self.scaler == None, "Cannot scale the data, scaler is not defined!"

        cols = self.df_rna.columns
        case_list = self.df_rna.index.values
        self.df_rna = pd.DataFrame(self.scaler.transform(self.df_rna), columns=cols)
        self.df_rna.insert(0, self.case_col, case_list)
        self.df_rna = self.df_rna.set_index(self.case_col)

    def setup_rna_pathways(self):
        """ Load Hallmarks biological pathways, which serve as the prototypes. """
        signatures = pd.read_csv(self.signatures_omics_path, index_col=0)
        self.rna_names = []
        self.pathway_names = []
        self.pathway_sizes = []

        # For each pathway
        for col in signatures.columns:
            omic = signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.df_rna.columns))

            # Store all genes involved
            self.rna_names.append(omic)

            # Store the name of the pathway
            self.pathway_names.append(col)

            # Store the number of genes in the pathway
            self.pathway_sizes.append(len(omic))

    # ----- Getters -----

    def get_split_dir(self):
        return self.split_dir

    def get_split_folder(self):
        """ Get the dir of this specific fold. """
        return self.split_dir

    def __len__(self):
        """ Get the number of samples. """
        return len(self.data_df)

    def find_corresponding_idx(self, idx):
        # For when we are oversampling
        slide_id = self.data_df.loc[idx][self.slide_col]
        # Find the corresponding index in the image data
        idx_img = self.data_df[self.data_df[self.slide_col] == slide_id].index
        for idx_i in idx_img:
            if idx_i < len(self.X):
                assert self.data_df.loc[idx_i][self.slide_col] == slide_id, f"Index {idx_i} does not correspond to the same slide id {slide_id} as index {idx}."
                return idx_i

        raise ValueError(f"No corresponding index found for slide id {slide_id} in the image data.")

    def get_rna(self, case_id):
        """ Get the expression values of all genes per pathway for a case. """
        pathway_summary = []
        for i in range(len(self.pathway_names)):
            pathway_summary.append(torch.Tensor(self.df_rna.loc[case_id, self.rna_names[i]]))
        return pathway_summary

    def get_img(self, idx):
        """ Get the slide embedding if already created, else the patch features of the slide. """
        if self.X is not None:
            # We already created the unsupervised slide embedding (slide summary)
            # For when we are oversampling
            if idx >= len(self.X):
                return self.X[self.find_corresponding_idx(idx)]
            return self.X[idx]

        return load_wsi_features(self.data_df.loc[idx]['fpath'])

    def __getitem__(self, idx):
        # Obtain labels
        out = self.get_label_dict(idx)

        # Obtain case and slide ids
        case_id = self.data_df.loc[idx][self.case_col]
        slide_id = self.data_df.loc[idx][self.slide_col]
        out['case_id'] = case_id
        out['slide_id'] = slide_id

        # Obtain RNA data
        out['rna'] = self.get_rna(case_id)

        # Obtain WSI data
        out['img'] = self.get_img(idx)

        return out


def load_wsi_features(feat_path):
    """ Load the patch features of a slide as a (numOffeats, feat_dim) tensor. """
    if feat_path.endswith('.h5'):  # CLAM
        with h5py.File(feat_path, 'r') as f:
            features = f['features'][:]
    elif feat_path.endswith('.pt'):  # slide2vec
        features = torch.load(feat_path, map_location='cpu', weights_only=True).float()
    else:
        raise ValueError(f'Unsupported feature file format: {feat_path}')

    if len(features.shape) > 2:
        assert features.shape[0] == 1, f'{features.shape} is not compatible! It has to be (1, numOffeats, feat_dim) or (numOffeats, feat_dim)'
        features = np.squeeze(features, axis=0)

    # If feature is not a tensor, convert it to a tensor
    if not isinstance(features, torch.Tensor):
        features = torch.from_numpy(features)
    return features


class MMSurvivalDataset(MMDatasetBase):
    """
        Multi Modal Dataset using RNA-seq data and WSI
        Used for multi modal survival prediction
    """
    def __init__(self,
                 args,
                 mode,
                 fold,
                 label_bins=None):
        """
        Args:
            - args          : All the arguments given by the user (Obj)
            - mode          : Specifies if we are in 'train' or 'test' mode (str)
            - fold          : Specifies the fold number (int)
            - label_bins    : Time bins for the discrete time labels, e.g. from the train split (array).
                              If None and n_label_bins > 0, they are fitted on this split.
        """
        self.label_bins = label_bins
        super().__init__(args, mode, fold)

    def init_label_args(self, args):
        self.survival_time_col = args.target_col
        self.target_col = args.target_col
        self.censorship_col = args.target_col.split('_')[0] + '_censorship'
        self.n_label_bins = args.n_label_bins

    def check_labels(self):
        """ Check the clinical survival data. """
        # Check that each slide has only one survival value
        self.check_unique_per_slide(self.survival_time_col, 'survival')

        # check that all survival values are positive
        assert (self.data_df[self.survival_time_col] >= 0).all(), 'Survival values must be positive.'

        # check that all censorship values are binary integers
        assert self.data_df[self.censorship_col].isin([0, 1]).all(), 'Censorship values must be binary integers.'

    def init_labels(self):
        """ Setup discrete time labels iff NLL. """
        if self.n_label_bins > 0:
            self.init_disc_labels()

    def init_disc_labels(self):
        """ Compute discrete time labels from continuous survival times. """
        disc_labels, label_bins = compute_discretization(df=self.data_df,
                                                         survival_time_col=self.survival_time_col,
                                                         censorship_col=self.censorship_col,
                                                         n_label_bins=self.n_label_bins,
                                                         label_bins=self.label_bins)
        self.data_df = self.data_df.join(disc_labels)
        self.label_bins = label_bins
        self.target_col = disc_labels.name

    def get_labels(self, idx):
        """ Get the survival time (days), censorship and target label (either continuous or discrete time labels). """
        labels = self.data_df.loc[idx][[self.survival_time_col, self.censorship_col, self.target_col]]
        return list(labels)

    def get_label_dict(self, idx):
        survival_time, censorship, label = self.get_labels(idx)
        return {'survival_time': torch.Tensor([survival_time]),
                'censorship': torch.Tensor([censorship]),
                'label': torch.Tensor([label])}

    def get_all_labels(self):
        """ Get the survival time (days) and censorship for all samples. """
        cs_list = list(self.data_df.loc[:][self.censorship_col])
        st_list = list(self.data_df.loc[:][self.survival_time_col])
        return np.array(cs_list), np.array(st_list)

    def get_label_bins(self):
        """ Get the time bins for the discrete time labels."""
        return self.label_bins


class MMClassificationDataset(MMDatasetBase):
    """
        Multi Modal Dataset using RNA-seq data and WSI
        Used for multi modal classification
    """
    def init_label_args(self, args):
        self.target_col = args.target_col
        self.n_classes = args.n_classes

    def check_labels(self):
        """ Check the clinical classification data. """
        # Check that each slide has only one label value
        self.check_unique_per_slide(self.target_col, 'label')

        # Check that all label values are valid class indices
        assert self.data_df[self.target_col].isin(range(self.n_classes)).all(), \
            f'Label values must be integers in [0, {self.n_classes - 1}].'

    def get_label(self, idx):
        """ Get the class label for a sample. """
        return int(self.data_df.loc[idx][self.target_col])

    def get_label_dict(self, idx):
        return {'label': torch.tensor(self.get_label(idx), dtype=torch.long)}

    def get_all_labels(self):
        """ Get the class labels for all samples. """
        return np.array(list(self.data_df.loc[:][self.target_col]))
