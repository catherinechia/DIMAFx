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

class MMSurvivalDataset(Dataset):
    """
        Multi Modal Dataset using RNA-seq data and WSI 
        Used for multi modal survival prediction
    """
    def __init__(self,
                 args, 
                 mode, 
                 fold,
                 slide_col='slide_id'):
        """
        Args:
            - args          : All the arguments given by the user (Obj)
            - mode          : Specifies if we are in 'train' or 'test' mode (str)
            - fold          : Specifies the fold number (int)
            - slide_col     : Name of the column storing the slide ids (str)
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
        if args.data_filter_type == 'none':
            self.data_file = f'{self.mode}.csv'
        else:
            if mode =='test' and 'censor' in args.data_filter_type:
                self.data_file = f'{self.mode}_filtered.csv'
            else:
                self.data_file = f'{self.mode}_{args.data_filter_type}.csv'

        # WSI args
        self.slide_col = slide_col
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
        self.survival_time_col = args.target_col
        self.target_col = args.target_col
        self.censorship_col = args.target_col.split('_')[0] + '_censorship'
        self.n_label_bins = args.n_label_bins
        self.label_bins = None

        # Setup and check GT discrete labels iff NLL #TODO: Move to after all inits, so that data_df is initialized
        if self.n_label_bins > 0:
            self.init_disc_labels()

        # Setup and check clinical data
        self.init_df()
        # Setup and check WSI data
        self.init_df_wsi()
        # Setup and check RNA data
        self.init_df_rna()

        
    def check_data_file(self):
        """Check the clinical survival dataset. """

        # Check that each case_id has only one survival value
        num_unique_surv_times = self.data_df.groupby(self.slide_col)[self.survival_time_col].unique().apply(len)
        assert (num_unique_surv_times == 1).all(), 'Each case_id must have only one unique survival value.'

        # check that all survival values are positive
        assert (self.data_df[self.survival_time_col] >= 0).all(), 'Survival values must be positive.'

        # check that all censorship values are binary integers
        assert self.data_df[self.censorship_col].isin([0, 1]).all(), 'Censorship values must be binary integers.'

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
        assert not self.df_rna['case_id'].duplicated().any(), "There are duplicates in the rna data."

    def init_df(self):
        """ Set up clinical data of this split. """
        split_file = os.path.join(self.split_dir, self.data_file)
        self.data_df = pd.read_csv(split_file)
        self.check_data_file()

    def init_df_wsi(self):
        """ Set up WSI data of this split. """
        # Obtain directory containing the patch features
        #self.feat_wsi_dir = os.path.join(self.data_source, f"wsi/{self.wsi_feats}/feats_h5")
        self.data_df[self.slide_col] = self.data_df[self.slide_col].astype(str)

        # Store feature paths. If extensions are present in the slide ids, we can strip them off to match the slide ids in the splits file. Use l_wsi_extensions
        feats_wsi_df = pd.DataFrame([(e.path, os.path.splitext(e.name)[0]) for e in os.scandir(self.feat_wsi_dir)], columns=['fpath', self.slide_col]).reset_index(drop=True)
        feats_wsi_df[self.slide_col] = feats_wsi_df[self.slide_col].apply(lambda x: os.path.splitext(x)[0] if os.path.splitext(x)[1].lower() in l_wsi_extensions else x)
        self.check_wsi_files(feats_wsi_df)

        if self.duplicates:
            # All slide ids to feature paths should have a one-to-one mapping. Raises ValueError if not.
            # Add feature paths to data frame
            self.data_df = self.data_df.merge(feats_wsi_df, how='inner', on=self.slide_col)
        else:
            # All slide ids to feature paths should have a one-to-one mapping. Raises ValueError if not.
            # Add feature paths to data frame
            self.data_df = self.data_df.merge(feats_wsi_df, how='inner', on=self.slide_col, validate='1:1')
        self.data_df = self.data_df[list(self.data_df.columns[-1:]) + list(self.data_df.columns[:-1])]
    
    def init_df_rna(self):
        """ Set up RNA data of this split. """
        # Read RNA data
        #self.feat_dir_rna = os.path.join(self.data_source, f"rna/{self.omics_type}.csv")
        if os.path.isfile(self.feat_omics_path):
            self.df_rna = pd.read_csv(self.feat_omics_path, engine='python')
            # Rename first column with any name to 'case_id'
            self.df_rna = self.df_rna.rename(columns={self.df_rna.columns[0]: 'case_id'})
        else:
            raise FileNotFoundError(f"{self.feat_omics_path} not found!")
        
        # Check for duplicates
        self.check_rna_files()

        # Keep only the patients that are in this split and have WSI data
        case_ids_overlap = overlap_col_df(self.df_rna, self.data_df, 'case_id')
        sample_list = sorted(case_ids_overlap)

        self.data_df = self.data_df[self.data_df['case_id'].isin(sample_list)].reset_index(drop=True)
        self.df_rna = self.df_rna[self.df_rna['case_id'].isin(sample_list)].reset_index(drop=True)
        self.df_rna = self.df_rna.set_index('case_id')

        # Set up hallmark pathways
        self.setup_rna_pathways()

        # Initialize and apply scaler for RNA data
        self.setup_scaler()
        self.apply_scaler()

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

    def get_split_dir(self):
        return self.split_dir
    
    def get_all_labels(self):
        """ Get the survival time (days) and censorship for all samples. """
        cs_list = list(self.data_df.loc[:][self.censorship_col])
        st_list = list(self.data_df.loc[:][self.survival_time_col])
        return np.array(cs_list), np.array(st_list)

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
        self.df_rna.insert(0, 'case_id', case_list)
        self.df_rna = self.df_rna.set_index('case_id')

    def setup_rna_pathways(self):
        """ Load Hallmarks biological pathways, which serve as the prototypes. """
        #signatures = pd.read_csv(os.path.join(self.data_source, f"../hallmarks_signatures.csv"))
        signatures = pd.read_csv(self.signatures_omics_path, index_col=0)
        self.rna_names = []
        self.pathway_names = []
        self.pathway_sizes = []

        # For each pathway 
        for col in signatures.columns:
            omic = signatures[col].dropna().unique()
            omic = sorted(_series_intersection(omic, self.df_rna.columns))

            # Store all genes invovlved
            self.rna_names.append(omic)

            # Store the name of the pathway
            self.pathway_names.append(col)

            # Store the number of genes in the pathway
            self.pathway_sizes.append(len(omic))

    def get_label_bins(self):
        """ Get the time bins for the discrete time labels."""
        return self.label_bins
    
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

    def __getitem__(self, idx):
        # Obtain labels
        survival_time, censorship, label = self.get_labels(idx)
        out = {'survival_time': torch.Tensor([survival_time]),
            'censorship': torch.Tensor([censorship]),
            'label': torch.Tensor([label])}
        
        # Obtain case and slide ids
        case_id = self.data_df.loc[idx]['case_id']
        slide_id = self.data_df.loc[idx][self.slide_col]
        out['case_id'] = case_id
        out['slide_id'] = slide_id

        # Obtain RNA data
        pathway_summary = []
        # For each pathway, obtain the expression values for all genes in this pathway
        for i in range(len(self.pathway_names)):
            pathway_summary.append(torch.Tensor(self.df_rna.loc[case_id, self.rna_names[i]]))
        
        out['rna'] = pathway_summary
    
        # Obtain WSI data
        if self.X is not None:
            # We already created the unsupervised slide embedding (slide summary)
            # For when we are oversampling
            if idx >= len(self.X):
                idx_img = self.find_corresponding_idx(idx)
                out['img'] = self.X[idx_img]
            else:
                out['img'] = self.X[idx]
        else:
            # Else obtain the image features
            feat_path = self.data_df.loc[idx]['fpath']
            # if file ends with .h5
            if feat_path.endswith('.h5'): #CLAM
                with h5py.File(feat_path, 'r') as f:
                    features = f['features'][:]
            elif feat_path.endswith('.pt'): #slide2vec
                features = torch.load(feat_path, map_location='cpu', weights_only=True).float()
            if len(features.shape) > 2:
                assert features.shape[0] == 1, f'{features.shape} is not compatible! It has to be (1, numOffeats, feat_dim) or (numOffeats, feat_dim)'
                features = np.squeeze(features, axis=0)
            
            # If feature is not a tensor, convert it to a tensor
            if not isinstance(features, torch.Tensor):
                features = torch.from_numpy(features)
            out['img'] = features

        return out