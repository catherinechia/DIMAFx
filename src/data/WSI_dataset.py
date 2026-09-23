import os
import json
import torch
import h5py
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from utils.data_utils import pd_diff

# WSI extensions that slide2vec keeps as part of its sample id (e.g. a tile embedding file for "filename.svs" is saved as "filename.svs.pt").
# Split files in this repo use the slide id without this extension, so it is stripped when normalizing slide ids for the 'pt' feat_format.
l_wsi_extensions = ('.svs', '.ndpi', '.tif', '.tiff', '.mrxs', '.scn', '.vms', '.vmu', '.svslide', '.bif', '.qptiff', '.dcm')


class WSIClusterDataset(Dataset):
    """
        WSI Feature Dataset
        Used for creating the prototypes
        Adapted from https://github.com/mahmoodlab/MMP/blob/main/src/wsi_datasets/wsi_prototype.py

        Supports two on-disk feature formats:
            - 'h5'  : CLAM/TRIDENT-style '<slide_id>.h5' files with a 'features' dataset
                      (and usually a 'coords' dataset). This is the original format.
            - 'pt'  : slide2vec output, i.e. one '<sample_id>.pt' file per slide holding a
                      plain (n_tiles, feat_dim) tensor of patch features. Tile coordinates
                      are optional and, if present, are stored alongside as
                      '<sample_id>.coordinates.npz' (keys: tile_index, x, y, tissue_fractions)
                      with an optional '<sample_id>.coordinates.meta.json' sidecar describing
                      how the tiles/features were extracted.
        """
    def __init__(self,
                 feat_dir,
                 split_file,
                 slide_col='slide_id',
                 feat_format='auto',
                 coord_dir=None,
                 bool_strip_slide_ext=True,
                 bool_return_coords=False):
        """
        Args:
            - feat_dir        : Dir of the WSI features, i.e. the '.h5' files ('h5' format) or  the '.pt' files ('pt' / slide2vec format) (str)
            - split_file      : Dir of the split file (str)
            - slide_col       : Name of the column storing the slide ids (str)
            - feat_format     : Feature file format: 'h5', 'pt' (slide2vec), or 'auto' to detect it from the files present in feat_dir (str)
            - coord_dir       : For 'pt' format only, dir holding the slide2vec'.coordinates.npz' / '.coordinates.meta.json' sidecars, if they are stored separately from feat_dir. Defaults to feat_dir (str)
            - bool_strip_slide_ext : For 'pt' format only, strip a trailing WSI image extension (e.g. '.svs') from the slide2vec sample id so it matches the slide ids used in the split file (bool)
            - bool_return_coords   : For 'pt' format only, return a dict with the tile coordinates and metadata alongside the features instead of just the features tensor (bool)
        """

        # Dataframe with the splits and all the clinical data
        self.data_df = pd.read_csv(split_file)
        self.slide_col = slide_col
        self.feat_dir = feat_dir
        self.coord_dir = coord_dir if coord_dir is not None else feat_dir
        self.bool_strip_slide_ext = bool_strip_slide_ext
        self.bool_return_coords = bool_return_coords

        # Make sure sample col is a string
        self.data_df[slide_col] = self.data_df[slide_col].astype(str)

        # Resolve the feature format before validating the features directory
        self.feat_format = self.detect_feat_format() if feat_format == 'auto' else feat_format
        assert self.feat_format in ('h5', 'pt'), f"feat_format must be 'h5', 'pt' or 'auto', got '{self.feat_format}'"

        # Check the dataframe and features directory
        self.check_df_file()
        # Obtain the paths of the features
        self.obtain_feat_paths()

    def detect_feat_format(self):
        """ Auto-detect the feature format from the extensions present in the features directory. """
        exts = {os.path.splitext(e.name)[1].lower() for e in os.scandir(self.feat_dir) if e.is_file()}
        if '.h5' in exts:
            return 'h5'
        if '.pt' in exts:
            return 'pt'
        raise ValueError(f"Could not auto-detect feat_format in '{self.feat_dir}' (found extensions {exts}). "
                          "Pass feat_format='h5' or feat_format='pt' explicitly.")

    def normalize_slide_id(self, stem):
        """ Strip a trailing WSI image extension (e.g. '.svs') from a slide2vec sample id. """
        if not self.bool_strip_slide_ext:
            return stem
        root, ext = os.path.splitext(stem)
        return root if ext.lower() in l_wsi_extensions else stem

    def obtain_feat_paths(self):
        """ Obtain paths of features used for creating the prototypes. """
        if self.feat_format == 'h5':
            self.feats_df = pd.DataFrame(
                [(e.path, os.path.splitext(e.name)[0]) for e in os.scandir(self.feat_dir) if e.name.endswith('.h5')],
                columns=['fpath', self.slide_col]
            ).reset_index(drop=True)
        else:
            # slide2vec 'pt' format: '<sample_id>.pt' features in feat_dir, with optional '<sample_id>.coordinates.npz' / '<sample_id>.coordinates.meta.json' in coord_dir (defaults to feat_dir).
            records = []
            for e in os.scandir(self.feat_dir):
                if not e.name.endswith('.pt'):
                    continue
                stem = e.name[:-len('.pt')]
                coords_path = os.path.join(self.coord_dir, f'{stem}.coordinates.npz')
                meta_path = os.path.join(self.coord_dir, f'{stem}.coordinates.meta.json')
                records.append((
                    e.path,
                    self.normalize_slide_id(stem),
                    coords_path if os.path.isfile(coords_path) else None,
                    meta_path if os.path.isfile(meta_path) else None,
                ))
            self.feats_df = pd.DataFrame(
                records, columns=['fpath', self.slide_col, 'coords_path', 'meta_path']
            ).reset_index(drop=True)

        self.check_wsi_files()

        # Bring the newly obtained feature columns (fpath, and for 'pt' also coords_path/meta_path) to the front of the dataframe.
        feat_cols = [c for c in self.feats_df.columns if c != self.slide_col]
        other_cols = [c for c in self.data_df.columns if c not in feat_cols]
        self.data_df = self.data_df[feat_cols + other_cols]

    def check_df_file(self):
        assert 'Unnamed: 0' not in self.data_df.columns
        if self.feat_format == 'h5':
            assert 'feats_h5' in self.feat_dir
        assert len(list(self.data_df[self.slide_col].astype(str).unique())) == len(list(self.data_df[self.slide_col].astype(str))), 'There are duplicates in the given splits file...'

    def check_wsi_files(self):
        """ Check that the wsi files are complete and that there are no duplicates """
        # Should be no missing files
        missing_feats_in_split = pd_diff(self.data_df[self.slide_col], self.feats_df[self.slide_col])
        assert len(missing_feats_in_split) == 0, f'Missing Features in Split:\n{missing_feats_in_split}'

        # All slide ids to feature paths should have a one-to-one mapping. Raises ValueError if not.
        # Add feature paths to data frame
        self.data_df = self.data_df.merge(self.feats_df, how='left', on=self.slide_col, validate='1:1')

        duplicates = self.feats_df[self.slide_col].duplicated()
        assert duplicates.sum() == 0, f'Features duplicated in data source(s):{self.feats_df[duplicates].to_string()}'

        print("Dataset check is complete!")

    def __len__(self):
        """ Get the total number of samples """
        return len(self.data_df)

    def load_coords(self, row):
        """ Load the slide2vec tile coordinates/metadata (if any) for a data_df row. """
        coords, tile_index, tissue_fractions, meta = None, None, None, None

        coords_path = row.get('coords_path')
        if isinstance(coords_path, str) and os.path.isfile(coords_path):
            with np.load(coords_path) as npz:
                coords = np.stack([npz['x'], npz['y']], axis=1)
                tile_index = npz['tile_index']
                tissue_fractions = npz['tissue_fractions']

        meta_path = row.get('meta_path')
        if isinstance(meta_path, str) and os.path.isfile(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)

        return coords, tile_index, tissue_fractions, meta

    def __getitem__(self, idx):
        row = self.data_df.loc[idx]
        feat_path = row['fpath']

        if self.feat_format == 'pt':
            # slide2vec stores a plain (n_tiles, feat_dim) tensor per slide.
            features = torch.load(feat_path, map_location='cpu', weights_only=True).float()
            #features = torch.load(feat_path, map_location='cpu', weights_only=True).float().clone()
        else:
            with h5py.File(feat_path, 'r') as f:
                features = f['features'][:]

            # Check the shape of the features
            if len(features.shape) > 2:
                assert features.shape[0] == 1, f'{features.shape} is not compatible! It has to be (1, numOffeats, feat_dim) or (numOffeats, feat_dim)'
                features = np.squeeze(features, axis=0)

            features = torch.from_numpy(features)
            #features = torch.from_numpy(features).clone()

        if not self.bool_return_coords:
            return features

        coords, tile_index, tissue_fractions, meta = self.load_coords(row)
        return {'features': features, 'coords': coords, 'tile_index': tile_index, 'tissue_fractions': tissue_fractions, 'meta': meta}



class WSIDataset(Dataset):
    """
        WSI Dataset for visualization purposes
    """
    def __init__(self,
                 data_source, 
                 wsi_feats,
                 mode, 
                 fold,
                 slide_col='slide_id'):
        """
        Args:
            - data_source   : Dir of the data (str)
            - wsi_feats     : TYpe of WSI feats (str)
            - mode          : Train or Test (str)
            - fold          : Fold/split number (int)
            - slide_col     : Name of the column storing the slide ids (str)
        """
        self.mode = mode
        self.fold = fold
        self.data_source = data_source
        self.split_dir = os.path.join(self.data_source, f'splits/{self.fold}/')

        # WSI args
        self.slide_col = slide_col
        self.wsi_feats = wsi_feats
        # Will store the slide summary
        self.X = None

        # Setup and check split data
        self.init_df()
        # Setup and check WSI data
        self.init_df_wsi()


    def check_wsi_files(self, feats_wsi_df):
        """ Check that the wsi files are complete and that there are no duplicates. """
        # Should be no missing files
        missing_feats_in_split = pd_diff(self.data_df[self.slide_col], feats_wsi_df[self.slide_col])
        assert len(missing_feats_in_split) == 0, f'Missing Features in Split:\n{missing_feats_in_split}'

        # Should be no duplicates
        duplicates = feats_wsi_df[self.slide_col].duplicated()
        assert duplicates.sum() == 0, f'Features duplicated in data source(s):{feats_wsi_df[duplicates].to_string()}'


    def init_df(self):
        """ Set up clinical data of this split. """
        split_file = os.path.join(self.split_dir, f'{self.mode}_filtered.csv')
        self.data_df = pd.read_csv(split_file)

        # Should be no duplicates in splits file
        assert len(list(self.data_df[self.slide_col].astype(str).unique())) == len(list(self.data_df[self.slide_col].astype(str))), 'There are duplicates in the given splits file...'


    def init_df_wsi(self):
        """ Set up WSI data of this split. """
        # Obtain directory containing the patch features
        self.feat_dir_wsi = os.path.join(self.data_source, f"wsi/{self.wsi_feats}/feats_h5")
        self.data_df[self.slide_col] = self.data_df[self.slide_col].astype(str)

        # Store feature paths 
        feats_wsi_df = pd.DataFrame([(e.path, os.path.splitext(e.name)[0]) for e in os.scandir(self.feat_dir_wsi)], columns=['fpath', self.slide_col]).reset_index(drop=True)
        self.check_wsi_files(feats_wsi_df)

        # All slide ids to feature paths should have a one-to-one mapping. Raises ValueError if not.
        # Add feature paths to data frame
        self.data_df = self.data_df.merge(feats_wsi_df, how='inner', on=self.slide_col, validate='1:1')
        self.data_df = self.data_df[list(self.data_df.columns[-1:]) + list(self.data_df.columns[:-1])]


    def __len__(self):
        """ Get the number of samples. """
        return len(self.data_df)
    

    def __getitem__(self, idx):
        # Obtain case and slide id
        out = {'case_id': self.data_df.loc[idx]['case_id'],
            'slide_id': self.data_df.loc[idx][self.slide_col]}
    
        if self.X is not None:
            # We already created the unsupervised slide embedding (slide summary)
            out['img'] = self.X[idx]
        else:
            # Else obtain the image features
            feat_path = self.data_df.loc[idx]['fpath']
            with h5py.File(feat_path, 'r') as f:
                features = f['features'][:]

            if len(features.shape) > 2:
                assert features.shape[0] == 1, f'{features.shape} is not compatible! It has to be (1, numOffeats, feat_dim) or (numOffeats, feat_dim)'
                features = np.squeeze(features, axis=0)
            
            features = torch.from_numpy(features)
            out['img'] = features

        return out