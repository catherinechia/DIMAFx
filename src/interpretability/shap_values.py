
import torch
import os
import shap
import sys
import numpy as np
from torch.utils.data import DataLoader

from embeddings.embeddings import prepare_embeddings
from models.DIMAFx import SHAP_DIMAFx, DIMAFxSurvival, DIMAFxClassifier
from data.mm_dataset import MMSurvivalDataset, MMClassificationDataset
from utils.general_utils import save_pkl
from survival.losses import NLLSurvLoss, CoxLoss, DisentangledSurvLoss
from classification.losses import CEClassLoss, SVMClassLoss


class ShapBase:
    """
    Shared base class to compute SHAP values of a trained DIMAFx model.
    Loads the train/test data, builds the model and computes the SHAP values of the test samples, given a background
    from the train samples. Task-specific classes such as ShapSurvival and ShapClassification only define the dataset,
    loss, model and explained output.
    """
    # Allowed values of args.shap_target. The first entry is the default.
    shap_targets = ('logits',)

    def __init__(self,
                 args,
                 fold,
                 post_attn='modal',
                 background_seed=None):
        """
        Args:
            - args              : All the arguments given by the user (Obj)
            - fold              : Specifies the fold number (int)
            - post_attn         : Which representation to explain: 'modal', 'post_attn' or 'post_attn_av' (str)
            - background_seed   : Seed for sampling the background distribution, None for unseeded (int)
        """
        self.args = args
        self.fold = fold
        self.post_attn = post_attn
        self.background_seed = background_seed
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Explained output
        self.shap_target = getattr(args, 'shap_target', None) or self.shap_targets[0]
        if self.shap_target not in self.shap_targets:
            sys.exit(f"SHAP target '{self.shap_target}' is not supported for {type(self).__name__} (choose from {self.shap_targets}), abborting....")

        # Dirs
        self.results_dir_fold = os.path.join(args.result_dir, f"Fold_{fold}/")
        self.pretrained_model_path = os.path.join(self.results_dir_fold, "model_checkpoint.pth")
        self.results_dir_shap = os.path.join(self.results_dir_fold, f'post_training/shap/{post_attn}')

    ############################################################
    # placeholders. if a new subclass is written, these functions need to be implemented
    def init_dataset(self, mode):
        """ Get the task-specific dataset of this split. """
        raise NotImplementedError

    def init_loss(self):
        """ Get the loss function and number of model outputs, as used during training. """
        raise NotImplementedError

    def init_model(self, rna_dims, histo_dim, num_classes, loss_fn):
        """ Get the (untrained) task-specific DIMAFx model. """
        raise NotImplementedError

    def get_output_fn(self, num_classes):
        """ Optional transform of the logits into the explained output, given args.shap_target. """
        return None

    def get_output_names(self, num_classes):
        """ Names of the explained outputs, None if there is a single output. """
        return None

    # ----- Shared pipeline -----

    def get_data(self, mode):
        """ Get the dataset of this split with its precomputed unsupervised slide embeddings. """
        dataset = self.init_dataset(mode)

        # The slide embeddings are created during training, SHAP only loads them
        embeddings_name = f"{mode}_{self.args.fm_type}_embeddings_wsi_proto_{self.args.n_proto}_em_{self.args.em_iter}_tau_{self.args.tau}.pkl"
        embeddings_path = os.path.join(self.args.proto_splits_dir, f'{self.fold}', embeddings_name)
        if not os.path.isfile(embeddings_path):
            raise FileNotFoundError(f"No slide embeddings found at {embeddings_path}. Train the model before you compute SHAP..")

        dataloader = DataLoader(dataset, batch_size=self.args.shap_bs, shuffle=False, num_workers=self.args.num_workers)
        dataloader, data_info = prepare_embeddings(self.args, mode, dataloader)

        return dataloader.dataset, data_info

    def get_feature_names(self, num_rna_proto):
        """ Names of the explained features of this representation. """
        num_proto_wsi = self.args.n_proto
        if self.post_attn == 'modal':
            return [f"wsi_pt_{i}" for i in range(num_proto_wsi)] + [f"rna_pt_{i}" for i in range(num_rna_proto)]
        elif self.post_attn == 'post_attn':
            return [f"rna_specific_{i}" for i in range(num_rna_proto)] + [f"wsi_rna_{i}" for i in range(num_rna_proto)] + [f"rna_wsi_{i}" for i in range(num_proto_wsi)] + [f"wsi_specific_{i}" for i in range(num_proto_wsi)]
        elif self.post_attn == 'post_attn_av':
            return ["wsi_rna_zhg", "rna_wsi_zgh", "rna_specific_zgg", "wsi_specific_zhh"]
        else:
            sys.exit("SHAP mode is not implemented, abborting....")

    def prepare_data(self, data, model):
        """ Compute the explained representation of all samples, with the feature names and sample ids. """
        dataloader = DataLoader(data, batch_size=model.batch_size, shuffle=False, num_workers=self.args.num_workers)
        if self.post_attn == 'modal':
            # Unimodal embeddings before fusion
            preproc_dataset = model.prep_data_pre_attn(dataloader)
        elif self.post_attn == 'post_attn':
            # Disentangled embeddings after fusion
            preproc_dataset = model.prep_data_post_attn(dataloader)
        elif self.post_attn == 'post_attn_av':
            # Disentangled, aggregated vectors
            preproc_dataset = model.prep_data_post_attn_av(dataloader)
        else:
            sys.exit("SHAP mode is not implemented, abborting....")

        feature_names = self.get_feature_names(len(data.pathway_sizes))
        samples = [data.data_df.loc[i][data.slide_col] for i in range(len(data))]

        return preproc_dataset, feature_names, samples

    def get_shap_model(self, data_info):
        """ Load the trained model and wrap it for SHAP. """
        loss_fn, num_classes = self.init_loss()
        dimafx = self.init_model(rna_dims=data_info['Pathway sizes'],
                                 histo_dim=data_info['Dim wsi'],
                                 num_classes=num_classes,
                                 loss_fn=loss_fn)
        dimafx.to(self.device)
        dimafx.from_pretrained(self.pretrained_model_path)

        model = SHAP_DIMAFx(model=dimafx,
                            device=self.device,
                            bs=self.args.shap_bs,
                            post_attn=self.post_attn,
                            output_fn=self.get_output_fn(num_classes))
        model.to(self.device)
        return model, num_classes

    def explain(self, model, background, test_data):
        """ Compute values with the given explainer. """
        if self.args.explainer == 'shap':
            print("Computing SHAP values....")
            explainer = shap.DeepExplainer(model, background)
            shap_values = explainer.shap_values(test_data, check_additivity=False)
        elif self.args.explainer == 'eg':
            print("Computing Expected Gradients values....")
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(test_data)
        else:
            sys.exit("Unspecified explainer! Abborting..")

        # Multiple outputs are returned as a list (older shap) or with a trailing output dim (newer shap)
        if isinstance(shap_values, list):
            shap_values = np.stack(shap_values, axis=-1)
        shap_values = np.asarray(shap_values)

        # Single output: [B, feats, dim]. Multiple outputs: [B, feats, dim, n_out]
        if shap_values.ndim == test_data.dim() + 1 and shap_values.shape[-1] == 1:
            shap_values = shap_values[..., 0]

        return shap_values

    def run(self):
        """ Obtain shap values for the trained model of this fold. """
        os.makedirs(self.results_dir_shap, exist_ok=True)

        # Create unimodal representations
        train_data, data_info = self.get_data('train')
        test_data, _ = self.get_data('test')

        # Obtain wrapper for SHAP
        model, num_classes = self.get_shap_model(data_info)

        # Obtain input data for the shap_module
        train_data, feature_names, samples_train = self.prepare_data(train_data, model)
        test_data, feature_names_test, samples_test = self.prepare_data(test_data, model)
        assert feature_names_test == feature_names

        # Background distribution
        if self.background_seed is not None:
            background = shap.sample(train_data.to(self.device), self.args.shap_refdist_n, random_state=self.background_seed)
        else:
            background = shap.sample(train_data.to(self.device), self.args.shap_refdist_n)

        test_data = test_data.to(self.device)
        shap_values = self.explain(model, background, test_data)

        # Save shap values
        print("Saving shap values..")
        if self.background_seed is not None:
            name = f'{self.args.explainer}_all_test_seed_{self.background_seed}'
        else:
            name = f'{self.args.explainer}_all_test_{self.args.shap_refdist_n}'

        save_pkl(self.results_dir_shap, f"{name}.pkl", {'shap values': shap_values,
                                                         'Feature names': feature_names,
                                                         'Samples': samples_test,
                                                         'Output names': self.get_output_names(num_classes)})


class ShapSurvival(ShapBase):
    """
        SHAP values of DIMAFx for survival prediction.
        shap_target 'logits' explains the raw model output (one value per time bin for NLL models),
        'risk' explains a single risk score per sample.
    """
    shap_targets = ('logits', 'risk')

    def init_dataset(self, mode):
        return MMSurvivalDataset(self.args, mode, self.fold)

    def init_loss(self):
        # Must match training so the classifier head has the same output size
        if self.args.loss_fn == 'nll':
            loss_fn = NLLSurvLoss(alpha=self.args.nll_alpha)
            num_classes = self.args.n_label_bins
        elif self.args.loss_fn == 'cox':
            loss_fn = CoxLoss()
            num_classes = 1
        else:
            loss_fn_split = self.args.loss_fn.split("_")
            loss_fn = DisentangledSurvLoss(loss_fn_split[0], loss_fn_split[1], weight_surv=self.args.w_surv, weight_disentanglement=self.args.w_dis, n_label_bins=self.args.n_label_bins, alpha=self.args.nll_alpha)
            num_classes = loss_fn.get_num_classes()
        return loss_fn, num_classes

    def init_model(self, rna_dims, histo_dim, num_classes, loss_fn):
        return DIMAFxSurvival(rna_dims=rna_dims,
                              histo_dim=histo_dim,
                              device=self.device,
                              single_out_dim=256,
                              num_classes=num_classes,
                              loss_fn=loss_fn,
                              aggr_post_embed=self.args.aggr_post_embed,
                              wsi_representation_type=self.args.wsi_repr,
                              num_proto_wsi=self.args.n_proto)

    def get_output_fn(self, num_classes):
        # A single output (Cox) already is the (log) risk score
        if self.shap_target == 'risk' and num_classes > 1:
            return nll_risk
        return None

    def get_output_names(self, num_classes):
        if self.shap_target == 'risk' or num_classes == 1:
            return None
        return [f"bin_{i}" for i in range(num_classes)]


class ShapClassification(ShapBase):
    """
        SHAP values of DIMAFx for classification.
        shap_target 'logits' explains the logit of every class.
    """
    shap_targets = ('logits',)

    def init_dataset(self, mode):
        return MMClassificationDataset(self.args, mode, self.fold)

    def init_loss(self):
        # Must match training so the classifier head has the same output size
        if self.args.loss_fn == 'ce':
            loss_fn = CEClassLoss(num_classes=self.args.n_classes)
        elif self.args.loss_fn == 'svm':
            loss_fn = SVMClassLoss(num_classes=self.args.n_classes, device=self.device)
        else:
            sys.exit("Classification loss is not implemented, aborting... ")
        return loss_fn, self.args.n_classes

    def init_model(self, rna_dims, histo_dim, num_classes, loss_fn):
        return DIMAFxClassifier(rna_dims=rna_dims,
                                histo_dim=histo_dim,
                                device=self.device,
                                single_out_dim=256,
                                num_classes=num_classes,
                                loss_fn=loss_fn,
                                aggr_post_embed=self.args.aggr_post_embed,
                                wsi_representation_type=self.args.wsi_repr,
                                num_proto_wsi=self.args.n_proto)

    def get_output_names(self, num_classes):
        return [f"class_{i}" for i in range(num_classes)]


def nll_risk(logits):
    """ Risk score of discrete time (NLL) logits, as in DIMAFxSurvival.compute_loss. """
    hazards = torch.sigmoid(logits)
    survival = torch.cumprod(1 - hazards, dim=1)
    return -torch.sum(survival, dim=1).unsqueeze(dim=1)


# Representations explained per fold
SHAP_REPRESENTATIONS = ['modal',          # Pre attention SHAP (modality experiment of 2 representations)
                        'post_attn',      # Post attention SHAP (shared/specific experiment of 4 representations)
                        'post_attn_av']   # Post attention and aggregation SHAP (shared/specific experiment of 4 representations)

def k_fold_shap(args, shap_cls):
    """ Obtain shap values for k-folds. """
    for i in range(args.folds):
        print(f"Fold {i}")
        for post_attn in SHAP_REPRESENTATIONS:
            shap_cls(args, i, post_attn).run()

def k_fold_shap_stable(args, shap_cls, seeds=(1, 2, 3, 4, 5)):
    """ Obtain shap values for k-folds, for multiple background seeds. """
    for seed in seeds:
        for i in range(args.folds):
            print(f"Fold {i}")
            for post_attn in SHAP_REPRESENTATIONS:
                shap_cls(args, i, post_attn, background_seed=seed).run()

def survival_shap(args, fold, post_attn='modal', background_seed=None):
    """ Obtain shap values for trained survival prediction model. Kept for backwards compatibility. """
    ShapSurvival(args, fold, post_attn, background_seed).run()
