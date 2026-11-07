import torch
import torch.nn as nn
import sys

from utils.train_utils import list_to_device

class SHAP_DIMAFx(nn.Module):
    """ SHAP wrapper for DIMAFx. Task agnostic: wraps any trained DIMAFxBase subclass (e.g. DIMAFxSurvival, DIMAFxClassifier). """
    def __init__(
            self,
            model,
            device,
            bs,
            post_attn,
            output_fn=None):
        """
        Args:
            - model         : Trained DIMAFx model, a DIMAFxBase subclass (nn.Module)
            - device        : Device to run on (torch.device)
            - bs            : Batch size of the SHAP forward passes (int)
            - post_attn     : Which representation to explain: 'start', 'modal', 'post_attn' or 'post_attn_av' (str)
            - output_fn     : Optional transform of the logits into the explained output, e.g. a risk score (callable)
        """

        super(SHAP_DIMAFx, self).__init__()

        # SHAP params
        self.post_attn = post_attn
        self.output_fn = output_fn

        # Run params
        self.batch_size = bs
        self.device = device

        # Trained model
        self.model = model

        # Data params
        self.pathway_sizes = model.rna_dims
        self.num_proto_wsi = model.nr_wsi_prototypes

        # Put trained model to eval
        self.model.eval()

    def forward_post_attn(self, data):
        """ Forward loop to compute SHAP values for disentangled representations just after fusion. """
        all_logits = []

        # Get the total number of samples
        num_samples = data.size(0)
        bs = self.batch_size

        # Loop through the dataset in batches
        for i in range(0, num_samples, bs):
            # Slice the tensors to get the current batch
            if bs + i > num_samples:
                bs = num_samples-i

            post_attn_batch = data[i:i+bs] # [B, 132, 144]

            out = self.model.forward_shap_post_attn(post_attn_batch)
            all_logits.append(out)

        logits = torch.cat(all_logits, dim=0)
        return logits

    def forward_post_attn_av(self, data):
        """ Forward loop to compute SHAP values for disentangled, aggregated vectors."""
        all_logits = []

        # Get the total number of samples
        num_samples = data.size(0)
        bs = self.batch_size

        # Loop through the dataset in batches
        for i in range(0, num_samples, bs):
            # Slice the tensors to get the current batch
            if bs + i > num_samples:
                bs = num_samples-i

            post_attn_av_batch = data[i:i+bs] # [B, 4, Dz]
            reshaped_tensor = post_attn_av_batch.reshape(bs, 4*post_attn_av_batch.shape[-1])
            out = self.model.classifier(reshaped_tensor)
            all_logits.append(out)

        logits = torch.cat(all_logits, dim=0)
        return logits

    def forward_pre_attn(self, data):
        """ Forward loop to compute SHAP values for the unimodal embeddings. """
        wsi_tensor, rna_tensor = data[:, :self.num_proto_wsi, :], data[:, self.num_proto_wsi:, :]
        all_logits = []

        # Get the total number of samplesxs
        num_samples = wsi_tensor.size(0)
        bs = self.batch_size

        # Loop through the dataset in batches
        for i in range(0, num_samples, bs):
            # Slice the tensors to get the current batch
            if bs + i > num_samples:
                bs = num_samples-i

            wsi_batch = wsi_tensor[i:i+bs] # [B, 16, 256]
            rna_batch = rna_tensor[i:i+bs] # [B, 50, 256]

            out = self.model.forward_shap_modal(wsi_batch, rna_batch)
            all_logits.append(out)

        logits = torch.cat(all_logits, dim=0)
        return logits

    def forward_start_attn(self, data):
        """ Forward loop to compute SHAP values for the input. """
        wsi_tensor, rna_tensor = data[:, :self.num_proto_wsi, :], data[:, self.num_proto_wsi:, :]
        all_logits = []

        # Get the total number of samplesxs
        num_samples = wsi_tensor.size(0)
        bs = self.batch_size

        # Loop through the dataset in batches
        for i in range(0, num_samples, bs):
            # Slice the tensors to get the current batch
            if bs + i > num_samples:
                bs = num_samples-i

            wsi_batch = wsi_tensor[i:i+bs]
            rna_batch = rna_tensor[i:i+bs]

            rna_batch_list = []
            for i in range(len(self.pathway_sizes)):
                rna_list = rna_batch[:, i, :self.pathway_sizes[i]]
                rna_batch_list.append(rna_list)

            out = self.model.forward_mm_no_loss(wsi_batch, rna_batch_list)

            all_logits.append(out['logits'])

        logits = torch.cat(all_logits, dim=0)
        return logits

    def prep_data_post_attn_av(self, dataloader):
        """ Prep data to compute SHAP for disentangled, aggregated vectors. """
        proc_dataset = []
        for batch in dataloader:
            # Separate features and labels
            batch_wsi = batch['img'].to(self.device)
            batch_rna = list_to_device(batch['rna'], self.device)
            post_attn_tokens = self.model.compute_post_attn_tokens_av(batch_wsi, batch_rna)
            proc_dataset.append(post_attn_tokens)

        mm_data_tensor = torch.cat(proc_dataset, dim=0)
        return mm_data_tensor

    def prep_data_post_attn(self, dataloader):
        """ Prep data to compute SHAP for disentangled representations just after fusion. """
        proc_dataset = []
        for batch in dataloader:
            # Separate features and labels
            batch_wsi = batch['img'].to(self.device)
            batch_rna = list_to_device(batch['rna'], self.device)
            post_attn_tokens = self.model.compute_post_attn_tokens(batch_wsi, batch_rna)
            proc_dataset.append(post_attn_tokens)

        mm_data_tensor = torch.cat(proc_dataset, dim=0)
        return mm_data_tensor

    def prep_data_pre_attn(self, dataloader):
        """ Prep data to compute SHAP for unimodal embeddings. """
        proc_dataset = []
        for batch in dataloader:
            # Separate features and labels
            batch_wsi = batch['img'].to(self.device)
            batch_rna = list_to_device(batch['rna'], self.device)

            wsi_post_fusion, rna_post_fusion = self.model.compute_pre_attn_tokens(batch_wsi, batch_rna)
            mm_data = torch.cat((wsi_post_fusion, rna_post_fusion), dim=1)
            proc_dataset.append(mm_data)

        mm_data_tensor = torch.cat(proc_dataset, dim=0)
        return mm_data_tensor

    def __call__(self, data):
        """ Compute SHAP for a specific feature types. """
        # Input
        if self.post_attn == 'start':
            logits = self.forward_start_attn(data)
        # Unimodal embeddings just before fusion
        elif self.post_attn == 'modal':
            logits = self.forward_pre_attn(data)
        # Multimodal embeddings just after fusion
        elif self.post_attn == 'post_attn':
            logits = self.forward_post_attn(data)
        # Disentangled and aggregated multimodal vectors
        elif self.post_attn == 'post_attn_av':
            logits = self.forward_post_attn_av(data)
        else:
            sys.exit("SHAP mode is not implemented, abborting....")

        # Transform the logits into the explained output (e.g. risk score)
        if self.output_fn is not None:
            return self.output_fn(logits)
        return logits
