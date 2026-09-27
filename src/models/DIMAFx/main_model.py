import torch
import torch.nn as nn
import sys

from .layers import MultiSNN, CrossAttentionLayer, PrototypeAggregator, FeedForwardEnsemble
from survival.losses import NLLSurvLoss, CoxLoss,  DisentangledSurvLoss
from classification.losses import CEClassLoss, SVMClassLoss

class DIMAFxBase(nn.Module):
    """
    Shared backbone of Explainable Disentangled and Interpretable Multimodal Attention Fusion.

    Builds the disentangled RNA/WSI cross-attention architecture and exposes the
    fusion pipeline (pre-attention embeddings, disentangled attention fusion, SHAP
    forward passes, checkpoint loading) reused by task-specific heads such as
    DIMAFxSurvival and DIMAFxClassifier.
    """
    def __init__(
            self,
            rna_dims,
            histo_dim,
            device,
            num_classes=1,
            single_out_dim=256,
            loss_fn='cox',
            aggr_post_embed='weighted_mean',
            num_proto_wsi=16,
            wsi_representation_type='importance',
            output_fnn_type='none'):
        """
        Args:
            - rna_dims                : Dimensions of the pathways (list)
            - histo_dim               : Dimension of the histology input features (int)
            - device                  : Device to run the model on, e.g. 'cpu' or 'cuda' (str)
            - num_classes             : Number of output classes (int), default=1
            - single_out_dim          : Output dimension of the unimodal embeddings (int), default=256
            - loss_fn                 : Loss function to use, e.g. 'cox' (str), default='cox'
            - aggr_post_embed         : Aggregation method of each disetnagled representation, default='weighted_mean'
            - num_proto_wsi           : Number of prototypes for WSI representation (int), default=16
            - wsi_representation_type : Type of WSI representation to use (str), default in DIMAFx is 'importance'
        """

        super(DIMAFxBase, self).__init__()

        self.device = device

        # Input args
        self.rna_dims = rna_dims
        self.histo_dim = histo_dim

        # Output args
        self.num_classes = num_classes

        # Architecture args
        self.single_out_dim = single_out_dim
        self.nr_wsi_prototypes = num_proto_wsi
        self.nr_rna_prototypes = len(rna_dims)
        self.wsi_representation_type = wsi_representation_type # normal or importance
        self.aggr_post_embed = aggr_post_embed
        self.output_fnn_type = output_fnn_type

        # Loss function
        self.loss_fn = loss_fn

        # Create the architecture
        self.create_mm_architecture()


    def create_mm_architecture(self):
        """ Create the DIMAFx architecture. """
        # Histology embeddings
        if self.wsi_representation_type == "importance":
            self.wsi_pre_fnn = nn.Sequential(nn.Linear(self.histo_dim-1, self.single_out_dim-1))
        else:
            self.wsi_pre_fnn = nn.Sequential(nn.Linear(self.histo_dim, self.single_out_dim))

        # Rna embeddings
        self.rna_pre_fnn = MultiSNN(self.rna_dims, self.single_out_dim)

        # Get the type embeddings to each prototype
        self.single_out_dim, self.wsi_pt_embedding, self.rna_pt_embedding = self.get_pt_embed()

        multi_out_dim = self.single_out_dim // 2

        # 4 seperate attention blocks
        self.rna_attention = CrossAttentionLayer(
                dim=self.single_out_dim,
                dim_head=multi_out_dim,
                heads=1)

        self.wsi_attention = CrossAttentionLayer(
                dim=self.single_out_dim,
                dim_head=multi_out_dim,
                heads=1)

        self.cross_attention_rna_wsi = CrossAttentionLayer(
                dim=self.single_out_dim,
                dim_head=multi_out_dim,
                heads=1)

        self.cross_attention_wsi_rna = CrossAttentionLayer(
                dim=self.single_out_dim,
                dim_head=multi_out_dim,
                heads=1)

        if self.output_fnn_type == 'indiv':
            # Feedforward ensemble --> Like MMP
            self.output_fnn = FeedForwardEnsemble(multi_out_dim, dropout=0.1, num=2*(self.nr_wsi_prototypes + self.nr_rna_prototypes))
        else:
            # DIMAFx uses none
            self.output_fnn = None

        self.layer_norm = nn.LayerNorm(multi_out_dim)

        # Cox PH risk predictor
        out_classifier_dim = 4 * multi_out_dim
        self.classifier = nn.Linear(out_classifier_dim, self.num_classes, bias=False)

        if self.aggr_post_embed == 'mean':
            self.aggr_rna = torch.mean
            self.aggr_wsi = torch.mean
            self.aggr_rna_wsi = torch.mean
            self.aggr_wsi_rna = torch.mean
        elif self.aggr_post_embed == 'weighted_mean':
            self.aggr_rna = PrototypeAggregator(multi_out_dim, 50)
            self.aggr_wsi = PrototypeAggregator(multi_out_dim, self.nr_wsi_prototypes)
            self.aggr_rna_wsi = PrototypeAggregator(multi_out_dim, self.nr_wsi_prototypes)
            self.aggr_wsi_rna = PrototypeAggregator(multi_out_dim, 50)
        else:
            sys.exit("Unspecified post attention prototype aggregation method! Abborting..")


    def get_pt_embed(self):
        """
        Per-prototype learnable/non-learnable embeddings to append to the original prototype embeddings
        """
        append_dim = 32
        path_proj_dim_new = self.single_out_dim + append_dim

        # Learnable encoding per prototype
        histo_embedding = torch.nn.Parameter(torch.randn(1, self.nr_wsi_prototypes, append_dim), requires_grad=True)
        gene_embedding = torch.nn.Parameter(torch.randn(1, self.nr_rna_prototypes, append_dim), requires_grad=True)


        return path_proj_dim_new, histo_embedding, gene_embedding

    def append_embed(self, wsi_embed, rna_embed):
        bs = wsi_embed.size(0)
        # Append gene prototype encoding

        rna_pt_embedding_exp = self.rna_pt_embedding.expand(bs, -1, -1)
        rna_pre_fusion_exp = torch.cat([rna_embed, rna_pt_embedding_exp], dim=-1)

        wsi_pt_embedding_exp = self.wsi_pt_embedding.expand(bs, -1, -1)
        wsi_pre_fusion_exp = torch.cat([wsi_embed, wsi_pt_embedding_exp], dim=-1)


        return rna_pre_fusion_exp, wsi_pre_fusion_exp

    def disentangled_attention_fusion(self, wsi_pre_fusion_exp, rna_pre_fusion_exp):
        # Pass through disentangled fusion

        # B, 50, multi_out_dim
        post_self_rna = self.rna_attention(rna_pre_fusion_exp, rna_pre_fusion_exp)

        # B, 50, multi_out_dim
        post_cross_wsi_rna = self.cross_attention_wsi_rna(rna_pre_fusion_exp, wsi_pre_fusion_exp)

        # B, 16, multi_out_dim
        post_cross_rna_wsi = self.cross_attention_rna_wsi(wsi_pre_fusion_exp, rna_pre_fusion_exp)

        # B, 16, multi_out_dim
        post_self_wsi = self.wsi_attention(wsi_pre_fusion_exp, wsi_pre_fusion_exp)

        # Concat
        post_attn_tokens = torch.cat([post_self_rna, post_cross_wsi_rna, post_cross_rna_wsi, post_self_wsi], dim=1)
        return post_attn_tokens


    def compute_pre_attn_tokens(self, wsi, rna):
        """ Compute RNA and WSI embeddings. """

        # wsi embeddings
        if self.wsi_representation_type == "normal":
            wsi_pre_fusion = self.wsi_pre_fnn(wsi)
        else:
            importance = wsi[:, :, 0].unsqueeze(-1)
            wsi_pre_fusion_int = self.wsi_pre_fnn(wsi[:, :, 1:])
            wsi_pre_fusion = torch.cat([importance, wsi_pre_fusion_int], dim=-1)

        # Rna embeddings
        rna_pre_fusion = self.rna_pre_fnn(rna)

        return wsi_pre_fusion, rna_pre_fusion

    def compute_post_attn_tokens(self, wsi, rna):
        """ Compute disentangled embeddings. """
        # get RNA and WSI embeddings
        wsi_pre_fusion, rna_pre_fusion = self.compute_pre_attn_tokens(wsi, rna)

        # Get rna and wsi embeddings with pt encoding
        rna_pre_fusion_exp, wsi_pre_fusion_exp = self.append_embed(wsi_pre_fusion, rna_pre_fusion)

        # Fusion
        post_attn_tokens = self.disentangled_attention_fusion(wsi_pre_fusion_exp, rna_pre_fusion_exp)



        return post_attn_tokens


    def compute_post_attn_tokens_av(self, wsi, rna):
        """ Compute disentangled, aggregated vectors. """
        # Get disentangled embeddings
        post_attn_tokens = self.compute_post_attn_tokens(wsi, rna)
        if self.output_fnn_type == 'indiv':
            fused_tokens = self.output_fnn(post_attn_tokens)
        else:
            fused_tokens = post_attn_tokens

        # Normalize
        fused_norm_tokens = self.layer_norm(fused_tokens)

        # Aggregate the protypes per disentangled representation
        rna_norm_tokens = fused_norm_tokens[:, :self.nr_rna_prototypes, :]
        rna_norm_tokens_aggr = self.aggr_rna(rna_norm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes
        wsi_rna_norm_tokens = fused_norm_tokens[:, count:count + self.nr_rna_prototypes, :]
        wsi_rna_norm_tokens_aggr = self.aggr_wsi_rna(wsi_rna_norm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes + self.nr_rna_prototypes
        rna_wsi_norm_tokens = fused_norm_tokens[:, count:count+self.nr_wsi_prototypes, :]
        rna_wsi_norm_tokens_aggr = self.aggr_rna_wsi(rna_wsi_norm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes + self.nr_rna_prototypes + self.nr_wsi_prototypes
        wsi_norm_tokens = fused_norm_tokens[:, count:, :]
        wsi_norm_tokens_aggr = self.aggr_wsi(wsi_norm_tokens, dim=1) # B, dim

        # stack the 4 disentangled, aggregated embeddings
        embedding = torch.stack([wsi_rna_norm_tokens_aggr, rna_wsi_norm_tokens_aggr, rna_norm_tokens_aggr, wsi_norm_tokens_aggr], dim=1)

        return embedding


    def forward_shap_post_attn(self, post_attn_tokens):
        """" Forward SHAP pass from the disentangled representations. """

        if self.output_fnn_type == 'indiv':
            fused_tokens = self.output_fnn(post_attn_tokens)
        else:
            fused_tokens = post_attn_tokens

        # Normalize
        fused_norm_tokens = self.layer_norm(fused_tokens)

        # Aggregate the protypes per disentangled representation
        rna_norm_tokens = fused_norm_tokens[:, :self.nr_rna_prototypes, :]
        rna_norm_tokens_aggr = self.aggr_rna(rna_norm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes
        wsi_rna_norm_tokens = fused_norm_tokens[:, count:count + self.nr_rna_prototypes, :]
        wsi_rna_norm_tokens_aggr = self.aggr_wsi_rna(wsi_rna_norm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes + self.nr_rna_prototypes
        rna_wsi_norm_tokens = fused_norm_tokens[:, count:count+self.nr_wsi_prototypes, :]
        rna_wsi_norm_tokens_aggr = self.aggr_rna_wsi(rna_wsi_norm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes + self.nr_rna_prototypes + self.nr_wsi_prototypes
        wsi_norm_tokens = fused_norm_tokens[:, count:, :]
        wsi_norm_tokens_aggr = self.aggr_wsi(wsi_norm_tokens, dim=1) # B, dim

        embedding = torch.concat([wsi_rna_norm_tokens_aggr, rna_wsi_norm_tokens_aggr, rna_norm_tokens_aggr, wsi_norm_tokens_aggr], dim=1)

        logits = self.classifier(embedding)

        return logits



    def forward_shap_modal(self, wsi_batch, rna_batch):
        """" Forward SHAP pass from the unimodal representations. """
        rna_pre_fusion_exp, wsi_pre_fusion_exp = self.append_embed(wsi_batch, rna_batch)

        post_attn_tokens = self.disentangled_attention_fusion(wsi_pre_fusion_exp, rna_pre_fusion_exp)

        logits = self.forward_shap_post_attn(post_attn_tokens)

        return logits

    def forward_mm_no_loss(self, wsi, rna, return_attn=False):

        bs = wsi.size(0)

        # Get wsi embeddings
        if self.wsi_representation_type == "normal":
            wsi_emb = self.wsi_pre_fnn(wsi)
        else:
            importance = wsi[:, :, 0].unsqueeze(-1)
            wsi_emb_int = self.wsi_pre_fnn(wsi[:, :, 1:])
            wsi_emb = torch.cat([importance, wsi_emb_int], dim=-1)


        # Rna embeddings
        rna_emb = self.rna_pre_fnn(rna)

        # Append prototype encoding
        rna_pt_embedding_exp = self.rna_pt_embedding.expand(bs, -1, -1)
        rna_emb_exp = torch.cat([rna_emb, rna_pt_embedding_exp], dim=-1)

        wsi_pt_embedding_exp = self.wsi_pt_embedding.expand(bs, -1, -1)
        wsi_emb_exp = torch.cat([wsi_emb, wsi_pt_embedding_exp], dim=-1)


        # Required for visualization
        if return_attn:
            with torch.no_grad():
                # B, 50, dim
                _, self_attn_rna = self.rna_attention(rna_emb_exp, rna_emb_exp, return_attention=True)
                # B, 50, dim: Histology --> Pathway attention
                _, cross_attn_wsi_rna = self.cross_attention_wsi_rna(rna_emb_exp, wsi_emb_exp, return_attention=True)
                # B, 16, dim: Pathway --> Histlogy attention
                _, cross_attn_rna_wsi = self.cross_attention_rna_wsi(wsi_emb_exp, rna_emb_exp, return_attention=True)
                # B, 16, dim
                _, self_attn_wsi = self.wsi_attention(wsi_emb_exp, wsi_emb_exp, return_attention=True)


        # Pass through disentangled fusion
        # B, 50, multi_out_dim
        post_self_rna = self.rna_attention(rna_emb_exp, rna_emb_exp)

        # B, 50, multi_out_dim
        post_cross_wsi_rna = self.cross_attention_wsi_rna(rna_emb_exp, wsi_emb_exp)

        # B, 16, multi_out_dim
        post_cross_rna_wsi = self.cross_attention_rna_wsi(wsi_emb_exp, rna_emb_exp)

        # B, 16, multi_out_dim
        post_self_wsi = self.wsi_attention(wsi_emb_exp, wsi_emb_exp)

        # Post fusion processing
        post_attn_tokens = torch.cat([post_self_rna, post_cross_wsi_rna, post_cross_rna_wsi, post_self_wsi], dim=1)
        if self.output_fnn_type == 'indiv':
            fused_tokens = self.output_fnn(post_attn_tokens)
        else:
            fused_tokens = post_attn_tokens

        fused_norm_tokens = self.layer_norm(fused_tokens)

        # Aggregate the protypes per disentangled representation
        rna_norm_tokens = fused_norm_tokens[:, :self.nr_rna_prototypes, :]
        rna_norm_tokens_aggr = self.aggr_rna(rna_norm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes
        wsi_rna_norm_tokens = fused_norm_tokens[:, count:count + self.nr_rna_prototypes, :]
        wsi_rna_norm_tokens_aggr = self.aggr_wsi_rna(wsi_rna_norm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes + self.nr_rna_prototypes
        rna_wsi_norm_tokens = fused_norm_tokens[:, count:count+self.nr_wsi_prototypes, :]
        rna_wsi_norm_tokens_aggr = self.aggr_rna_wsi(rna_wsi_norm_tokens, dim=1) # B, dim

        count = self.nr_rna_prototypes + self.nr_rna_prototypes + self.nr_wsi_prototypes
        wsi_norm_tokens = fused_norm_tokens[:, count:, :]
        wsi_norm_tokens_aggr = self.aggr_wsi(wsi_norm_tokens, dim=1) # B, dim

        # Concat disentangled vectors
        embedding = torch.concat([wsi_rna_norm_tokens_aggr, rna_wsi_norm_tokens_aggr, rna_norm_tokens_aggr, wsi_norm_tokens_aggr], dim=1)


        # Get risk
        logits = self.classifier(embedding)

        results = {"wsi_rna_repr": wsi_rna_norm_tokens_aggr,
                            "rna_wsi_repr": rna_wsi_norm_tokens_aggr,
                            "logits": logits,
                            "wsi_repr": wsi_norm_tokens_aggr,
                            "rna_repr": rna_norm_tokens_aggr
                            }

        if return_attn:
            results['self_attn_rna'] = self_attn_rna
            results['self_attn_wsi'] = self_attn_wsi
            results['cross_attn_rna_wsi'] = cross_attn_rna_wsi
            results['cross_attn_wsi_rna'] = cross_attn_wsi_rna

        return results

    def from_pretrained(self, cp_path):
        # Load weights from pretrained model
        state_dict = torch.load(cp_path, map_location=self.device)

        # Load the weights into the model
        self.load_state_dict(state_dict)


class DIMAFxSurvival(DIMAFxBase):
    """ DIMAFx for survival prediction, using the disentangled RNA/WSI attention fusion backbone. """

    def forward(self, wsi, rna, label, censorship, return_attn=False, return_embed=False):
        """ Main forward function"""
        # Forward pass
        output = self.forward_mm_no_loss(wsi, rna, return_attn)

        # Compute the total loss
        output_results, output_log = self.compute_loss(output, label, censorship)

        if return_attn:
            output_results['self_attn_rna'] = output['self_attn_rna']
            output_results['self_attn_wsi'] = output['self_attn_wsi']
            output_results['cross_attn_rna_wsi'] = output['cross_attn_rna_wsi']
            output_results['cross_attn_wsi_rna'] = output['cross_attn_wsi_rna']

        if return_embed:
            output_results['wsi_rna_repr'] = output['wsi_rna_repr']
            output_results['rna_wsi_repr'] = output['rna_wsi_repr']
            output_results['wsi_repr'] = output['wsi_repr']
            output_results['rna_repr'] = output['rna_repr']

        return output_results, output_log

    def compute_loss(self, output, label, censorship):
        """Compute the loss given the output of the model."""
        logits = output['logits']
        results_dict = {'logits': logits}

        if isinstance(self.loss_fn, NLLSurvLoss):
            total_loss, log_dict = self.loss_fn(logits=logits, times=label, censorships=censorship)
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1).unsqueeze(dim=1)
            results_dict.update({'wsi_repr': output['wsi_repr'],
                                 'rna_repr': output['rna_repr'],
                                 'mm_repr': torch.concat((output['rna_wsi_repr'], output['wsi_rna_repr']), dim=1)
                                 })
            results_dict.update({'hazards': hazards,
                                    'survival': survival,
                                    'risk': risk})

        elif isinstance(self.loss_fn, CoxLoss):
            total_loss, log_dict = self.loss_fn(logits=logits, times=label, censorships=censorship)
            risk = torch.exp(logits)
            results_dict['risk'] = risk
            results_dict.update({'wsi_repr': output['wsi_repr'],
                                 'rna_repr': output['rna_repr'],
                                 'mm_repr': torch.concat((output['rna_wsi_repr'], output['wsi_rna_repr']), dim=1)
                                 })

        elif isinstance(self.loss_fn, DisentangledSurvLoss):
            total_loss, log_dict = self.loss_fn(output=output, times=label, censorships=censorship)
            risk = torch.exp(logits)
            results_dict['risk'] = risk

        results_dict['loss'] = total_loss

        return results_dict, log_dict


class DIMAFxClassifier(DIMAFxBase):
    """
    DIMAFx for multimodal classification. Reuses the disentangled RNA/WSI attention
    fusion backbone of DIMAFxBase, swapping out the survival head/loss for a classification
    head/loss (no event time or censorship).
    """

    def forward(self, wsi, rna, label, return_attn=False, return_embed=False):
        """ Main forward function"""
        # Forward pass
        output = self.forward_mm_no_loss(wsi, rna, return_attn)

        # Compute the total loss
        output_results, output_log = self.compute_loss(output, label)

        if return_attn:
            output_results['self_attn_rna'] = output['self_attn_rna']
            output_results['self_attn_wsi'] = output['self_attn_wsi']
            output_results['cross_attn_rna_wsi'] = output['cross_attn_rna_wsi']
            output_results['cross_attn_wsi_rna'] = output['cross_attn_wsi_rna']

        if return_embed:
            output_results['wsi_rna_repr'] = output['wsi_rna_repr']
            output_results['rna_wsi_repr'] = output['rna_wsi_repr']
            output_results['wsi_repr'] = output['wsi_repr']
            output_results['rna_repr'] = output['rna_repr']

        return output_results, output_log

    def compute_loss(self, output, label):
        """ Compute the classification loss given the output of the model. """
        logits = output['logits']

        if not isinstance(self.loss_fn, (CEClassLoss, SVMClassLoss)):
            sys.exit("Classification loss is not implemented, aborting... ")

        total_loss, log_dict = self.loss_fn(logits=logits, labels=label)

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        results_dict = {'logits': logits,
                         'probs': probs,
                         'preds': preds,
                         'wsi_repr': output['wsi_repr'],
                         'rna_repr': output['rna_repr'],
                         'mm_repr': torch.concat((output['rna_wsi_repr'], output['wsi_rna_repr']), dim=1),
                         'loss': total_loss}

        return results_dict, log_dict
