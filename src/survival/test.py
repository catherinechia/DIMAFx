import os
import torch
import numpy as np



from .losses import NLLSurvLoss, CoxLoss, DisentangledSurvLoss
from .metrics import compute_disentanglement, compute_survival_metrics
from utils.train_utils import list_to_device, LoggingMeter
from utils.general_utils import save_json, save_pkl
from embeddings.embeddings import prepare_embeddings
from models.DIMAFx import DIMAFxSurvival

def test_survival_model(model, test_dl, device, survival_info_train=None, return_attn=False, result_dir=None, mode='post_training'):
    """ Test a survival prediction model for a single fold. """
    model.eval()

    all_case_ids, all_slide_ids = [], []
    all_risk_scores, all_censorships, all_event_times = [], [], []
    all_rna_attn, all_rna_wsi_attn, all_wsi_rna_attn, all_wsi_attn = [], [], [], []
    all_rna_emb, all_rna_wsi_emb, all_wsi_rna_emb, all_wsi_emb = [], [], [], []
    test_log = {}

    # Loop over data
    with torch.no_grad():
        for idx, batch in enumerate(test_dl):
            # Get the data and labels
            wsi = batch['img'].to(device)
            rna = list_to_device(batch['rna'], device)
            label = batch['label'].to(device)
            event_time = batch['survival_time'].to(device)
            censorship = batch['censorship'].to(device)

            # forward pass
            out, log_dict = model(wsi, rna, label=label, censorship=censorship, return_attn=return_attn, return_embed=True)
            all_case_ids.append(np.array(batch['case_id']))
            all_slide_ids.append(np.array(batch['slide_id']))
            
            # Logging!
            if return_attn:
                if len(out['self_attn_rna'].shape) == 2:
                    out['self_attn_rna'] = out['self_attn_rna'].unsqueeze(0)
                    out['cross_attn_rna_wsi'] = out['cross_attn_rna_wsi'].unsqueeze(0)
                    out['cross_attn_wsi_rna'] = out['cross_attn_wsi_rna'].unsqueeze(0)
                    out['self_attn_wsi'] = out['self_attn_wsi'].unsqueeze(0)
                all_rna_attn.append(out['self_attn_rna'].detach().cpu().numpy())
                all_rna_wsi_attn.append(out['cross_attn_rna_wsi'].detach().cpu().numpy())
                all_wsi_rna_attn.append(out['cross_attn_wsi_rna'].detach().cpu().numpy())
                all_wsi_attn.append(out['self_attn_wsi'].detach().cpu().numpy())
                
            # Logging!
            
            all_rna_emb.append(out['rna_repr'].detach())
            all_rna_wsi_emb.append(out['rna_wsi_repr'].detach())
            all_wsi_rna_emb.append(out['wsi_rna_repr'].detach())
            all_wsi_emb.append(out['wsi_repr'].detach())

            for key, val in log_dict.items():
                if key not in test_log:
                    test_log[key] = LoggingMeter(key)
                test_log[key].update(val, n=len(wsi))
            
            all_risk_scores.append(out['risk'].detach().cpu().numpy())
            all_censorships.append(censorship.cpu().numpy())
            all_event_times.append(event_time.cpu().numpy())
    
        all_risk_scores = np.concatenate(all_risk_scores).squeeze(1)
        all_censorships = np.concatenate(all_censorships).squeeze(1)
        all_event_times = np.concatenate(all_event_times).squeeze(1)

        all_rna_emb = torch.cat(all_rna_emb, dim=0)
        all_rna_wsi_emb = torch.cat(all_rna_wsi_emb, dim=0)
        all_wsi_rna_emb = torch.cat(all_wsi_rna_emb, dim=0)
        all_wsi_emb = torch.cat(all_wsi_emb, dim=0)
        
        # Compute disentanglement metrics
        dcor_dict = compute_disentanglement(all_rna_emb, all_wsi_emb, all_wsi_rna_emb, all_rna_wsi_emb, type='dcor')
        orth_dict = compute_disentanglement(all_rna_emb, all_wsi_emb, all_wsi_rna_emb, all_rna_wsi_emb, type='orth')

        # Compute c-index
        c_index, c_index_ipcw = compute_survival_metrics(all_censorships, all_event_times, all_risk_scores, survival_info_train)
        
        results = {item: meter.avg for item, meter in test_log.items()}
        results.update({'c_index': c_index})
        results.update({'c_index_ipcw': c_index_ipcw})
        results.update(dcor_dict)
        results.update(orth_dict)

        # Save the predicted risk scores
        if mode == 'post_training':
            risk_scores_dict = {'case_ids': np.concatenate(all_case_ids, axis=0),
                                'slide_ids': np.concatenate(all_slide_ids, axis=0),
                                'Risk scores': all_risk_scores, 
                                'Censorship': all_censorships,
                                'Time': all_event_times}
            save_pkl(os.path.join(result_dir, mode), f"predicted_risk_scores_test.pkl", risk_scores_dict)

        # Logging!
        if return_attn:
            assert result_dir is not None, "Result dir is not specified, please do so."
            attention_data = {'Risk scores': all_risk_scores,
                              "self_attn_rna": np.concatenate(all_rna_attn, axis=0),
                        "cross_attn_rna_wsi": np.concatenate(all_rna_wsi_attn, axis=0),
                        "cross_attn_wsi_rna": np.concatenate(all_wsi_rna_attn, axis=0),
                        "self_attn_wsi": np.concatenate(all_wsi_attn, axis=0), 
                        "case_ids": np.concatenate(all_case_ids, axis=0)}
            attn_dis_dir = os.path.join(os.path.join(result_dir, mode), 'attention')
            os.makedirs(attn_dis_dir, exist_ok=True)
            save_pkl(attn_dis_dir, f"learned_attention_matrices_test.pkl", attention_data)
    
    return results

def survival_test(args, test_dl, fold, survival_info_train):
    """ Load and test survival prediction model. """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir_fold =  os.path.join(args.result_dir, f"Fold_{fold}/")
    pretrained_model_path = os.path.join(results_dir_fold, "model_checkpoint.pth")

    # Initialize loss function
    if args.loss_fn == 'nll':
        loss_fn = NLLSurvLoss(alpha=args.nll_alpha)
        num_classes = args.n_label_bins
    elif args.loss_fn == 'cox':
        loss_fn = CoxLoss()
        num_classes = 1
    else:
        loss_fn_split = args.loss_fn.split("_")
        loss_fn = DisentangledSurvLoss(loss_fn_split[0], loss_fn_split[1], weight_surv=args.w_surv, weight_disentanglement=args.w_dis, n_label_bins=args.n_label_bins, alpha=args.nll_alpha)
        num_classes = loss_fn.get_num_classes()

    # Create unimodal representations
    test_dl, data_info  = prepare_embeddings(args, 'test', test_dl)

    # Load model
    model = DIMAFxSurvival(rna_dims=data_info['Pathway sizes'],
                       histo_dim=data_info['Dim wsi'],
                       device=device,
                       single_out_dim=256,
                       num_classes=num_classes,
                       loss_fn=loss_fn,
                       aggr_post_embed=args.aggr_post_embed,
                       wsi_representation_type=args.wsi_repr,
                       num_proto_wsi=args.n_proto
                       )
    model.to(device)
    model.from_pretrained(pretrained_model_path)

    # Test the model
    results = test_survival_model(model, test_dl, device, survival_info_train, return_attn=args.return_attn, result_dir=results_dir_fold)
    save_json(results_dir_fold, f"test_summary.json", results)
    return results