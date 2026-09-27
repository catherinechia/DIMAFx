import os
import torch
import numpy as np

from .metrics import compute_classification_metrics
from utils.train_utils import list_to_device, LoggingMeter
from utils.general_utils import save_pkl


def test_classification_model(model, test_dl, device, return_attn=False, result_dir=None, mode='post_training'):
    """ Test a classification model for a single fold. """
    model.eval()

    all_case_ids, all_slide_ids = [], []
    all_labels, all_preds, all_probs = [], [], []
    all_rna_attn, all_rna_wsi_attn, all_wsi_rna_attn, all_wsi_attn = [], [], [], []
    test_log = {}

    # Loop over data
    with torch.no_grad():
        for idx, batch in enumerate(test_dl):
            # Get the data and labels
            wsi = batch['img'].to(device)
            rna = list_to_device(batch['rna'], device)
            label = batch['label'].to(device)

            # forward pass
            out, log_dict = model(wsi, rna, label=label, return_attn=return_attn, return_embed=True)
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

            for key, val in log_dict.items():
                if key not in test_log:
                    test_log[key] = LoggingMeter(key)
                test_log[key].update(val, n=len(wsi))

            all_labels.append(label.detach().cpu().numpy())
            all_preds.append(out['preds'].detach().cpu().numpy())
            all_probs.append(out['probs'].detach().cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_preds = np.concatenate(all_preds)
        all_probs = np.concatenate(all_probs)

        # Compute classification metrics
        metrics = compute_classification_metrics(all_labels, all_preds, all_probs, model.num_classes)

        results = {item: meter.avg for item, meter in test_log.items()}
        results.update(metrics)

        # Save the predictions
        if mode == 'post_training':
            preds_dict = {'case_ids': np.concatenate(all_case_ids, axis=0),
                          'slide_ids': np.concatenate(all_slide_ids, axis=0),
                          'Labels': all_labels,
                          'Predictions': all_preds,
                          'Probabilities': all_probs}
            save_pkl(os.path.join(result_dir, mode), f"predictions_test.pkl", preds_dict)

        # Logging!
        if return_attn:
            assert result_dir is not None, "Result dir is not specified, please do so."
            attention_data = {'Predictions': all_preds,
                              "self_attn_rna": np.concatenate(all_rna_attn, axis=0),
                        "cross_attn_rna_wsi": np.concatenate(all_rna_wsi_attn, axis=0),
                        "cross_attn_wsi_rna": np.concatenate(all_wsi_rna_attn, axis=0),
                        "self_attn_wsi": np.concatenate(all_wsi_attn, axis=0),
                        "case_ids": np.concatenate(all_case_ids, axis=0)}
            attn_dis_dir = os.path.join(os.path.join(result_dir, mode), 'attention')
            os.makedirs(attn_dis_dir, exist_ok=True)
            save_pkl(attn_dis_dir, f"learned_attention_matrices_test.pkl", attention_data)

    return results
