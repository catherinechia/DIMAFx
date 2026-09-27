import os
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from .losses import CEClassLoss, SVMClassLoss
from .metrics import compute_classification_metrics
from .test import test_classification_model
from embeddings.embeddings import prepare_embeddings
from models.DIMAFx import DIMAFxClassifier
from utils.general_utils import save_json
from utils.train_utils import get_optim, get_lr_scheduler, list_to_device, LoggingMeter, log_results


def classification_train(args, fold, train_dl, test_dl=None):
    """ Train a classification model for a single fold. """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up results and log dir.
    result_dir_fold = os.path.join(args.result_dir, f"Fold_{fold}")
    log_dir_fold = os.path.join(args.log_dir, f"Fold_{fold}")
    results = {}

    os.makedirs(result_dir_fold, exist_ok=True)
    os.makedirs(log_dir_fold, exist_ok=True)

    writer = SummaryWriter(log_dir=log_dir_fold)

    # Initialize loss function
    if args.loss_fn == 'ce':
        loss_fn = CEClassLoss(num_classes=args.n_classes)
    elif args.loss_fn == 'svm':
        loss_fn = SVMClassLoss(num_classes=args.n_classes, device=device)
    num_classes = args.n_classes

    print('\nCreate unimodal representations...', end=' ')
    train_dl, data_info = prepare_embeddings(args, 'train', train_dl)

    if not test_dl == None:
        test_dl, _ = prepare_embeddings(args, 'test', test_dl)

    print('\nInit Model...', end=' ')
    model = DIMAFxClassifier(rna_dims=data_info['Pathway sizes'],
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

    print('\nInit optimizer ...')
    optimizer = get_optim(model=model, args=args)
    lr_scheduler = get_lr_scheduler(args, optimizer, len(train_dl))

    #####################
    # The training loop #
    #####################
    # Logging
    if not test_dl == None:
        init_results = test_classification_model(model, test_dl, device, return_attn=True, result_dir=result_dir_fold, mode='pre_training')
        log_results(writer, init_results, -1, mode='test')

    for epoch in range(args.max_epochs):
        # Train
        print('#' * 10, f'TRAIN Epoch: {epoch}', '#' * 10)
        train_results = train_loop(model, train_dl, optimizer, lr_scheduler, device)
        log_results(writer, train_results, epoch, mode='train')

        # Logging
        if not test_dl == None:
            int_results = test_classification_model(model, test_dl, device, mode='during_training')
            log_results(writer, int_results, epoch, mode='test')

    # Save last model
    torch.save(model.state_dict(), os.path.join(result_dir_fold, "model_checkpoint.pth"))

    # End of epoch: Save the last train and test results
    print(f'End of training. Evaluating on Split {fold}...:')
    if not test_dl == None:
        results = test_classification_model(model, test_dl, device, return_attn=True, result_dir=result_dir_fold)
        save_json(result_dir_fold, f"train_test_summary.json", results)

    writer.close()
    return results


def train_loop(model, dataloader, optimizer, lr_scheduler, device):
    """
        Train loop for classification
    """
    model.train()
    train_log = {}
    all_labels, all_preds, all_probs = [], [], []

    # Loop over all data samples
    for idx, batch in enumerate(dataloader):
        # Get the data and labels
        wsi = batch['img'].to(device)
        rna = list_to_device(batch['rna'], device)
        label = batch['label'].to(device)

        # Forward pass
        output_results, log_dict = model(wsi, rna, label=label)

        # Backward pass
        loss = output_results['loss']
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # For logging purposes
        for key, val in log_dict.items():
            if key not in train_log:
                train_log[key] = LoggingMeter(key)
            train_log[key].update(val, n=len(wsi))

        all_labels.append(label.detach().cpu().numpy())
        all_preds.append(output_results['preds'].detach().cpu().numpy())
        all_probs.append(output_results['probs'].detach().cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_preds = np.concatenate(all_preds)
    all_probs = np.concatenate(all_probs)

    # Compute classification metrics
    metrics = compute_classification_metrics(all_labels, all_preds, all_probs, model.num_classes)

    results = {item: meter.avg for item, meter in train_log.items()}
    results.update(metrics)
    results['lr'] = optimizer.param_groups[0]['lr']
    return results
