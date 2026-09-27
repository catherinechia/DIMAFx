import torch.nn as nn


class CEClassLoss(nn.Module):
    """
    Cross-entropy loss for multimodal classification.
    """
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.loss_fn = nn.CrossEntropyLoss()

    def get_num_classes(self):
        return self.num_classes

    def __call__(self, logits, labels):
        loss = self.loss_fn(logits, labels)
        return loss, {'loss': loss.item()}


class SVMClassLoss(nn.Module):
    """
    Smooth top-1 SVM loss for multimodal classification.
    Requires the `topk` package (https://github.com/oval-group/smooth-topk),
    as used in https://github.com/mahmoodlab/CLAM.
    """
    def __init__(self, num_classes, device):
        super().__init__()
        from topk.svm import SmoothTop1SVM

        self.num_classes = num_classes
        self.loss_fn = SmoothTop1SVM(n_classes=num_classes).to(device)

    def get_num_classes(self):
        return self.num_classes

    def __call__(self, logits, labels):
        loss = self.loss_fn(logits, labels)
        return loss, {'loss': loss.item()}
