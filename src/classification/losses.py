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
        self._built_device = None
        
    def get_num_classes(self):
        return self.num_classes

    # def __call__(self, logits, labels):
    #     loss = self.loss_fn(logits, labels)
    #     return loss, {'loss': loss.item()}

    # def __call__(self, logits, labels):
    #     # topk stores its class indices as a plain attribute (not a buffer), so .to(device) does not move it; keep it on the same device as the logits.
    #     if self.loss_fn.labels.device != logits.device:
    #         self.loss_fn.labels = self.loss_fn.labels.to(logits.device)
    #         self.loss_fn.get_losses()
    #     loss = self.loss_fn(logits, labels)
    #     return loss, {'loss': loss.item()}
    
    def __call__(self, logits, labels):
        # topk's loss functions (F, F_s) capture the class-index tensor in a
        # closure when built, so moving the module (or its labels) afterwards
        # leaves them on the old device; rebuild them for the logits' device
        # (mirrors topk's own .cuda()).
        if self._built_device != logits.device:
            self.loss_fn.labels = self.loss_fn.labels.to(logits.device)
            self.loss_fn.get_losses()
            self._built_device = logits.device
        loss = self.loss_fn(logits, labels)
        return loss, {'loss': loss.item()}
