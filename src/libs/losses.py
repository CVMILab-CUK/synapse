import torch
import torch.nn as nn
from libs.metric import SignalDice, SoftSignalDice

def l1(x, y):
    return torch.abs(x-y)


def l2(x, y):
    return torch.pow((x-y), 2)

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params?
        num: int?the number of loss
        x: multi-task loss
    Examples?
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = nn.Parameter(params)
       
    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum
        
class SignalDiceLoss(nn.Module):

    def __init__(self, sep=True,  eps=1e-6, soft=True, alpha=100):
        super(SignalDiceLoss, self).__init__()
        self.eps  = eps
        self.sep  = sep
        if soft:
            self.sdsc = SoftSignalDice(eps, sep, alpha=alpha)
        else:
            self.sdsc = SignalDice(eps, sep)
    
    def forward(self, inputs, targets):
        sdsc_value = self.sdsc(inputs, targets)

        if self.sep:
            return torch.mean(1 - torch.mean((2*torch.sum(self.sdsc.intersection, dim=2)) / (torch.sum(self.sdsc.union, dim=2) + self.eps)))
        else:
            return 1 - sdsc_value

class ContrastiveLoss(nn.Module):
    def __init__(self, logit_scale_init_value = 2.6592):
        super(ContrastiveLoss, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()
        self.logit_scale = nn.Parameter(torch.tensor(logit_scale_init_value))

    def clip_loss(self, similarity:torch.Tensor)->torch.Tensor:
        caption_loss = self.contrastive_loss(similarity)
        image_loss   = self.contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0
    
    def contrastive_loss(self, logits: torch.Tensor) -> torch.Tensor:
        return self.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

    def forward(self, eeg_embeds:torch.Tensor, image_embeds:torch.Tensor, eps:float=1e-8) -> torch.Tensor:
        image_embeds = image_embeds / (image_embeds.norm(p=2, dim=-1, keepdim=True) + eps)
        eeg_embeds   = eeg_embeds   / (eeg_embeds.norm(p=2, dim=-1, keepdim=True) + eps)

        logit_scale = self.logit_scale.exp()
        logits_per_eeg   = torch.matmul(eeg_embeds, image_embeds.t()) * logit_scale.to(eeg_embeds.device)
        logits_per_image = logits_per_eeg.t()

        return self.clip_loss(logits_per_eeg)


