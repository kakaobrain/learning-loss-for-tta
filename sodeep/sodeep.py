"""
****************** COPYRIGHT AND CONFIDENTIALITY INFORMATION ******************
Copyright (c) 2019 [Thomson Licensing]
All Rights Reserved
This program contains proprietary information which is a trade secret/business \
secret of [Thomson Licensing] and is protected, even if unpublished, under \
applicable Copyright laws (including French droit d'auteur) and/or may be \
subject to one or more patent(s).
Recipient is to retain this program in confidence and is not permitted to use \
or make copies thereof other than as permitted in a written agreement with \
[Thomson Licensing] unless otherwise expressly allowed by applicable laws or \
by [Thomson Licensing] under express agreement.
Thomson Licensing is a company of the group TECHNICOLOR
*******************************************************************************
This scripts permits one to reproduce training and experiments of:
    Engilberge, M., Chevallier, L., Pérez, P., & Cord, M. (2019, June).
    SoDeep: A Sorting Deep Net to Learn Ranking Loss Surrogates.
    In Proceedings of CVPR

Author: Martin Engilberge
"""
import numpy as np
import torch

from commons import encoded_tta_default
from sodeep.model import model_loader
from sodeep.utils import get_rank


def load_sorter(checkpoint_path):
    sorter_checkpoint = torch.load(checkpoint_path)

    model_type = sorter_checkpoint["args_dict"].model_type
    seq_len = sorter_checkpoint["args_dict"].seq_len
    state_dict = sorter_checkpoint["state_dict"]

    return model_type, seq_len, state_dict


class RankHardLoss(torch.nn.Module):
    """ Loss function  inspired by hard negative triplet loss, directly applied in the rank domain """
    def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None, margin=0.2, nmax=1):
        super(RankHardLoss, self).__init__()
        self.nmax = nmax
        self.margin = margin

        self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

    def hc_loss(self, scores):
        rank = self.sorter(scores)

        diag = rank.diag()

        rank = rank + torch.diag(torch.ones(rank.diag().size(), device=rank.device) * 50.0)

        sorted_rank, _ = torch.sort(rank, 1, descending=False)

        hard_neg_rank = sorted_rank[:, :self.nmax]

        loss = torch.sum(torch.clamp(-hard_neg_rank + (1.0 / (scores.size(1)) + diag).view(-1, 1).expand_as(hard_neg_rank), min=0))

        return loss

    def forward(self, scores):
        """ Expect a score matrix with scores of the positive pairs are on the diagonal """
        caption_loss = self.hc_loss(scores)
        image_loss = self.hc_loss(scores.t())

        image_caption_loss = caption_loss + image_loss

        return image_caption_loss


class RankLoss(torch.nn.Module):
    """ Loss function  inspired by recall """
    def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None,):
        super(RankLoss, self).__init__()
        self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

    def forward(self, scores):
        """ Expect a score matrix with scores of the positive pairs are on the diagonal """
        caption_rank = self.sorter(scores)
        image_rank = self.sorter(scores.t())

        image_caption_loss = torch.sum(caption_rank.diag()) + torch.sum(image_rank.diag())

        return image_caption_loss


class MapRankingLoss(torch.nn.Module):
    """ Loss function  inspired by mean Average Precision """
    def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None):
        super(MapRankingLoss, self).__init__()

        self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

    def forward(self, output, target):
        # Compute map for each classes
        map_tot = 0
        for c in range(target.size(1)):
            gt_c = target[:, c]

            if torch.sum(gt_c) == 0:
                continue
            rank_pred = self.sorter(output[:, c].unsqueeze(0)).view(-1)
            rank_pos = rank_pred * gt_c

            map_tot += torch.sum(rank_pos)

        return map_tot


class SpearmanLoss(torch.nn.Module):
    """ Loss function  inspired by spearmann correlation.self
    Required the trained model to have a good initlization.

    Set lbd to 1 for a few epoch to help with the initialization.
    """
    def __init__(self, sorter_type, seq_len=None, sorter_state_dict=None, lbd=0, t_easy=0.05):
        super(SpearmanLoss, self).__init__()
        self.sorter = model_loader(sorter_type, seq_len, sorter_state_dict)

        self.criterion_mse = torch.nn.MSELoss(reduction='none')
        self.criterionl1 = torch.nn.L1Loss()

        self.lbd = lbd
        self.t_easy = t_easy

    def forward(self, pred, gt, masked=False):
        rank_gt = get_rank(gt, dim=1)
        rank_pred = self.sorter(pred)

        if masked:
            mask = (pred >= pred[:, encoded_tta_default()].unsqueeze(-1)).float()

            _pred = pred * mask + (pred * (1 - mask)).detach()
            assert torch.all(torch.eq(pred, _pred))
            pred = _pred
            rank_pred = rank_pred * mask + (rank_pred * (1 - mask)).detach()

        losses = self.criterion_mse(rank_pred, rank_gt)
        mask_easy = (losses > self.t_easy).float()
        losses = losses * mask_easy
        loss_mean = torch.mean(losses)

        return loss_mean + self.lbd * self.criterionl1(pred, gt)
