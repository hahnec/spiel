import torch

class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, reference, components, labels, activation_opt=False):

        euclidean_distance = torch.cdist(reference, components, p=2).flatten()
        euclidean_distance = torch.sigmoid(euclidean_distance) if activation_opt else euclidean_distance
        loss_contrastive = torch.mean((1-labels) * torch.pow(euclidean_distance, 2) + (labels) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))

        return loss_contrastive
