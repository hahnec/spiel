import torch

from multimodal_emg.batch_staged_memgo import batch_staged_memgo
from multimodal_emg import emg_envelope_model

inv_relative_l2_norm = lambda f, g: 1 / ((f/g.max() - g/g.max())**2).sum(-1)**.5 if g.max() != 0 else 0


class SpielNetHandcraft(torch.torch.nn.Module):

    def __init__(self, device=None, max_iter=None, echo_thresh=None):
        super(SpielNetHandcraft, self).__init__()

        self.device = 'cpu' if device is None else device
        self.max_iter = 20 if max_iter is None else max_iter
        self.echo_thresh = 20 if echo_thresh is None else echo_thresh

        # MLP
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(8, 32),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(32, 32),
            torch.nn.ReLU(inplace=True),

            torch.nn.Linear(32, 4),
            )
        
        self.activation_layer = torch.nn.Sequential(
            torch.nn.Linear(4, 1),
            torch.nn.Sigmoid(),
            )

        self.to(self.device)
        self.double()

    def forward(self, x):

        # sample position vector
        t = torch.arange(0, len(x[0, ...]), device=x.device, dtype=x.dtype)

        # feature extraction
        _, _, _, echo_list = batch_staged_memgo(x, t, max_iter_per_stage=self.max_iter, echo_threshold=self.echo_thresh, grad_step=1, fs=1, oscil_opt=False)

        # prepare hand-crafted features
        hand_feats = torch.zeros((echo_list.shape[0], echo_list.shape[1], 8), device=echo_list.device, dtype=echo_list.dtype)

        # amplitude
        hand_feats[..., 0] = echo_list[..., 2]

        # mu (position)
        hand_feats[..., 1] = echo_list[..., 1]

        # half width
        hand_feats[..., 2] = (echo_list[..., 1]-echo_list[..., 0])/2.5

        # power
        hand_feats[..., 3] = (hand_feats[..., 2]*hand_feats[..., 0])

        # repeat features for consistent dimensions
        hand_feats[..., 4:] = hand_feats[..., :4]

        # MLP
        learned_feats = self.mlp(hand_feats)
        activation = self.activation_layer(learned_feats)

        return activation, learned_feats, hand_feats, torch.zeros_like(x)
