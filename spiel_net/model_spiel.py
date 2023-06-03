import torch

from multimodal_emg.batch_staged_memgo import batch_staged_memgo
from multimodal_emg import emg_envelope_model

inv_relative_l2_norm = lambda f, g: 1 / ((f/g.max() - g/g.max())**2).sum(-1)**.5 if g.max() != 0 else 0


class SpielNet(torch.torch.nn.Module):

    def __init__(self, device=None, max_iter=None, echo_thresh=None):
        super(SpielNet, self).__init__()

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

        # MEMG feature extraction
        memg_feats, results, conf, echo_list = batch_staged_memgo(x, t, max_iter_per_stage=self.max_iter, echo_threshold=self.echo_thresh, grad_step=1, fs=1, oscil_opt=False)

        # prepare hand-crafted features
        memg_feats = torch.dstack([memg_feats, torch.ones([memg_feats.shape[0], memg_feats.shape[1], 4], device=x.device)])
        stacked_memg_feats = memg_feats[..., :4].reshape(-1, 4).T[..., None]
        stacked_comps = emg_envelope_model(*stacked_memg_feats, x=t[None, :], exp_fun=torch.exp, erf_fun=torch.erf)

        # component confidence
        memg_feats[..., 4] = conf
        memg_feats[..., 4][memg_feats[..., 4]>2**32-1] = 2**32-1    # truncate confidence for numerical MLP stability

        # power
        memg_feats[..., 5] = torch.nansum(stacked_comps, axis=-1).reshape(memg_feats.shape[:2])

        # ToA
        memg_feats[..., 6] = echo_list[..., 1] if echo_list.numel() > 0 else memg_feats[..., 6]

        # frame confidence
        memg_feats[..., 7] = (1 / torch.abs(x-results.squeeze(1)).sum(-1))[:, None].repeat([1, memg_feats.shape[1]]) if len(results.shape) == 3 else memg_feats[..., 7]

        # MLP
        learned_feats = self.mlp(memg_feats)
        activation = self.activation_layer(learned_feats)

        return activation, learned_feats, memg_feats, results
