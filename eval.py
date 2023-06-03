import torch
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt
import shutil
from sklearn.metrics import f1_score
from omegaconf import OmegaConf

from spiel_net.dataset_spiel import SpielDataset, sample2dist, dist2sample
from spiel_net.model_spiel import SpielNet
from spiel_net.model_spiel_hand import SpielNetHandcraft
from spiel_net.construct_labels import construct_labels_indexing
from spiel_net.plots import spiel_cube_plot, spiel_2d_projected_plots, cross_sectional_plot, spiel_colorbar

from pace.parallax_solver.ellipsoid_transducer import EllipsoidTransducer
from pace.parallax_solver import ParallaxGradientSolver
from pace.utils.channel_plot import channel_plot
from pace.utils.ellipsoid_plot import plot_ellipsoids


def ellipsoid_intersection(sensors, echoes):

    assert len(sensors) == len(echoes)

    elps = []
    for sensor_pos, echo in zip(sensors, echoes):
        toa_distance = sample2dist(echo, c=c, fkHz=fkHz, sample_rate=1/8)*2
        if toa_distance > 75:
            elp = EllipsoidTransducer(sen=sensor_pos, src=np.zeros(3), toa=toa_distance)
        else:
            print('No echo detected')
            return elps, [np.zeros(3)], False, np.ones(3)*np.inf
        elps.append(elp)

    # find 3 space coordinates
    solver = ParallaxGradientSolver(elps)
    p_list, valid, eps = solver.multivar_gradient_scipy()
    
    return elps, p_list, valid, eps

# load config
cfg = OmegaConf.load(str(Path('.') / 'spiel_net' / 'config.yml'))

# override config with CLI
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

net = SpielNet(device=cfg["device"], max_iter=cfg["max_iter"], echo_thresh=cfg["echo_thresh"])
if cfg["ablation_memg"]: net = SpielNetHandcraft(device=cfg["device"], max_iter=cfg["max_iter_train"], echo_thresh=cfg["echo_thresh_train"])

fkHz = 8*22
c = 343

test_dataset = SpielDataset(
    dataset_path=Path.cwd() / 'dataset' / 'test_run',
    transform=None,
    flip=False,
    bg_subtract=cfg["bg_subtract"],
    blind_zone_idx=9,
    skip_transducer_idcs=None,
    x_position_gap=80,
    y_position_gap=80,
    z_position_gap=80,
    pow_law_opt=cfg["pow_law_opt"],
    x_range = [-80, 81],
    y_range = [-80, 81],
    z_range = [100, 201],
    c = c,
    fkHz = fkHz/8,
    )

sensor_arr = test_dataset.get_sensor_coords(radius=75)

point_toa_projection = lambda p, sensors, c=c, fkHz=fkHz, sample_rate=1/8: [dist2sample(sum((p-sloc)**2)**.5+sum((p)**2)**.5, c, fkHz, sample_rate) for sloc in sensors]

test_dataloader = DataLoader(
                            test_dataset,
                            shuffle=False,
                            num_workers=2,
                            batch_size=cfg["batch_size"],
                            )

ckpt_path = Path.cwd() / 'spiel_net' / cfg["model_path"]
if cfg["ablation_memg"]: ckpt_path = ckpt_path.parent / 'spiel_net_epoch_47_wo_memg.pt'
state_dict = torch.load(ckpt_path, map_location=torch.device(cfg["device"]))
net.load_state_dict(state_dict['model_state_dict'])
net.eval()
net.train(False)

efs_path = Path('./efs')
if efs_path.exists(): shutil.rmtree(str(efs_path))
efs_path.mkdir(exist_ok=True)

valids = []
est_positions = []
gt_positions = []
true_matches = []
pred_matches = []
for i, data in enumerate(test_dataloader):

    frames, gt_pos, gt_samples, fname = data
    frames, gt_pos, gt_samples = frames.to(cfg["device"]), gt_pos.to(cfg["device"]), gt_samples.to(cfg["device"])
    gt_positions.append(gt_pos.squeeze(0).cpu())

    activations, learned_feats, memg_feats, results = net(frames.squeeze(0))

    # do evaluation on CPU
    frames = frames.squeeze().cpu()
    gt_samples = gt_samples.squeeze().cpu()
    results = results.squeeze().cpu()
    memg_feats = memg_feats.detach().cpu()
    learned_feats = learned_feats.detach().cpu()
    activations = activations.detach().cpu()

    # get ground-truth matches
    toas = memg_feats[..., 1].clone()
    toas += cfg.toa_offset if hasattr(cfg, 'toa_offset') else 0
    labels, _ = construct_labels_indexing(toas, gt_samples)
    match_true = np.where(~labels.to(dtype=bool))[-1]
    true_matches.append(match_true)

    # never pick reference from send channel
    activations[0].T[0] = torch.inf

    # pick reference component
    comp_mins, comp_args = torch.min(activations, axis=1)
    frame_idx = torch.argmin(comp_mins)
    ref_comp = learned_feats[frame_idx][comp_args[frame_idx]]
    ref_memg = memg_feats[frame_idx][comp_args[frame_idx]]

    # MLP ablation
    if cfg["ablation_mlp"]:
        amp_args = np.array([[torch.max(frame_feats[..., 0]), torch.argmax(frame_feats[..., 0])] for frame_feats in memg_feats])
        frame_idx = np.argmax(amp_args[..., 0])
        ref_memg = memg_feats[frame_idx][int(amp_args[frame_idx][1])][None, ...]

    # construct dissimilarities and indices of all other comps
    all_comp = learned_feats.reshape(-1, learned_feats.shape[-1])
    dissimilarities = torch.stack([torch.cdist(ref_comp, frame_comps, p=2)[0] for frame_comps in learned_feats])
    match_idcs = [int(torch.argmin(torch.cdist(ref_comp, frame_comps, p=2))) for frame_comps in learned_feats]

    # Contrastive ablation
    if cfg["ablation_contrastive"]:
        dissimilarities = torch.stack([1/torch.cdist(memg_frame, ref_memg[None, :]) for memg_frame in memg_feats]).squeeze()
        match_idcs = [torch.argmin(torch.cdist(memg_frame, ref_memg[None, :])[0]) for memg_frame in memg_feats]
    pred_matches.append(match_idcs)

    # select echoes (index 1 for mu_k and -2 for ToA)
    if cfg["ablation_memg"]:
        echo_tuple = np.array([frame_feats[idx][1] for frame_feats, idx in zip(memg_feats, match_idcs)])
    else:
        echo_tuple = np.array([(frame_feats[idx][1]+frame_feats[idx][-2])/2 for frame_feats, idx in zip(memg_feats, match_idcs)])
    echo_tuple += cfg.toa_offset if hasattr(cfg, 'toa_offset') else 0

    # compute intersection
    elps, p_list, valid_idx, eps = ellipsoid_intersection(sensor_arr[1:], echo_tuple[1:])
    p_star = torch.tensor(p_list[-1])
    est_positions.append(p_star)
    valids.append(valid_idx)

    # results
    rmse = ((est_positions[-1]-gt_positions[-1])**2).sum()**.5
    rmse_perc = rmse/((gt_positions[-1])**2).sum(0)**.5 * 100
    if valid_idx:
        result = [str(np.round(r.numpy(), 1)) for r in [est_positions[-1], gt_positions[-1], rmse, rmse_perc]]
        print('estimated vs ground-truth position: %s vs %s @ %s mm and %s perc. RMSE' % (result[0], result[1], result[2], result[3]))
    else:
        print('estimated vs ground-truth position: no solution')
    
    print('F1-score: %s' % round(f1_score(match_true, match_idcs, average='macro'), 4))

    if valid_idx and cfg["plot_opt"]:
        # plot channels
        title = ' '.join([str(el)+'mm' for el in gt_positions[-1].numpy()])
        fig = channel_plot(frames.numpy(), results.numpy(), memg_feats.numpy(), match_idcs, gt_samples, figsize=(10, 8), show_opt=cfg["plot_opt"], title=None)
        fig.savefig(efs_path / (fname[0].split('_frames.csv')[0]+'_efs.pdf'))
        plt.close(fig)
        fig, ax = plot_ellipsoids(elps, pts=p_list, obj_pos=gt_positions[-1], save_opt=False, gray_opt=False, legend_opt=True, hide_axes_opt=True, show_opt=False)
        plt.close(fig)

# get indices for position re-ordering
_, idcs = np.unique(torch.vstack(gt_positions).numpy(), axis=0, return_index=True)

# prepare results for plotting
valid_idx = np.array(valids)[idcs]
est_xyz =  torch.vstack(est_positions)[idcs].T
gt_xyz = torch.vstack(gt_positions)[idcs].T
est_xyz[:, ~valid_idx] = float('NaN')
rmse = ((est_xyz - gt_xyz)**2).sum(0)**.5
rmse_perc = rmse/((gt_xyz)**2).sum(0)**.5 * 100
mae = (abs(est_xyz - gt_xyz)).mean(0)
mae_perc = mae/abs(gt_xyz).mean(0) * 100
norm_rmse = rmse/np.nanmax(rmse)

lines = np.hstack([np.round(gt_xyz.numpy().T, 0).astype('int'), np.round(est_xyz.numpy().T, 1), np.round(rmse[:, None], 1), np.round(rmse_perc[:, None], 1)])
print(" \\\\\n".join([" & ".join(map(str, line)) for line in lines]))

print('Mean RMSE: %s' % np.nanmean(rmse))
print('Mean Percentage RMSE: %s' % np.nanmean(rmse_perc))
print('Std. RMSE: %s' % np.std(rmse[~rmse.isnan()].numpy()))
print('Std. Percentage RMSE: %s' % np.std(rmse_perc[~rmse_perc.isnan()].numpy()))
print('Accuracy: %s' % round(np.sum(valids)/len(valids), 4))
print('F1-score: %s' % round(f1_score(np.hstack(true_matches), np.hstack(pred_matches), average='macro'), 4))

if cfg["plot_opt"]:
    fig, axs = plt.subplots(1, 2, figsize=(2.5, 2.5))
    ax, est_scatter_color = spiel_cube_plot(est_xyz.numpy(), gt_xyz.numpy(), c=rmse.numpy(), hide_axes_opt=False, show_opt=False)
    plt.savefig('./spiel_results_a.eps')

    spiel_colorbar(est_scatter_color)
    plt.savefig('./spiel_results_e.eps')

    subcaption_letters = ['a', 'b', 'c', 'd']
    for i in range(3):
        fig, ax = plt.subplots(1, 1, figsize=(2.5, 2.5))
        cross_sectional_plot(ax, i, est_xyz.numpy(), gt_xyz.numpy(), valid=valid_idx, c=rmse.numpy(), circle_radius=np.nanmean(rmse))
        plt.savefig('./spiel_results_'+subcaption_letters[i+1]+'.eps')

np.savetxt('./results_rmse.csv', np.hstack([torch.vstack(gt_positions).numpy(), torch.vstack(est_positions).numpy(), valid_idx[:, None]]))
