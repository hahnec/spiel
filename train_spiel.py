from pathlib import Path
import torch
from torch.utils.data import DataLoader, random_split
from torch import optim
import wandb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from omegaconf import OmegaConf

from spiel_net.dataset_spiel import SpielDataset
from spiel_net.model_spiel import SpielNet
from spiel_net.model_spiel_hand import SpielNetHandcraft
from spiel_net.contrastive_loss import ContrastiveLoss
from spiel_net.early_stop import EarlyStopping
from spiel_net.construct_labels import construct_labels_indexing
from pace.utils.channel_plot import channel_plot


def save_model(path: Path, epoch: int = 0):
    torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                }, path / 'spiel_net' / 'ckpts' / Path('spiel_net_epoch_'+str(epoch)+'.pt'))


def wrapped_channel_plot(frames, results, memg_feats, match_idcs, gt_pos, gt_samples, log_key="img_chart"):

        title = ' '.join([str(float(pos)) for pos in gt_pos[0]])
        vars_ch_plt = [v.squeeze().cpu().numpy() for v in [frames, results, memg_feats, match_idcs, gt_samples]]
        vars_ch_plt[2] = vars_ch_plt[2][:, None, :] if len(vars_ch_plt[2].shape) == 2 else vars_ch_plt[2]
        fig = channel_plot(*vars_ch_plt, figsize=(12, 3), title=title)
        fig.canvas.draw()
        width, height = [int(round(hw)) for hw in fig.get_size_inches() * fig.get_dpi()]
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(height, width, 3)
        wandb.log({log_key: wandb.Image(img)})
        plt.close(fig)

# load config
cfg = OmegaConf.load(str(Path('.') / 'spiel_net' / 'config.yml'))

# override config with CLI
cfg = OmegaConf.merge(cfg, OmegaConf.from_cli())

fkHz = 8*22
c = 343

train_dataset = SpielDataset(
    dataset_path=cfg["fpath"],
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

train_val_ratio = 0.7
train_num = int(len(train_dataset)*train_val_ratio)
data_train, data_val = random_split(train_dataset, (train_num, len(train_dataset)-train_num), generator=torch.Generator().manual_seed(42))

train_dataloader = DataLoader(
                            data_train,
                            shuffle=True,
                            num_workers=4,
                            batch_size=cfg["batch_size"],
                            )

val_dataloader = DataLoader(
                            data_val,
                            shuffle=True,
                            num_workers=4,
                            batch_size=cfg["batch_size"],
                            )

net = SpielNet(device=cfg["device"], max_iter=cfg["max_iter_train"], echo_thresh=cfg["echo_thresh_train"])
if cfg["ablation_memg"]: net = SpielNetHandcraft(device=cfg["device"], max_iter=cfg["max_iter_train"], echo_thresh=cfg["echo_thresh_train"])
bce_criterion = torch.nn.BCELoss()
constrastive_criterion = ContrastiveLoss()
optimizer = optim.Adam(net.parameters(), lr=cfg["lr"])
early_stopping = EarlyStopping(tolerance=15, min_delta=0.01)
conf_threshold = 1e-5

if cfg.logging:
    experiment = wandb.init(project="icra23", config=cfg, name='SPiEL')
    experiment.config.update(cfg)
    wandb.watch(models=net, criterion=constrastive_criterion,log="parameters", log_freq=1)


for epoch in range(0, cfg["epochs"]):

    # train epoch
    net.train(True)
    running_loss = 0.0
    for i, data in enumerate(train_dataloader):
        frames, gt_pos, gt_samples, fname = data
        frames, gt_pos, gt_samples = frames.to(cfg["device"]), gt_pos.to(cfg["device"]), gt_samples.to(cfg["device"])
        optimizer.zero_grad()
        
        activations, learned_feats, memg_feats, results = net(frames.squeeze(0))

        # prevent MLP training with inputs causing NaNs
        if torch.any(learned_feats.isnan()):
            continue

        # only use components from frames with high confidence (avoid corrupted features to influence train loss)
        if not torch.any(memg_feats[..., -1] > conf_threshold):
            continue

        # tbd: extent to cross-frame similarities? e.g. from 1xC components to NxC components
        toas = memg_feats[..., 1].clone()
        toas += cfg.toa_offset if hasattr(cfg, 'toa_offset') else 0
        labels, comp_idcs = construct_labels_indexing(toas, gt_samples[0])
        frame_idx = torch.argmax(memg_feats[~labels.bool()][:, 4])  # decide based on strongest confidence among labeled comps
        ref_comp = learned_feats[frame_idx][~labels[frame_idx].bool()][None, ...]

        alt_labels, all_comp = list(zip(*[[l, f] for l, f, m in zip(labels, learned_feats, memg_feats) if m[0, -1] > conf_threshold]))

        loss_bce = cfg["bce_weight"] * bce_criterion(activations.flatten(), labels.flatten())
        loss_contrastive = constrastive_criterion(ref_comp, torch.vstack(all_comp), torch.hstack(alt_labels))
        loss_sum = loss_bce + loss_contrastive
        loss_sum.backward()
        optimizer.step()

        if cfg.logging:
            wandb.log({"train_loss_sum": loss_sum})
            wandb.log({"train_loss_contrastive": loss_contrastive})
            wandb.log({"train_loss_bce": loss_bce})
        running_loss += loss_sum.item()

        if cfg["logging"] and cfg["plot_opt"] and i%10 == 1:
            try:
                wrapped_channel_plot(frames, results, memg_feats, comp_idcs, gt_pos, gt_samples, log_key="train_img_chart")
            except TypeError as e:
                print(e)

    avg_loss = running_loss / (i + 1)
    if cfg.logging: wandb.log({"train_loss_avg_epoch": avg_loss})

    # validation epoch
    net.train(False)
    running_vloss = 0.0
    running_vf1 = 0.0
    plot_list = []
    for i, vdata in enumerate(val_dataloader):
        frames, gt_pos, gt_samples, fname = data
        frames, gt_pos, gt_samples = frames.to(cfg["device"]), gt_pos.to(cfg["device"]), gt_samples.to(cfg["device"])

        activations, learned_feats, memg_feats, results = net(frames.squeeze(0))

        # tbd: extent to cross-frame similarities? e.g. from 1xC components to NxC components
        toas = memg_feats[..., 1].clone()
        toas += cfg.toa_offset if hasattr(cfg, 'toa_offset') else 0
        labels, comp_idcs = construct_labels_indexing(toas, gt_samples[0])
        frame_idx = torch.argmax(memg_feats[~labels.bool()][:, 4])  # decide based on strongest confidence among labeled comps
        ref_comp = learned_feats[frame_idx][~labels[frame_idx].bool()][None, ...]

        alt_labels, all_comp = list(zip(*[[l, f] for l, f, m in zip(labels, learned_feats, memg_feats) if m[0, -1] > conf_threshold]))

        # find corresponding components
        match_idcs = [[int(torch.argmin(torch.cdist(ref_comp, frame_comps, p=2)))] for frame_comps in learned_feats]

        # validation loss
        loss_bce = cfg["bce_weight"] * bce_criterion(activations.flatten(), labels.flatten())
        loss_contrastive = constrastive_criterion(ref_comp, torch.vstack(all_comp), torch.hstack(alt_labels))
        loss_sum = loss_bce + loss_contrastive
        running_vloss += loss_sum

        # f1 score
        match_true = torch.where(~labels.to(dtype=bool))[-1].cpu()
        match_idcs = [int(torch.argmin(torch.cdist(ref_comp, frame_comps, p=2)).cpu()) for frame_comps in learned_feats]
        val_f1_score = f1_score(match_true, match_idcs, average='macro')
        running_vf1 += val_f1_score

        if cfg["logging"] and cfg["plot_opt"] and i%10 == 1:
            try:
                wrapped_channel_plot(frames, results, memg_feats, comp_idcs, gt_pos, gt_samples, log_key="val_img_chart")
            except TypeError as e:
                print(e)
        
        if cfg["logging"]:
            wandb.log({"val_loss_sum": loss_sum})
            wandb.log({"val_loss_contrastive": loss_contrastive})
            wandb.log({"val_loss_bce": loss_bce})
        
    avg_vloss = running_vloss / (i + 1)
    avg_vf1 = running_vf1 / (i + 1)
    if cfg.logging:
        wandb.log({"val_loss_avg_epoch": avg_vloss})
        wandb.log({"val_f1_score_avg_epoch": avg_vf1})

    print('Epoch number {}\ntrain loss: {}, valid. loss: {}'.format(epoch, avg_loss, avg_vloss))

    # early stopping
    early_stopping(train_loss=avg_loss, validation_loss=avg_vloss)
    if early_stopping.early_stop:
        save_model(Path('.'), epoch=epoch)
        print("Finished at epoch:", epoch)
        break

save_model(Path('.'), epoch=epoch)
