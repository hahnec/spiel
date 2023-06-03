import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from torch.utils.data import DataLoader

from config import construct_space_net_cfg
from spiel_net.dataset_spiel import SpielDataset, sample2dist
from pace.echo_detector.grad_peak_detect import GradPeakDetector


cfg = construct_space_net_cfg()

test_dataset = SpielDataset(
    dataset_path=cfg["fpath"],
    transform=None,
    flip=False,
    bg_subtract=cfg["bg_subtract"],
    blind_zone_idx=11,
    skip_transducer_idcs=None,#0,
    xy_position_gap=1,
    x_range = [-100, 100],
    y_range = [-100, 100],
    z_range = [100, 260],
    )

test_dataloader = DataLoader(
                            test_dataset,
                            shuffle=False,
                            num_workers=2,
                            batch_size=cfg["batch_size"],
                            )

pow_law_list = []
plt.figure()
for i, data in enumerate(test_dataloader):

    frames, gt_pos, gt_samples = data
    frames, gt_pos, gt_samples = frames.to(cfg["device"]), gt_pos.to(cfg["device"]), gt_samples.to(cfg["device"])

    arr = frames[0][1].cpu().numpy()

    det = GradPeakDetector(hilbert_data=arr)
    det._grad_step = 2
    det._thres_pos = 1e1
    det._ival_list = [1, 200]
    det.gradient_hysteresis()
    #plot_hyst(detector, t=t)
    #plot_grad(detector, t=t)
    #feat_plot(detector)

    echo_list = det.echo_list
    print('%s echo(es) detected' % len(echo_list))

    echo_idx = np.argmax([echo[3] for echo in det.echo_list])
    echo_est = det.echo_list[echo_idx][1]
    echo_pos = echo_est + np.argmax(arr[echo_est-2:echo_est+3]) - 2
    echo_amp = np.mean(arr[echo_pos-1:echo_pos+2])

    pow_law_list.append([echo_pos, echo_amp]) if echo_pos > 11 else None

    plt.plot(arr)   
    plt.plot(echo_pos, echo_amp, 'k.')
plt.show()

sample2dist = lambda x, c=343, fkHz=175, usr=2: c/2 * x * usr/fkHz

# power law fit
pow_law_arr = np.array(pow_law_list)
x = pow_law_arr[:, 0]
x = sample2dist(x)
y = pow_law_arr[:, 1]/max(pow_law_arr[:, 1])
n = len(x)
b = -1*(n*np.sum(np.log(x)*np.log(y)) - np.sum(np.log(x))*np.sum(np.log(y))) / \
    (n*np.sum(np.log(x)**2) - np.sum(np.log(x))**2)
a = np.exp((np.sum(np.log(y)) + b*np.sum(np.log(x))) / n)
c = 0
print(a, b)

if False:
    cost_fun = lambda p, x=x, y=y: (1/x**p[0] + p[1] - y)**2
    res = least_squares(fun=cost_fun, x0=[2, 0])
    b, c = res.x
    print(b, c)

t = np.arange(20, max(x), dtype=np.float)
f = a/t**b+c
p = t**-2

plt.figure()
plt.plot(x, y, color='k', marker='.', linestyle='', label='data points')
plt.plot(t, f, color='k', label='fit')
plt.plot(t, p, color='r', label='inverse square law')
plt.legend()
plt.show()

# plot with power law compensation
t = np.arange(0, len(arr))
g = a/t**b+c
plt.figure()
for i, data in enumerate(test_dataloader):
    frames, gt_pos, gt_samples = data
    frames, gt_pos, gt_samples = frames.to(cfg["device"]), gt_pos.to(cfg["device"]), gt_samples.to(cfg["device"])

    if i % 4 == 0:
        arr = frames[0][2]
        plt.plot(t, arr/g, label=str(i))

    #for i, arr in enumerate(frames[0]):
    #    plt.plot(t, arr/g, label=str(i))
plt.legend()
plt.show()
