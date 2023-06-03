import numpy as np
import matplotlib.pyplot as plt

from multimodal_emg import emg_envelope_model


def channel_plot(
        frames, 
        results, 
        memg_feats, 
        match_idcs, 
        gt_samples=None, 
        figsize=(10, 35), 
        save_opt=False, 
        show_opt=False, 
        norm_opt=False,
        title:str=None, 
        legend_opt=True,
        x=None,
    ):

    fig, axs = plt.subplots(nrows=len(frames), ncols=1, figsize=figsize)
    font_size = 24
    if title is not None: fig.suptitle(title)

    max_val = max([max(max(f), max(r)) for f, r in zip(frames, results)])
    xlen = len(frames[0])
    x = np.arange(xlen) if x is None else x

    colors = ['#0051a2', '#97964a', '#ffd44f', '#f4777f', '#93003a']
    lw = 3

    for i, (f, r) in enumerate(zip(frames, results)):
        (l1,) = axs[i].plot(x, f, color=colors[4], lw=lw, linestyle='dashed', label='$\mathcal{H}[y_n(\mathbf{x})]$')
        (l2,) = axs[i].plot(x, r, color=colors[3], lw=lw, linestyle='dashdot', label='$M(\mathbf{\hat{p}}^\star_n;\mathbf{x})$')
        axs[i].set_ylim((0, max_val)) if norm_opt else axs[i].set_ylim((0, axs[i].get_ylim()[-1]))
        axs[i].set_xlim((0, len(f)-1))
        conf_str = '$C_{'+str(i)+'}='+str(round(float(memg_feats[i, 0, -1])*1e3, 2))+'$'
        axs[i].text(x=axs[i].get_xlim()[1]*.025, y=axs[i].get_ylim()[1]*.5, s=conf_str, fontsize=font_size)
        axs[i].tick_params(axis='y', top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)
        axs[i].tick_params(axis='x', top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

        # detected component
        j = match_idcs[i]
        mu = memg_feats[i, j, 1]
        toa = memg_feats[i, j, 6]
        h = emg_envelope_model(*memg_feats[i, j, :4], x=np.arange(xlen))
        (p1,) = axs[i].plot([mu, mu], [max_val, 0], color='tab:red', lw=lw, linestyle='-.', label='$\mu^\star_{n,k}$')
        p2 = axs[i].fill_between(np.linspace(0, len(h)-1, len(h)), h, y2=axs[i].get_ylim()[0], hatch='/', facecolor='darkgreen', alpha=0.3, label='$\mathbf{p}^\star_{n,k}$')
        (p3,) = axs[i].plot([toa, toa], [max_val, 0], color='tab:green', lw=lw, linestyle='--', label='$t^\star_{n,k}$')

        # plot ground-truth distance (if available)
        (p4,) = axs[i].plot([gt_samples[i], gt_samples[i]], [max_val, 0], color='tab:blue', lw=lw, linestyle=':', label='$\mu_{gt}$') if gt_samples is not None else [None]
    
    # labels
    axs[-1].set_xlabel(r'Time $t_i$ [a.u.]', fontsize=font_size)
    for n, ax in enumerate(axs):
        ax.set_ylabel(r'$y_'+str(n+1)+'(t_i)$ [a.u.]', fontsize=font_size)

    # global legend
    if legend_opt:
        handles, labels = axs[-1].get_legend_handles_labels()
        handles = [l1, l2, p1, p2] if not sum([True for m in match_idcs if m == -1]) == len(match_idcs) and len(f) > 0 else [l1, l2]
        handles = handles + [p3,] if len(f) > 0 and memg_feats.shape[-1] >= 7 else handles
        handles = handles + [p4,] if gt_samples is not None else handles
        fig.legend(handles, labels, fontsize=font_size, title_fontsize=font_size, loc='upper right')
        
    plt.tight_layout()
    if save_opt: plt.savefig('./figs/classification.pdf', format="pdf", transparent=True)
    if show_opt: plt.show()

    return fig