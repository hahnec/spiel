import matplotlib.pyplot as plt
import numpy as np


def spiel_cube_plot(est_xyz, gt_xyz=None, c=None, hide_axes_opt=False, hide_grid_opt=False, hide_ticks_opt=False, show_opt=False, fig=None, axs=None, margin=.2):

    # LaTeX style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # 3-D plot layout
    plt.rcParams['grid.color'] = "k"
    plt.rcParams['grid.alpha'] = 1.0
    plt.rcParams['grid.linewidth'] = .1

    fig = plt.figure(figsize=(2.5, 2.5)) if fig is None else fig
    #ax0 = fig.add_subplot(111)
    ax1 = fig.add_subplot(111, projection='3d')
    #fig, axs = plt.subplots(1, 2, figsize=(2.5, 2.5)) if fig is None and axs is None else fig, axs
    ax1.set_facecolor('white')
    #ax1.set_axis_bgcolor('white')

    if gt_xyz is not None:
        ax1.scatter(*gt_xyz, marker='+', c='k', s=50, label='Ground Truth $\mathbf{s}$')

    est_scatter_color = ax1.scatter(*est_xyz, c=c, cmap=plt.cm.cool, marker='o', s=40, alpha=.5)
    est_scatter_point = ax1.scatter(*est_xyz, c='k', marker='.', s=10, label='Estimates $\mathbf{s}^\star$')

    ax1.set_xlabel('$x$ [mm]', fontsize=12)
    ax1.set_ylabel('$y$ [mm]', fontsize=12)
    ax1.set_zlabel('$z$ [mm]', fontsize=12)

    # enforce automated equal axes
    ref = gt_xyz.T if gt_xyz is not None else est_xyz.T
    # lowest number in the array
    xmin = np.amin(ref[:, 0])*(1+margin)
    xmax = np.amax(ref[:, 0])*(1+margin)  # highest number in the array
    ymin = np.amin(ref[:, 1])*(1+margin)
    ymax = np.amax(ref[:, 1])*(1+margin)  # highest number in the array
    zmin = np.amin(ref[:, 2])*(1-margin)
    zmax = np.amax(ref[:, 2])*(1+margin)  # highest number in the array
    ax1.set_xlim3d(xmin, xmax)
    ax1.set_ylim3d(ymin, ymax)
    ax1.set_zlim3d(zmin, zmax)

    # plot line from measured point to corresponding ground-truth
    for est, gt in zip(est_xyz.T, gt_xyz.T):
        ax1.plot([est[0], gt[0]], [est[1], gt[1]], [est[2], gt[2]], color='gray', linestyle=':', linewidth=.5, alpha=.2)

    ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    # turn axes off
    if hide_axes_opt:
        plt.axis('off')

    # turn grid lines off
    if hide_grid_opt:
        ax1.grid(False)

    # Hide axes ticks
    if hide_ticks_opt:
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_zticks([])

    plt.legend(loc='lower left', bbox_to_anchor=(0.45, .78))#

    plt.tight_layout(rect=[-0.1, 0.1, 1, 1])#

    if show_opt: plt.show()

    return ax1, est_scatter_color


def spiel_colorbar(est_scatter_color):

    # LaTeX style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    fig = plt.figure(figsize=(1.2, 2.5))
    ax1 = fig.add_subplot(111)

    fig.colorbar(est_scatter_color, cax=ax1, label='RMSE [mm]', pad=0.04, shrink=2/3)
    ax1.set_ylabel(r'RMSE [mm]', color='black', labelpad=12, fontsize=18) #, labelpad=-65, y=.75
    #ax1.xaxis.set_ticks_position("top")

    plt.tight_layout()

    return ax1

def spiel_2d_projected_plots(est, gt=None, valid=None, c=None, circle_radius=1, font_size=18, show_opt=False, fig=None, axs=None):

    # LaTeX style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    projected_points = lambda xyz, idx: np.delete(xyz, idx, axis=0)
    projected_axis = lambda idx: ['x', 'y', 'z'][idx]

    valid = np.ones(est.shape[-1], dtype=bool) if valid is None else valid

    est_xyz = est[:, valid]
    gt_xyz = gt[:, valid]

    fig, axs = plt.subplots(ncols=3, nrows=1, figsize=(5.5, 3.5), constrained_layout=True) if fig is None else fig, axs
    for idx in range(3):
        #ax.set_title(projected_axis(idx))
        measurements = axs[idx].scatter(*projected_points(est_xyz.copy(), idx=idx), c='k', marker='.', s=10, label='Locations $\mathbf{s}^\star$')
        axs[idx].scatter(*projected_points(est_xyz.copy(), idx=idx), c=c, cmap=plt.cm.cool, marker='o', s=40, alpha=.5)
        if gt is not None:
            gt_scatter = axs[idx].scatter(*projected_points(gt.copy(), idx=idx), marker='+', c='k', s=50, label='Ground Truth')
            # plot RMSE as circles
            for gt_pt in projected_points(gt.copy(), idx=idx).T:
                circle = plt.Circle(gt_pt, circle_radius, color='k', fill=False, linestyle='--', label='Mean RMSE')
                axs[idx].add_artist(circle)
            # helper plots
            for (est_pt, gt_pt) in zip(projected_points(est_xyz.copy(), idx=idx).T, projected_points(gt_xyz.copy(), idx=idx).T):
                # plot line from measured point to corresponding ground-truth
                axs[idx].plot([est_pt[0], gt_pt[0]], [est_pt[1], gt_pt[1]], color='gray' , linestyle=':', alpha=0.5)

        axs[idx].set_xlim(-125, 125)
        axs[idx].set_ylim(-125, 125) if projected_axis(idx) == 'z' else axs[idx].set_ylim(30, 270)

        # equal aspect ratio in each subplot
        aspect_ratio = 1.0
        xleft, xright = axs[idx].get_xlim()
        ybottom, ytop = axs[idx].get_ylim()
        axs[idx].set_aspect(abs((xright-xleft)/(ybottom-ytop))*aspect_ratio)

        label_size = 15
        if projected_axis(idx) == 'x':
            axs[idx].set_xlabel('y [mm]', fontsize=label_size)
            axs[idx].set_ylabel('z [mm]', fontsize=label_size) 
        elif projected_axis(idx) == 'y': 
            axs[idx].set_xlabel('x [mm]', fontsize=label_size) 
            axs[idx].set_ylabel('z [mm]', fontsize=label_size)
        elif projected_axis(idx) == 'z': 
            axs[idx].set_xlabel('x [mm]', fontsize=label_size) 
            axs[idx].set_ylabel('y [mm]', fontsize=label_size)

    #fig.suptitle('2-D projected')

    # global legend
    handles, labels = axs[-1].get_legend_handles_labels()
    handles = [measurements, gt_scatter, circle] if gt is not None else [measurements]
    fig.legend(handles, labels, fontsize=font_size, title_fontsize=font_size, loc='upper right')

    if show_opt: plt.show()

    return axs

def cross_sectional_plot(ax, idx, est, gt=None, valid=None, c=None, circle_radius=1, font_size=18, fig=None):

    # LaTeX style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    projected_points = lambda xyz, idx: np.delete(xyz, idx, axis=0)
    projected_axis = lambda idx: ['x', 'y', 'z'][idx]

    valid = np.ones(est.shape[-1], dtype=bool) if valid is None else valid

    est_xyz = est[:, valid]
    gt_xyz = gt[:, valid]

    measurements = ax.scatter(*projected_points(est_xyz.copy(), idx=idx), c='k', marker='.', s=10, label='Locations $\mathbf{s}^\star$')
    ax.scatter(*projected_points(est_xyz.copy(), idx=idx), c=c[~np.isnan(c)], cmap=plt.cm.cool, marker='o', s=40, alpha=.5)
    ax.scatter(*projected_points(est_xyz.copy(), idx=idx), c=c[~np.isnan(c)], cmap=plt.cm.cool, marker='o', s=40, alpha=.5)
    ax.scatter(*projected_points(est_xyz.copy(), idx=idx), c='k', marker='.', s=10, label='$\mathbf{s}^\star$')
    if gt is not None:
        gt_scatter = ax.scatter(*projected_points(gt.copy(), idx=idx), marker='+', c='k', s=50, label='Ground Truth')
        # plot RMSE as circles
        for gt_pt in projected_points(gt.copy(), idx=idx).T:
            circle = plt.Circle(gt_pt, circle_radius, color='k', fill=False, linestyle='--', label='Mean RMSE')
            ax.add_artist(circle)
        # helper plots
        for (est_pt, gt_pt) in zip(projected_points(est_xyz.copy(), idx=idx).T, projected_points(gt_xyz.copy(), idx=idx).T):
            # plot line from measured point to corresponding ground-truth
            ax.plot([est_pt[0], gt_pt[0]], [est_pt[1], gt_pt[1]], color='gray' , linestyle=':', linewidth=.5, alpha=.5)

    ax.set_xlim(-125, 125)
    ax.set_ylim(-125, 125) if projected_axis(idx) == 'z' else ax.set_ylim(15, 265)

    # equal aspect ratio in each subplot
    aspect_ratio = 1.0
    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*aspect_ratio)
    
    label_size = 15
    if projected_axis(idx) == 'x':
        ax.set_xlabel(r'$y$ [mm]', fontsize=label_size)
        ax.set_ylabel(r'$z$ [mm]', fontsize=label_size) 
    elif projected_axis(idx) == 'y': 
        ax.set_xlabel(r'$x$ [mm]', fontsize=label_size) 
        ax.set_ylabel(r'$z$ [mm]', fontsize=label_size)
    elif projected_axis(idx) == 'z': 
        ax.set_xlabel(r'$x$ [mm]', fontsize=label_size) 
        ax.set_ylabel(r'$y$ [mm]', fontsize=label_size)
    ax.yaxis.set_label_coords(-.25, .5)     # set x-position of y-label to force all plots to be the same size

    if fig is not None:
        # global legend
        handles, labels = ax.get_legend_handles_labels()
        handles = [measurements, gt_scatter, circle] if gt is not None else [measurements]
        fig.legend(handles, labels, fontsize=font_size, title_fontsize=font_size, loc='upper right')

    plt.tight_layout()

    return ax
