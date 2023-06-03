import matplotlib.pyplot as plt
from pace.utils.axis_equal_3d import axis_equal_3D
import numpy as np


def plot_ellipsoids(elps=None, pts=None, obj_pos=None, save_opt=False, eig_opt=False, gray_opt=False, legend_opt=False, show_opt=False, fig=None, ax=None, hide_axes_opt=False, figsize=(15, 5)):

    # LaTeX style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    #fig, ax = (plt.figure(), plt.axes(projection='3d'))#
    fig = plt.figure(figsize=figsize) if fig is None else fig
    ax = fig.add_axes([.25, .25, 1, 1], projection='3d') if ax is None else ax
    #ax = fig.add_subplot(projection='3d')
    fig.patch.set_visible(False)

    colors = [None]*len(elps) if gray_opt else ['k', 'b', 'g', 'r', 'orange', 'cyan', 'gray'][:len(elps)]

    for i, (elp, c) in enumerate(zip(elps, colors)):

        # ellipsoid surface plots
        xx, yy, zz = elp.coords
        ax.plot_wireframe(xx, yy, zz, color=c, linewidth=.1, zorder=0.5)
        #ax.plot_surface(xx, yy, zz, color=c, alpha=.1, label='Channel #'+str(i), zorder=0.5)

        # transmitter and receiver position
        s1 = ax.plot(*elp.source_pos, color='k', marker='d', markersize=10, linestyle='', label=r'Transmitter $\mathbf{u}$')
        s2 = ax.plot(*elp.sensor_pos, color=c, marker='.', markersize=12, linestyle='', label=r'Receivers $\mathbf{v}_i$')
        c1 = ax.plot(*elp.center_pos, color=c, marker='x', markersize=5, linestyle='', label=r'Centers $\mathbf{c}_i$')

        # receiver orientation
        vec = elp.normal_rxn / (elp.normal_rxn**2).sum()**.5 * min(elp.eig_val)
        ax.quiver(*elp.sensor_pos, *vec, linewidth=1.5, color=c, label=None, zorder=1)

        # major axis plot
        if eig_opt:
            major_axis_rel = elp.eig_vec[0]*elp.eig_val[0] + elp.center_pos
            major_axis_abs = np.stack([elp.center_pos, major_axis_rel]).T
            ax.plot(*major_axis_abs, linestyle='--', color=c)
    
    o1 = ax.plot(*obj_pos, color='k', marker='s', markersize=10, linestyle='', label='Ground-Truth Target', zorder=1.5) if obj_pos is not None else None

    if pts is not None and isinstance(pts, (list, tuple)):
        
        # plot gradient steps
        x_p = pts[0]
        p1s = []
        for x_k in pts:
            p1 = ax.quiver(*x_p, *(x_k-x_p), linewidth=1.5, color='k', label=r'Search path $\mathbf{s}^{(j)}$', zorder=1)
            p1s.append(p1)
            x_p = x_k

        # initial guess and solution
        l1 = ax.scatter(pts[0][0], pts[0][1], pts[0][2], s=100, marker='*', color='black', label=r'Initial guess $\mathbf{s}^{(1)}$')
        l2 = ax.scatter(pts[-1][0], pts[-1][1], pts[-1][2], s=120, marker='*', color='white', edgecolors='k', linewidths=.75, label=r'Solution $\mathbf{s}^\star$', zorder=2)

    ax.set_xlabel(r'$x$ [a.u.]', fontsize=10)
    ax.set_ylabel(r'$y$ [a.u.]', fontsize=10)
    ax.set_zlabel(r'$z$ [a.u.]', fontsize=10)

    # mock-up legend markers
    import matplotlib.lines as mlines
    s2 = mlines.Line2D([], [], color='gray', marker='.', markersize=12, linestyle='', label=r'Receivers $\mathbf{v}_i$')
    c1 = mlines.Line2D([], [], color='gray', marker='x', markersize=5, linestyle='', label=r'Centers $\mathbf{c}_i$')

    if legend_opt:
        handles = [s1[0], s2, c1, l1, p1s[0], l2]
        handles = handles + [o1[0],] if o1 is not None else None
        ax.legend(handles=handles, fontsize=10, title_fontsize=12, bbox_to_anchor=(.2, .75))#, loc='left'

    if False:
        ax.view_init(elev=0, azim=-90)
        ax.tick_params(axis='y', top=False, bottom=False, left=False, right=False, labelleft=False, labelbottom=False)

    # turn axes off
    if hide_axes_opt:
        plt.axis('off')

    axis_equal_3D(ax)
    #fig.tight_layout()
    #plt.tight_layout(pad=0.05)
    #plt.axis('off')
    fig.savefig('ellipsoid_intersection.svg', transparent=True, bbox_inches='tight', pad_inches = 0) if save_opt else None
    plt.show() if show_opt else None

    return fig, ax