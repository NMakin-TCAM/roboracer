"""preview.png: map, centerline, track bounds and the raceline coloured by speed."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from planning.centerline import normals_left  # noqa: E402


def save_preview(path, m, track, traj=None, title=''):
    h, w = m.img.shape
    x0, y0 = m.origin[:2]
    extent = [x0, x0 + w * m.res, y0, y0 + h * m.res]   # assumes origin yaw = 0 (map_saver default)

    fig, ax = plt.subplots(figsize=(12, 12 * h / w + 1.0))
    ax.imshow(m.img, cmap='gray', vmin=0, vmax=255, extent=extent, interpolation='nearest')

    xy = track[:, :2]
    nl = normals_left(xy)
    close = lambda a: np.vstack((a, a[:1]))  # noqa: E731
    left = close(xy + nl * track[:, 3:4])
    right = close(xy - nl * track[:, 2:3])
    ax.plot(left[:, 0], left[:, 1], '-', color='tab:blue', lw=1, label='track bounds')
    ax.plot(right[:, 0], right[:, 1], '-', color='tab:blue', lw=1)
    ax.plot(*close(xy).T, '--', color='0.4', lw=1, label='centerline')

    if traj is not None:
        sc = ax.scatter(traj[:, 1], traj[:, 2], c=traj[:, 5], cmap='viridis', s=8, zorder=3)
        fig.colorbar(sc, ax=ax, fraction=0.03, pad=0.02, label='vx [m/s]')
        # direction of travel at the start
        p, q = traj[0, 1:3], traj[min(5, len(traj) - 1), 1:3]
        ax.annotate('', xy=q, xytext=p, zorder=4,
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
        ax.plot(*p, 'o', color='red', ms=6, zorder=4, label='start')

    ax.set_aspect('equal')
    pad = 0.5
    ax.set_xlim(min(left[:, 0].min(), right[:, 0].min()) - pad, max(left[:, 0].max(), right[:, 0].max()) + pad)
    ax.set_ylim(min(left[:, 1].min(), right[:, 1].min()) - pad, max(left[:, 1].max(), right[:, 1].max()) + pad)
    ax.set_xlabel('x [m] (map)')
    ax.set_ylabel('y [m] (map)')
    ax.set_title(title)
    ax.legend(loc='upper right', fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches='tight')
    plt.close(fig)
