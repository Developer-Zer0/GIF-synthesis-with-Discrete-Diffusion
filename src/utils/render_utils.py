# Inspired by
# - https://github.com/anindita127/Complextext2animation/blob/main/src/utils/visualization.py
# - https://github.com/facebookresearch/QuaterNet/blob/main/common/visualization.py

from typing import List, Tuple
import numpy as np
from src.utils.joint_map_smplh import mmm_kinematic_tree, mmm_to_smplh_scaling_factor
from src.utils.joint_map_smplh import smplh_kinematic_tree
from src.utils.joint_map_smplh import mmm_kinematic_tree

mmm_colors = ['black', 'magenta', 'red', 'green', 'blue']

import math

def smoothing_factor(t_e, cutoff):
    r = 2 * math.pi * cutoff * t_e
    return r / (r + 1)

def exponential_smoothing(a, x, x_prev):
    return a * x + (1 - a) * x_prev

class OneEuroFilter:
    def __init__(self, t0, x0, dx0=0.0, min_cutoff=1.0, beta=0.0,
                 d_cutoff=1.0):
        """Initialize the one euro filter."""
        # The parameters.
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        # Previous values.
        self.x_prev = x0
        self.dx_prev = dx0
        self.t_prev = t0

    def __call__(self, t, x):
        """Compute the filtered signal."""
        t_e = t - self.t_prev

        # The filtered derivative of the signal.
        a_d = smoothing_factor(t_e, self.d_cutoff)
        dx = (x - self.x_prev) / t_e
        dx_hat = exponential_smoothing(a_d, dx, self.dx_prev)

        # The filtered signal.
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = smoothing_factor(t_e, cutoff)
        x_hat = exponential_smoothing(a, x, self.x_prev)

        # Memorize the previous values.
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t

        return x_hat


def init_axis(fig, title, radius=1.5, dist=10):
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.view_init(elev=20., azim=-60)

    fact = 2
    ax.set_xlim3d([-radius / fact, radius / fact])
    ax.set_ylim3d([-radius / fact, radius / fact])
    ax.set_zlim3d([0, radius])

    ax.set_aspect('auto')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.set_axis_off()

    ax.dist = dist
    ax.grid(b=False)

    ax.set_title(title, loc='center', wrap=True, fontsize=15)
    return ax


def plot_floor(ax, minx, maxx, miny, maxy, minz):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    # Plot a plane XZ
    verts = [
        [minx, miny, minz],
        [minx, maxy, minz],
        [maxx, maxy, minz],
        [maxx, miny, minz]
    ]
    xz_plane = Poly3DCollection([verts], zorder=1)
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 1))
    ax.add_collection3d(xz_plane)

    # Plot a bigger square plane XZ
    radius = max((maxx - minx), (maxy - miny))

    # center +- radius
    minx_all = (maxx + minx) / 2 - radius
    maxx_all = (maxx + minx) / 2 + radius

    miny_all = (maxy + miny) / 2 - radius
    maxy_all = (maxy + miny) / 2 + radius

    verts = [
        [minx_all, miny_all, minz],
        [minx_all, maxy_all, minz],
        [maxx_all, maxy_all, minz],
        [maxx_all, miny_all, minz]
    ]
    xz_plane = Poly3DCollection([verts], zorder=1)
    xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
    # ax.add_collection3d(xz_plane)
    return ax


def update_camera(ax, root, radius=1.5):
    fact = 2
    ax.set_xlim3d([-radius / fact + root[0], radius / fact + root[0]])
    ax.set_ylim3d([-radius / fact + root[1], radius / fact + root[1]])


def render_animation(joints: np.ndarray, output: str = "notebook", title: str = "",
                     fps: float = 12.5,
                     # kinematic_tree: List[List[int]] = mmm_kinematic_tree,
                     colors: List[str] = mmm_colors,
                     figsize: Tuple[int] = (4, 4),
                     fontsize: int = 25,
                     dataset_name: str = "HumanML3D"):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    import matplotlib.patheffects as pe
    # plt.rcParams.update({'font.size': fontsize})

    # Z is gravity here
    x, y, z = 0, 1, 2

    # Convert mmm joints for visualization
    # into smpl-h "scale" and axis
    if dataset_name == "HumanML3D":
        joints = joints.copy()[..., [2, 0, 1]]
        kinematic_tree = smplh_kinematic_tree
    else:
        joints = joints.copy()[..., [2, 0, 1]] * mmm_to_smplh_scaling_factor
        kinematic_tree = mmm_kinematic_tree

    # render_sample_image(joints, kinematic_tree, title, output)
    # return

    # Create a figure and initialize 3d plot
    fig = plt.figure(figsize=figsize)
    ax = init_axis(fig, title)

    # Create spline line
    trajectory = joints[:, 0, [x, y]]
    # joints[..., 0] -= joints[:, 0:1, 0]
    # joints[..., 1] -= joints[:, 0:1, 1]
    avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
    draw_offset = int(25 / avg_segment_length)
    spline_line, = ax.plot(*trajectory.T, zorder=10, color="white")

    # Create a floor
    minx, miny, _ = joints.min(axis=(0, 1))
    maxx, maxy, _ = joints.max(axis=(0, 1))
    plot_floor(ax, minx, maxx, miny, maxy, 0)

    # Put the character on the floor
    height_offset = np.min(joints[:, :, z])  # Min height
    joints = joints.copy()
    joints[:, :, z] -= height_offset

    # Initialization for redrawing
    lines = []
    initialized = False

    def update(frame):
        nonlocal initialized
        skeleton = joints[frame]

        root = skeleton[0]
        update_camera(ax, root)

        for index, (chain, color) in enumerate(zip(reversed(kinematic_tree), reversed(colors))):
            if not initialized:
                lines.append(ax.plot(skeleton[chain, x],
                                     skeleton[chain, y],
                                     skeleton[chain, z], linewidth=8.0, color=color, zorder=20,
                                     path_effects=[pe.SimpleLineShadow(), pe.Normal()]))

            else:
                lines[index][0].set_xdata(skeleton[chain, x])
                lines[index][0].set_ydata(skeleton[chain, y])
                lines[index][0].set_3d_properties(skeleton[chain, z])

        left = max(frame - draw_offset, 0)
        right = min(frame + draw_offset, trajectory.shape[0])

        spline_line.set_xdata(trajectory[left:right, 0])
        spline_line.set_ydata(trajectory[left:right, 1])
        spline_line.set_3d_properties(np.zeros_like(trajectory[left:right, 0]))
        initialized = True

    fig.tight_layout()
    frames = joints.shape[0]
    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, repeat=False)

    if output == "notebook":
        from IPython.display import HTML
        HTML(anim.to_jshtml())
    else:
        anim.save(output, writer='ffmpeg', fps=fps)

    plt.close()

def render_sample_image(joints, kinematic_tree, title, output):
    x, y, z = 0, 1, 2
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe
    plt.rcParams.update({'font.size': 25})
    fig1 = plt.figure(figsize=(10, 10))
    ax1 = init_axis(fig1, title)

    # Create spline line
    # joints[:, :, [x, y]] *= 1.2
    trajectory = joints[:, 0, [x, y]]
    # joints[..., 0] -= joints[:, 0:1, 0]
    # joints[..., 1] -= joints[:, 0:1, 1]
    avg_segment_length = np.mean(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)) + 1e-3
    draw_offset = int(25 / avg_segment_length)
    spline_line, = ax1.plot(*trajectory.T, zorder=10, color="white")

    # Create a floor
    minx, miny, _ = joints.min(axis=(0, 1))
    maxx, maxy, _ = joints.max(axis=(0, 1))
    plot_floor(ax1, minx, maxx, miny, maxy, 0)

    # Put the character on the floor
    height_offset = np.min(joints[:, :, z])  # Min height
    joints = joints.copy()
    joints[:, :, z] -= height_offset


    # Initialization for redrawing
    lines = []
    initialized = False

    # frames = np.append(np.arange(0, joints.shape[0], joints.shape[0]/5, dtype=int), [joints.shape[0]-1])
    frames = np.arange(0, joints.shape[0], joints.shape[0]/6, dtype=int)

    mean = 0
    for frame in frames:
        mean += joints[frame][0]
    mean = mean/len(frames)

    update_camera(ax1, mean, radius=2.0)

    for frame in frames:
        # nonlocal initialized_t
        skeleton = joints[frame]

        # root = skeleton[0]
        # update_camera(ax1, root)

        for index, (chain, color) in enumerate(zip(reversed(kinematic_tree), reversed(mmm_colors))):
            if not initialized:
                lines.append(ax1.plot(skeleton[chain, x],
                                     skeleton[chain, y],
                                     skeleton[chain, z], linewidth=10.0, color=color, zorder=20, alpha=(frame+20)/(frames[-1]+20),
                                     path_effects=[pe.SimpleLineShadow(), pe.Normal()]))

            else:
                lines[index][0].set_xdata(skeleton[chain, x])
                lines[index][0].set_ydata(skeleton[chain, y])
                lines[index][0].set_3d_properties(skeleton[chain, z])

        left = max(frame - draw_offset, 0)
        right = min(frame + draw_offset, trajectory.shape[0])

        spline_line.set_xdata(trajectory[left:right, 0])
        spline_line.set_ydata(trajectory[left:right, 1])
        spline_line.set_3d_properties(np.zeros_like(trajectory[left:right, 0]))
        # initialized = True

    plt.tight_layout()
    plt.savefig(output)

    print('DONE!')

