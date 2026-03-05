import numpy as np
from math import sin, cos, radians
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from matplotlib.ticker import MultipleLocator
import torch


def cart2sph(
    cart: np.ndarray,  # (..., 2/3)
) -> np.ndarray:  # (..., 2/3)
    '''
    https://github.com/DavidDiazGuerra/Cross3D/blob/master/acousticTrackingDataset.py
    '''
    xy2 = cart[..., 0]**2 + cart[..., 1]**2
    ndim = cart.shape[-1]
    assert ndim in [2, 3], cart.shape
    if ndim == 3:
        return np.stack(
            (
                np.sqrt(xy2 + cart[..., 2]**2),
                np.arctan2(cart[..., 2], np.sqrt(xy2)),  # elevation defined between (-pi/2, pi/2)
                np.arctan2(cart[..., 1], cart[..., 0])
            ), axis=-1
        )  # (..., 3)
    else:
        return np.stack(
            (
                np.sqrt(xy2),
                np.arctan2(cart[..., 1], cart[..., 0])
            ), axis=-1
        )  # (..., 2)
    

def rotated_triangle_marker(rot_az_angle):
    verts = np.array([
        [0, 1],
        [-np.sqrt(3)/2, -0.5],
        [np.sqrt(3)/2, -0.5],
        [0, 1]
    ])
    if isinstance(rot_az_angle, np.ndarray):
        rot_az_angle = rot_az_angle.item()  # avoid __array_wrap__ warnings
    theta = radians(float(rot_az_angle + 30))  # + 30deg to align with coordinate system
    c, s = cos(theta), sin(theta)
    R = np.array([[c, -s], [s, c]])
    verts_rot = verts @ R.T
    return Path(verts_rot)


def plot_acoustics(exp, example, cmap='Greys', figsize=None):
    print(
        f"Speakers starting at",  
        *[f"{angle:.2f}°" for angle in example['target_angle'][0, :, 0, -1].tolist()]
    )

    # forward experiment
    exp.preprocessing(example)
    example['mix_spectrum'] = exp.get_spatial_spectrum(example['mix_stft'].unsqueeze(1), example)
    example['dry_spectrum'] = exp.get_spatial_spectrum(example['dry_stft'], example)
    
    # set plot params
    n_speaker = exp.n_speaker
    fig, axs = plt.subplots(ncols=2 + n_speaker, figsize=figsize)
    colors = plt.cm.tab10.colors
    room_sz = example['room_sz'][0]
    
    # plot acoustic setup
    axs[0].set_xlim(0, room_sz[0])
    axs[0].set_ylim(0, room_sz[1])
    # axs[0].set_aspect(room_sz[0] / room_sz[1])
    axs[0].set_aspect('equal', adjustable='box')
    axs[0].xaxis.set_major_locator(MultipleLocator(1))
    axs[0].yaxis.set_major_locator(MultipleLocator(1))
    axs[0].grid(True)
    axs[0].tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)
    axs[0].scatter(
        example['mic_pos'][0, :, 0], example['mic_pos'][0, :, 1], color='k', marker='.', s=5
    )
    for spk_idx in range(n_speaker):
        # start
        axs[0].plot(
            example['spk_traj'][0, spk_idx, 0, 0], 
            example['spk_traj'][0, spk_idx, 0, 1], 
            marker='+', color=colors[spk_idx], markersize=7, markeredgewidth=2
        )

        # stop
        axs[0].plot(
            example['spk_traj'][0, spk_idx, -1, 0], 
            example['spk_traj'][0, spk_idx, -1, 1], 
            marker='.', color=colors[spk_idx], markersize=7
        )

        # trajectory
        axs[0].scatter(
            example['spk_traj'][0, spk_idx, :, 0], 
            example['spk_traj'][0, spk_idx, :, 1], 
            color=colors[spk_idx], s=1, alpha=0.1
        )
    axs[0].set_aspect('equal', adjustable='box')
    
    # mixed speech
    n_frames = example['mix_stft'].shape[-1]
    extent = (0, n_frames, 0, exp.n_az_doa)
    axs_pos = axs[0].get_position()
    axs[1].imshow(
        example['mix_spectrum'][0, 0], origin='lower', extent=extent, aspect='auto', cmap=cmap
    )
    
    for spk_idx in range(n_speaker):
        axs[1].scatter(
            torch.arange(n_frames),
            example['target_idx'][0, spk_idx, :, -1], 
            color=colors[spk_idx], alpha=0.1, s=1
        )
    axs[1].set_xticks([])
    axs[1].set_yticks([])
    pos = axs[1].get_position()
    axs[1].set_position([pos.x0, axs_pos.y0, axs_pos.width, axs_pos.height])
    
    # anechoic speech
    for spk_idx in range(n_speaker):
        axs[2+spk_idx].imshow(
            example['dry_spectrum'][0, spk_idx], origin='lower', extent=extent, aspect='auto', cmap=cmap
        )
        axs[2+spk_idx].scatter(
            torch.arange(n_frames), 
            example['target_idx'][0, spk_idx, :, -1], 
            color=colors[spk_idx], alpha=0.5, s=1
        )
        axs[2+spk_idx].set_xticks([])
        axs[2+spk_idx].set_yticks([])
        pos = axs[2+spk_idx].get_position()
        axs[2+spk_idx].set_position([pos.x0, axs_pos.y0, axs_pos.width, axs_pos.height])
    plt.show()


def animate_trajectory(
    example,
    n_speaker: int = 2,
    wall_dist: float = 0.5,
    fps: int = 20,
    dpi: int = 60,
    base_markersize: float = 6,
    base_linewidth: float = 2,
    base_dpi: float = 100,
    ax: mpl.axes.Axes = None,
    show_path: bool = False,
    show_forces: bool = False,
    show_boundaries: bool = False,
    show_driving_rect: bool = False,
):
    arrow_scale = 2.0
    force_scale = 100
    arrow_width = 0.01

    border_color = 'red'
    room_sz = example['room_sz'][:2]
    array_pos = example['array_pos'][:2]
    drv_rect_size = 2 * wall_dist
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=room_sz, dpi=dpi)
    ax.set_xlim(0, room_sz[0])
    ax.set_ylim(0, room_sz[1])
    ax.set_aspect('auto')
    ax.xaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(1))
    ax.grid(True)
    ax.tick_params(axis='both', which='both', length=0, labelleft=False, labelbottom=False)
    
    scale = base_dpi / fig.get_dpi()
    markersize = base_markersize * scale
    linewidth = base_linewidth * scale

    # boundaries
    if show_boundaries:
        xmin_o, ymin_o = 0, 0
        xmax_o, ymax_o = room_sz[:2]
        outer = [(xmin_o, ymin_o), (xmax_o, ymin_o), (xmax_o, ymax_o), (xmin_o, ymax_o), (xmin_o, ymin_o)]
        xmin_i, ymin_i = np.ones(2) * wall_dist
        xmax_i, ymax_i = room_sz[:2] - wall_dist
        inner = [(xmin_i, ymin_i), (xmin_i, ymax_i), (xmax_i, ymax_i), (xmax_i, ymin_i), (xmin_i, ymin_i)]
        vertices = outer + inner
        codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY] + [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
        path = Path(vertices, codes)
        frame_patch = mpl.patches.PathPatch(path, facecolor='white', edgecolor=border_color, hatch='/', linewidth=1.0, alpha=1.0)
        ax.add_patch(frame_patch)
        circle = mpl.patches.Circle(array_pos, 0.5, facecolor='white', edgecolor=border_color, hatch='/', linewidth=1.0, alpha=1)
        ax.add_patch(circle)

    # plot mic array
    marker_path = rotated_triangle_marker(example['mic_rot'])
    ax.plot(array_pos[0], array_pos[1], linestyle='-', color='k', marker=marker_path, markersize=1.5*markersize)

    # containers
    lines, points, drv_patches = [], [], []
    bforce_quivers, dforce_quivers = [], []
    ssforce_quivers = []  # list of list: ssforce_quivers[i][j] = arrow from speaker j to i
    colors = plt.cm.tab10.colors

    for i in range(n_speaker):
        color = colors[i % 10]

        line = ax.plot([], [], color=color, ls='-', lw=linewidth, alpha=0.5)[0] if show_path else None
        lines.append(line)
        point = ax.plot([], [], marker='o', color=color, markersize=markersize, alpha=1)[0]
        points.append(point)

        if show_driving_rect:
            drv_rect = mpl.patches.Rectangle((0,0), drv_rect_size, drv_rect_size, facecolor='none', edgecolor=color, lw=linewidth, alpha=0.9)
            ax.add_patch(drv_rect)
        else:
            drv_rect = None
        drv_patches.append(drv_rect)

        # boundary and driving quivers
        if show_forces:
            bq = ax.quiver([0], [0], [0], [0], color=border_color, angles='xy', scale_units='xy', scale=arrow_scale, width=arrow_width, alpha=0.8)
            dq = ax.quiver([0], [0], [0], [0], color=color, angles='xy', scale_units='xy', scale=arrow_scale, width=arrow_width, alpha=0.8)
        else:
            bq = dq = None
        bforce_quivers.append(bq)
        dforce_quivers.append(dq)

        # speaker–speaker quivers
        quivers_i = []
        if show_forces:
            for j in range(n_speaker):
                q = ax.quiver([0], [0], [0], [0], color=colors[j % 10], angles='xy', scale_units='xy', scale=arrow_scale, width=arrow_width, alpha=0.8)
                quivers_i.append(q)
        ssforce_quivers.append(quivers_i)

    # interpolation helper
    n_frames = int(example['audio_frames'])
    n_ani_frames = int(example['audio_time'] * fps)

    def interpolate_trajectory(traj):
        orig_shape = traj.shape
        n_batch = np.prod(orig_shape[:-2]).astype(int)
        traj_interp = np.empty((n_batch, n_ani_frames, 2))
        for batch_idx in range(n_batch):
            for coord in range(2):
                traj_interp[batch_idx,:,coord] = np.interp(
                    np.arange(n_ani_frames),
                    np.linspace(0, n_ani_frames, num=n_frames, endpoint=False),
                    traj.reshape(n_batch, n_frames, 2)[batch_idx,:,coord]
                )
        return traj_interp.reshape(*traj.shape[:-2], n_ani_frames, 2)

    spk_pos = interpolate_trajectory(example['spk_traj'][..., :2])
    boundary_force = interpolate_trajectory(example['boundary_force'])
    driving_force = interpolate_trajectory(example['driving_force'])
    driving_pos = interpolate_trajectory(example['driving_pos'])
    spk_spk_force = interpolate_trajectory(example['spk_spk_force'])  # shape: (n_speaker, n_speaker, n_frames, 2)

    # init function
    def init():
        for line, point in zip(lines, points):
            if line: line.set_data([], [])
            point.set_data([], [])
        for rect in drv_patches:
            if rect: rect.set_xy((-100,-100))
        for bq, dq, quivers_i in zip(bforce_quivers, dforce_quivers, ssforce_quivers):
            if bq: bq.set_offsets([[0,0]]); bq.set_UVC([0],[0])
            if dq: dq.set_offsets([[0,0]]); dq.set_UVC([0],[0])
            for q in quivers_i:
                q.set_offsets([[0,0]]); q.set_UVC([0],[0])
        return [obj for obj in (lines + points + drv_patches + bforce_quivers + dforce_quivers + sum(ssforce_quivers, [])) if obj]

    # update function
    def update(frame):
        for i in range(n_speaker):
            if show_path: lines[i].set_data(spk_pos[i,:frame+1,0], spk_pos[i,:frame+1,1])
            points[i].set_data([spk_pos[i,frame,0]], [spk_pos[i,frame,1]])

            if show_driving_rect:
                dx, dy = driving_pos[i, frame]
                drv_patches[i].set_xy((dx - drv_rect_size/2, dy - drv_rect_size/2))

            if show_forces:
                px, py = spk_pos[i, frame]
                bf = boundary_force[i, frame]; df = driving_force[i, frame]
                bforce_quivers[i].set_offsets([[px, py]]); bforce_quivers[i].set_UVC([bf[0]],[bf[1]])
                dforce_quivers[i].set_offsets([[px, py]]); dforce_quivers[i].set_UVC([df[0]],[df[1]])
                for j, q in enumerate(ssforce_quivers[i]):
                    f = spk_spk_force[i, j, frame]
                    q.set_offsets([[px, py]])
                    q.set_UVC([f[0]], [f[1]])

        return [obj for obj in (lines + points + drv_patches + bforce_quivers + dforce_quivers + sum(ssforce_quivers, [])) if obj]

    ani = FuncAnimation(
        fig if fig is not None else ax.figure,
        update,
        frames=n_ani_frames,
        init_func=init,
        blit=True,
        interval=1000/fps,
    )

    if fig is not None:
        fig.tight_layout(pad=0)
        plt.close(fig)

    return ani

