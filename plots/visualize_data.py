#!/usr/bin/env python3
import datetime
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# ORIGINAL CODE FROM WHOI
def main():
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

    data_file = "../data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"
    mat = loadmat(os.path.join(project_path, "data", data_file))

    out_dir = os.path.join(project_path, f"data/{data_file.split('.')[0]}")

    os.makedirs(out_dir, exist_ok=True)

    lat = mat['lat'][0,:]
    lon = mat['lon'][:,0]
    time = mat['ocean_time'].squeeze()
    time = [datetime.datetime.fromordinal(int(t)) + datetime.timedelta(days=t%1) - datetime.timedelta(days = 366) for t in time]

    current_u = mat['u'].transpose((2,0,1)) # lon,lat,idx -> idx,lon,lat
    current_v = mat['v'].transpose((2,0,1)) # lon,lat,idx -> idx,lon,lat

    for t, u, v in tqdm(zip(time, current_u, current_v), total=len(time)):
        plt.figure(figsize=[12,6])
        ax = plt.gca()
        plotQuiverData(ax=ax,
                       x_coords=lon,
                       y_coords=lat,
                       u_component=u,
                       v_component=v,
                       quiver_stride=3)

        plt.show()


def plotQuiverData(x_coords, y_coords, u_component, v_component, ax=None, quiver_stride=1):
    """
    x_coords: Mx1 Vector of uniformly spaced coordinates
    y_coords: Nx1 Vector of uniformly spaced coordinates
    u_component: MxN matrix of u-component magnitudes
    v_component: MxN matrix of v-component magnitudes
    ax: (optional) plotting axes.  If not provided, a new figure will be created
    quiver_stride: (optional) subsampling factor for quiver plot.  If not provided, defaults to 1
    """

    if ax is None:
        plt.figure(figsize=[12,6])
        ax = plt.gca()

    lx = len(x_coords)
    ly = len(y_coords)

    print(f"x_coords shape: {x_coords.shape}, y_coords shape: {y_coords.shape}")
    print(f"u_component shape: {u_component.shape}, v_component shape: {v_component.shape}")
    print(f"lx: {lx}, ly: {ly}")


    # assert u_component.shape == (lx, ly)
    # assert v_component.shape == (lx, ly)


    yy, xx = np.meshgrid(y_coords, x_coords)
    vorticity = computeVorticity(u_component, v_component)

    ds_xx = xx[::quiver_stride,::quiver_stride]
    ds_yy = yy[::quiver_stride,::quiver_stride]
    ds_u_component = u_component[::quiver_stride,::quiver_stride]
    ds_v_component = v_component[::quiver_stride,::quiver_stride]

    plt.imshow(np.isnan(vorticity).T,
               origin="lower",
               cmap="winter",
               extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()])

    plt.quiver(ds_xx, ds_yy, ds_u_component, ds_v_component, color="k")

    im = plt.imshow(vorticity.T,
               cmap="seismic",
               vmin=-np.nanmax(np.abs(vorticity)),
               vmax=np.nanmax(np.abs(vorticity)),
               extent=[x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()],
               origin='lower')

    ax.set_xlim([x_coords.min(), x_coords.max()])
    ax.set_ylim([y_coords.min(), y_coords.max()])
    ax.set_title("Ram Head Currents")
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    plt.colorbar(im, cax=cax)
    plt.tight_layout()

def computeVorticity(u, v):
    dudy = np.diff(u, axis=1)[:-1,:]
    dvdx = np.diff(v, axis=0)[:,:-1]
    return (dvdx - dudy)


if __name__ == '__main__':
    main()
