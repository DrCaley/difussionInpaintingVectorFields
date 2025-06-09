#!/usr/bin/env python3

import argparse
import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
import oyaml as yaml
from PIL import Image

from scipy.io import loadmat
from tqdm import tqdm

"""From WHOI. We don't use this"""
def main():
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_path = os.path.dirname(current_script_dir)

    parser = argparse.ArgumentParser(add_help=False)

    default_dataset_path = os.path.join(project_path, "../difussionInpaintingVectorFields/data/rams_head/", "stjohn_hourly_5m_velocity_ramhead_v2.mat")

    parser.add_argument("--dataset", "-d", default=default_dataset_path)

    parser.add_argument("--boundaries", "-b", default="full")

    args = parser.parse_args()

    mat = loadmat(args.dataset)
    boundaries = args.boundaries

    lat = mat['lat'][0, :]
    lon = mat['lon'][:, 0]
    time = mat['ocean_time'].squeeze()

    current_u = mat['u'].transpose((2, 0, 1))  # lat,lon,idx -> idx,lat,lon
    current_v = mat['v'].transpose((2, 0, 1))  # lat,lon,idx -> idx,lat,lon

    with open(os.path.join(project_path, "../difussionInpaintingVectorFields/data/rams_head/", "boundaries.yaml"), 'r') as f:
        boundaries_cfg = yaml.load(f.read(), Loader=yaml.SafeLoader)

    if boundaries == "full":
        boundaries_coords = {"s": lat.min(),
                             "n": lat.max(),
                             "e": lon.max(),
                             "w": lon.min()}

    elif boundaries in boundaries_cfg:
        boundaries_coords = boundaries_cfg[boundaries]

        if boundaries_coords['s'] is None:
            boundaries_coords['s'] = lat.min()

        if boundaries_coords['n'] is None:
            boundaries_coords["n"] = lat.max()

        if boundaries_coords['e'] is None:
            boundaries_coords["e"] = lon.max()

        if boundaries_coords['w'] is None:
            boundaries_coords["w"] = lon.min()

    else:
        raise ValueError(f"Unknown Boundries \"{boundaries}\"")

    w_idx = np.argmin(np.abs(lon - boundaries_coords['w']))
    e_idx = np.argmin(np.abs(lon - boundaries_coords['e']))
    s_idx = np.argmin(np.abs(lat - boundaries_coords['s']))
    n_idx = np.argmin(np.abs(lat - boundaries_coords['n']))

    lon = lon[w_idx:e_idx + 1]
    lat = lat[s_idx:n_idx + 1]
    lonlon, latlat = np.meshgrid(lon, lat)

    current_u = current_u[:, w_idx:e_idx + 1, s_idx:n_idx + 1]
    current_v = current_v[:, w_idx:e_idx + 1, s_idx:n_idx + 1]

    time = [datetime.datetime.fromordinal(int(t)) + datetime.timedelta(days=t % 1) - datetime.timedelta(days=366) for t
            in time]
    dataset = [np.dstack([u, v]) for u, v in zip(current_u, current_v)]
    dataset = [np.ma.array(d, mask=np.isnan(d)) for d in dataset]

    print(f"Loaded {len(time)} model output snapshots")
    print(f"\tStart Time: {np.min(time)}")
    print(f"\tEnd Time: {np.max(time)}")

    lon_spacing = lon[1] - lon[0]
    lat_spacing = lat[1] - lat[0]
    lon_uniform = np.arange(lon[0], lon[-1] + lon_spacing, lon_spacing)
    lat_uniform = np.arange(lat[0], lat[-1] + lat_spacing, lat_spacing)
    lonlon, latlat = np.meshgrid(lon_uniform, lat_uniform)

    for t, u, v in tqdm(zip(time, current_u, current_v), total=len(time)):

        plt.figure(figsize=(16, 16))
        plt.title(f"Ram's Head Currents -- {str(t)}")
        speed = np.sqrt(u ** 2 + v ** 2)
        plt.quiver(lonlon, latlat, u.T, v.T) # color=speed.T, cmap='winter', density=2
        print(u.T)
        plt.imshow(np.isnan(u).T, origin="lower", cmap="winter", extent=(lon.min(), lon.max(), lat.min(), lat.max()),
                   alpha=0.5)
        plt.colorbar(label='Speed (m/s)')


        plt.show()
        plt.close("all")

if __name__ == '__main__':
    main()
