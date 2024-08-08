import pandas as pd
from PIL import Image
from matplotlib.colors import ListedColormap
from scipy.io import loadmat
import numpy as np
import os
import itertools
import matplotlib.pyplot as plt
from flow_observation import FlowObservation
from gaussian_process.incompressible_gp.incompressible_gp_model import IncompressibleGP, CompressibleGP
import csv

def calculate_mse(true_u, true_v, pred_u, pred_v, mask):
    mse_u = np.nanmean((true_u[~mask] - pred_u[~mask]) ** 2)
    mse_v = np.nanmean((true_v[~mask] - pred_v[~mask]) ** 2)
    return mse_u, mse_v

def lla2ned(lat, lon, alt, lat_ref, lon_ref, alt_ref):
    a = 6378137.0  # WGS-84 Earth semimajor axis (m)
    f = 1 / 298.257223563  # WGS-84 flattening
    e2 = 2 * f - f ** 2  # Square of eccentricity

    lat = np.radians(lat)
    lon = np.radians(lon)
    lat_ref = np.radians(lat_ref)
    lon_ref = np.radians(lon_ref)

    sin_lat = np.sin(lat_ref)
    cos_lat = np.cos(lat_ref)
    N = a / np.sqrt(1 - e2 * sin_lat ** 2)

    d_lat = lat - lat_ref
    d_lon = lon - lon_ref

    d_north = (N + alt_ref) * d_lat
    d_east = (N + alt_ref) * cos_lat * d_lon
    d_down = alt_ref - alt

    return np.vstack((d_north, d_east, d_down)).T

def main():
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
    historical_data = loadmat(os.path.join(project_path, "./data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"))

    lat = historical_data['lat'][0, :]
    lon = historical_data['lon'][:, 0]

    land_color = 'blue'
    ocean_color = '#90EE90'
    custom_cmap = ListedColormap([land_color, ocean_color])

    latlon = np.array([(foo, bar) for foo, bar in itertools.product(lon, lat)])
    ned_coords = lla2ned(lat=latlon[:, 1], lon=latlon[:, 0], alt=np.zeros(len(latlon)),
                         lat_ref=np.mean(lat), lon_ref=np.mean(lon), alt_ref=0)

    northing = ned_coords[:, 0]
    easting = ned_coords[:, 1]
    xx = easting.reshape((len(lon), len(lat)))
    yy = northing.reshape((len(lon), len(lat)))

    current_u = historical_data['u'].transpose((2, 0, 1))  # lat, lon, idx -> idx, lat, lon
    current_v = historical_data['v'].transpose((2, 0, 1))  # lat, lon, idx -> idx, lat, lon

    mask = np.isnan(current_u[0])

    ran_mask_path = '../images/0_mask_random_path40_s_0.png'
    land_mask_path = '../images/land_mask_cropped.png'
    ran_mask_image = Image.open(ran_mask_path).convert('L')
    land_mask_image = Image.open(land_mask_path).convert('L')
    ran_mask = np.array(ran_mask_image, dtype=np.bool_)
    land_mask = np.array(land_mask_image, dtype=np.bool_)

    combined_mask = np.logical_xor(ran_mask, land_mask)
    combined_mask = combined_mask.T
    combined_mask = np.fliplr(combined_mask)

    dataset = [np.dstack([u, v]) for u, v in zip(current_u, current_v)]

    # 100 world test indexes
    world_indexes = [5, 425, 805, 1406, 1622, 1790, 2013, 2637, 3512, 3543, 3623, 3692, 3828, 3876, 3927, 3996,
                     4613, 4635, 4729, 5194, 5244, 5389, 5607, 5732, 5924, 6030, 6148, 6231, 6327, 6433, 6556,
                     7127, 7261, 7284, 7387, 7631, 7747, 7987, 8047, 8115, 8298, 8670, 8754, 8788, 9035, 9135,
                     9220, 9309, 9447, 9479, 9566, 9658, 9670, 10098, 10229, 10307, 10372, 10426, 10609, 10864,
                     11091, 11201, 11351, 11371, 11421, 11523, 11562, 11568, 12143, 12162, 12307, 12443, 12568,
                     12576, 12692, 13051, 13299, 13742, 14037, 14038, 14220, 14536, 14577, 14846, 15032, 15138,
                     15287, 15685, 16032, 16076, 16315, 16374, 16748, 16750, 16952]

    mse_results = []

    for world_idx in world_indexes:
        uu = dataset[world_idx][:, :, 0]
        vv = dataset[world_idx][:, :, 1]

        print(f"World Index {world_idx}: max u {np.nanmax(uu):.3f}, min u {np.nanmin(uu):.3f}")
        print(f"World Index {world_idx}: max v {np.nanmax(vv):.3f}, min v {np.nanmin(vv):.3f}")

        cfg = {'lengthscale': 10, 'kernel_noise': 0.01, 'observation_noise': 0.00000001}
        gp = IncompressibleGP(xx, yy, cfg)
        rbf_gp = CompressibleGP(xx, yy, cfg)

        # Observation indexes
        # These should be the indexes where combined_mask is true
        indexes = np.where(combined_mask)

        cell_x = indexes[0].tolist()
        cell_y = indexes[1].tolist()

        for obs_x, obs_y in zip(cell_x, cell_y):
            new_obs = FlowObservation(x=xx[int(obs_x), int(obs_y)], y=yy[int(obs_x), int(obs_y)],
                                      u=uu[int(obs_x), int(obs_y)], v=vv[int(obs_x), int(obs_y)])
            if np.isnan(new_obs.u) or np.isnan(new_obs.v):
                continue
            gp.addObs(new_obs)
            rbf_gp.addObs(new_obs)

        mask_x, mask_y = np.where(mask == 1)

        for x, y in zip(mask_x, mask_y):
            if x < xx.shape[0] and y < yy.shape[1]:
                gp.addObs(FlowObservation(xx[x, y], yy[x, y], 0, 0))
                rbf_gp.addObs(FlowObservation(xx[x, y], yy[x, y], 0, 0))

        gp.fit()
        rbf_gp.fit()
        gp_mean, gp_cov = gp.predict()
        pred_u = gp_mean[..., 0]
        pred_v = gp_mean[..., 1]
        rbf_gp_mean, rbf_gp_cov = rbf_gp.predict()
        rbf_pred_u = rbf_gp_mean[..., 0]
        rbf_pred_v = rbf_gp_mean[..., 1]

        pred_uu = pred_u.reshape(uu.shape)
        pred_vv = pred_v.reshape(vv.shape)
        rbf_pred_uu = rbf_pred_u.reshape(uu.shape)
        rbf_pred_vv = rbf_pred_v.reshape(vv.shape)
        pred_uu[mask] = np.nan
        pred_vv[mask] = np.nan
        rbf_pred_uu[mask] = np.nan
        rbf_pred_vv[mask] = np.nan

        mse_u, mse_v = calculate_mse(uu, vv, pred_uu, pred_vv, mask)
        mse_results.append({'world_idx': world_idx, 'mse_u': mse_u, 'mse_v': mse_v})

        stride = 2
        xx_stride = xx[::stride, ::stride]
        yy_stride = yy[::stride, ::stride]
        uu_stride = uu[::stride, ::stride]
        vv_stride = vv[::stride, ::stride]
        rbf_pred_uu_stride = rbf_pred_uu[::stride, ::stride]
        rbf_pred_vv_stride = rbf_pred_vv[::stride, ::stride]

        xmin, xmax = xx.min(), xx.max()
        ymin, ymax = yy.min(), yy.max()

        plt.figure(figsize=(8, 5))
        plt.title(f"Ground Truth - World Index {world_idx}")
        plt.imshow(mask.T, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap=custom_cmap)
        plt.quiver(xx_stride, yy_stride, uu_stride, vv_stride, color="white", scale=10)
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.scatter([o.x for o in gp.observations], [o.y for o in gp.observations], marker="x", c='r', label="Observation")
        plt.legend()

        plt.figtext(0.5, -0.05, f'Figure: Ground Truth with Observations and Vector Field for World Index {world_idx}', ha='center', fontsize=12)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(8, 5))
        plt.title(f"Gaussian Process (GP) Predicted - World Index {world_idx}")
        plt.xlabel("Easting (m)")
        plt.ylabel("Northing (m)")
        plt.quiver(xx_stride, yy_stride, rbf_pred_uu_stride, rbf_pred_vv_stride, color='white', scale=10)
        plt.imshow(mask.T, origin="lower", extent=[xmin, xmax, ymin, ymax], cmap=custom_cmap)
        plt.show()

    # with open('mse_results.csv', 'w', newline='') as csvfile:
    #     fieldnames = ['world_idx', 'mse_u', 'mse_v']
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #
    #     writer.writeheader()
    #     for row in mse_results:
    #         writer.writerow(row)

if __name__ == '__main__':
    main()
