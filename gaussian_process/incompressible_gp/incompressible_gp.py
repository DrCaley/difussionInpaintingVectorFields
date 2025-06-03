from PIL import Image
from matplotlib.colors import ListedColormap
from scipy.io import loadmat
import numpy as np
from navpy import lla2ned
import os
import itertools
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from DataPrep.ocean_image_dataset import OceanImageDataset
from flow_observation import FlowObservation
from gaussian_process.incompressible_gp.incompressible_gp_model import IncompressibleGP, CompressibleGP

def main():
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
    historical_data = loadmat(os.path.join(project_path, "./data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"))

    lat = historical_data['lat'][0, :]
    lon = historical_data['lon'][:, 0]

    land_color = 'blue'
    ocean_color = '#90EE90'
    custom_cmap = ListedColormap(['blue', '#90EE90'])

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

    world_idx = 5
    uu = dataset[world_idx][:, :, 0]
    vv = dataset[world_idx][:, :, 1]

    print(f"Ground Truth: max u {np.nanmax(uu):.3f}, min u {np.nanmin(uu):.3f}")
    print(f"Ground Truth: max v {np.nanmax(vv):.3f}, min v {np.nanmin(vv):.3f}")

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

    stride = 2
    xx = xx[::stride, ::stride]
    yy = yy[::stride, ::stride]
    uu = uu[::stride, ::stride]
    vv = vv[::stride, ::stride]
    rbf_pred_uu = rbf_pred_uu[::stride, ::stride]
    rbf_pred_vv = rbf_pred_vv[::stride, ::stride]

    plt.figure(figsize=(8, 5))
    plt.title(f"Ground Truth - 1/23/2021 06:00:00 ")
    plt.imshow(mask.T, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap=custom_cmap)
    plt.quiver(xx, yy, uu, vv, color="white", scale=10)
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.scatter([o.x for o in gp.observations], [o.y for o in gp.observations], marker="x", c='r', label="Observation")
    plt.legend()

    plt.figtext(0.5, -0.05, 'Figure 1: Ground Truth with Observations and Vector Field', ha='center', fontsize=12)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.title("Gaussian Process (GP) Predicted")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.quiver(xx, yy, rbf_pred_uu, rbf_pred_vv, color='white', scale=10)
    plt.imshow(mask.T, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap=custom_cmap)

    plt.show()

if __name__ == '__main__':
    main()
