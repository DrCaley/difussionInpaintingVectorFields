

from scipy.io import loadmat
import numpy as np
from navpy import lla2ned
import os

import itertools
import pdb
import random

import matplotlib.pyplot as plt

from flow_observation import FlowObservation
from gaussian_process.incompressible_gp.incompressible_gp_model import IncompressibleGP, CompressibleGP
from gaussian_process.simple_gp.simple_gp_model import GPModel_2D


def main():
    project_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../..")
    historical_data = loadmat(os.path.join(project_path, "./data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"))

    lat = historical_data['lat'][0,:]
    lon = historical_data['lon'][:,0]

    x = lon
    y = lat

    latlon = np.array([(foo, bar) for foo, bar in itertools.product(lon, lat)])

    ned_coords = lla2ned(lat=latlon[:,1],
                         lon=latlon[:,0],
                         alt=np.zeros(len(latlon)),
                         lat_ref=np.mean(lat),
                         lon_ref=np.mean(lon),
                         alt_ref=0)

    northing = ned_coords[:, 0]
    easting = ned_coords[:, 1]

    xx = easting.reshape((len(lon), len(lat)))
    yy = northing.reshape((len(lon), len(lat)))

    current_u = historical_data['u'].transpose((2,0,1)) # lat,lon,idx -> idx,lat,lon
    current_v = historical_data['v'].transpose((2,0,1)) # lat,lon,idx -> idx,lat,lon        

    mask = np.isnan(current_u[0])

    dataset = [np.dstack([u, v]) for u, v in zip(current_u, current_v)]

    world_idx = 13969
    # world_idx = 11085
    # world_idx = random.randint(0, 17000)

    uu = dataset[world_idx][:,:,0]
    vv = dataset[world_idx][:,:,1]


    # x = np.arange(-10, 11)
    # y = np.arange(-15, 16)

    # yy, xx = np.meshgrid(y, x)

    # vv = -xx/25
    # uu = yy/25

    # mask = np.zeros(xx.shape,dtype=bool)

    print(f"Ground Truth: max u {np.nanmax(uu):.3f}, min u {np.nanmin(uu):.3f}")
    print(f"Ground Truth: max v {np.nanmax(vv):.3f}, min v {np.nanmin(vv):.3f}")

    cfg = {'lengthscale': 10,
           'kernel_noise': 0.01,
           'observation_noise': 0.00000001}

    gp = IncompressibleGP(xx, yy, cfg)
    rbf_gp = CompressibleGP(xx, yy, cfg)

    # cell_x = np.arange(0, len(x), 2)
    # cell_y = np.ones(cell_x.shape)+10
    # cell_x = [random.randint(0, len(x)-1) for _ in range(10)]
    # cell_y = [random.randint(0, len(y)-1) for _ in range(10)]

    cell_x = [11, 40, 14, 51, 39, 93, 85, 22, 12, 78]
    cell_y = [25, 0, 43, 30, 32, 19, 22, 12, 19, 23]


    print(cell_x)
    print(cell_y)

    for obs_x, obs_y in zip(cell_x, cell_y):

        new_obs = FlowObservation(x = xx[int(obs_x), int(obs_y)],
                                  y = yy[int(obs_x), int(obs_y)],
                                  u = uu[int(obs_x), int(obs_y)],
                                  v = vv[int(obs_x), int(obs_y)])


        if np.isnan(new_obs.u) or np.isnan(new_obs.v):
            continue

        print(new_obs)

        gp.addObs(new_obs)
        rbf_gp.addObs(new_obs)

    mask_x, mask_y = np.where(mask==1)

    for x, y in zip(mask_x-1, mask_y-1):
        gp.addObs(FlowObservation(xx[x,y], yy[x,y], 0, 0))
        rbf_gp.addObs(FlowObservation(xx[x,y], yy[x,y], 0, 0))



    gp.fit()
    rbf_gp.fit()
    gp_mean, gp_cov = gp.predict()
    pred_u = gp_mean[...,0]
    pred_v = gp_mean[...,1]

    rbf_gp_mean, rbf_gp_cov = rbf_gp.predict()
    rbf_pred_u = rbf_gp_mean[...,0]
    rbf_pred_v = rbf_gp_mean[...,1]

    pred_uu = pred_u.reshape(uu.shape)
    pred_vv = pred_v.reshape(vv.shape)

    rbf_pred_uu = rbf_pred_u.reshape(uu.shape)
    rbf_pred_vv = rbf_pred_v.reshape(vv.shape)

    pred_uu[mask] = np.nan
    pred_vv[mask] = np.nan

    rbf_pred_uu[mask] = np.nan
    rbf_pred_vv[mask] = np.nan

    print(f"Predicted: max u {np.nanmax(pred_uu):.3f}, min u {np.nanmin(pred_uu):.3f}")
    print(f"Predicted: max v {np.nanmax(pred_vv):.3f}, min v {np.nanmin(pred_vv):.3f}")

    print(f"RBF Predicted: max u {np.nanmax(rbf_pred_uu):.3f}, min u {np.nanmin(rbf_pred_uu):.3f}")
    print(f"RBF Predicted: max v {np.nanmax(rbf_pred_vv):.3f}, min v {np.nanmin(rbf_pred_vv):.3f}")

    stride = 2
    
    x_cell_size = np.mean(np.diff(xx[:,0]))
    y_cell_size = np.mean(np.diff(yy[0]))

    max_mag = np.max([np.nanmax(np.abs(rbf_pred_uu)), np.nanmax(np.abs(rbf_pred_vv))])

    plt.figure()
    ax = plt.subplot(121)
    plt.title("RBF U")
    plt.imshow(rbf_pred_uu.T, origin="lower", cmap="coolwarm", vmin=-max_mag, vmax=max_mag)
    plt.scatter(cell_x, cell_y, marker="x", c='r')
    ax = plt.subplot(122)
    plt.title("RBF V")
    plt.imshow(rbf_pred_vv.T, origin="lower", cmap="coolwarm", vmin=-max_mag, vmax=max_mag)
    plt.scatter(cell_x, cell_y, marker="x", c='r')
    
    xx = xx[::stride,::stride]
    yy = yy[::stride,::stride]
    uu = uu[::stride,::stride]
    vv = vv[::stride,::stride]
    pred_uu = pred_uu[::stride,::stride]
    pred_vv = pred_vv[::stride,::stride]
    rbf_pred_uu = rbf_pred_uu[::stride,::stride]
    rbf_pred_vv = rbf_pred_vv[::stride,::stride]

    plt.figure(figsize=(8,5))
    plt.title(f"Ground Truth idx={world_idx}")
    plt.imshow(mask.T, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap="winter")
    plt.quiver(xx, yy, uu, vv, color="white", scale=10)

    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")

    plt.scatter([o.x for o in gp.observations], [o.y for o in gp.observations], marker="x", c='r', label="Observation")
    plt.legend()

    plt.figure(figsize=(8,5))
    plt.title("Incompressible Predicted")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")

    plt.quiver(xx, yy, pred_uu, pred_vv, color='white')
    # plt.quiver(xx, yy, pred_uu, pred_vv, color='white', scale=10)
    plt.imshow(mask.T, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap="winter")

    plt.figure(figsize=(8,5))
    plt.title("RBF Predicted")

    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.quiver(xx, yy, rbf_pred_uu, rbf_pred_vv, color='white', scale=10)
    plt.imshow(mask.T, origin="lower", extent=[xx.min(), xx.max(), yy.min(), yy.max()], cmap="winter")

    plt.show()
    pdb.set_trace()



if __name__ == '__main__':
    main()