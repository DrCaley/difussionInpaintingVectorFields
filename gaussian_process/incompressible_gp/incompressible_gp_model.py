import numpy as np

def rbfKernel(x1, x2, lengthscale, kernel_noise):
    gamma = 1 / lengthscale**2
    dist = np.linalg.norm(x1 - x2[:, np.newaxis], axis=2)
    return (kernel_noise**2)*np.exp(-0.5*gamma*dist)


def incompressibleRBFKernel(x1, x2, lengthscale, kernel_noise):
    gamma = 1 / lengthscale**2

    dist = np.linalg.norm(x1 - x2[:, np.newaxis], axis=2)
    kse = (kernel_noise**2)*np.exp(-0.5*gamma*dist)

    dxx = -(gamma**2) * (x1[:,0] - x2[:,0,np.newaxis]) * (x1[:,0] - x2[:,0,np.newaxis]) * kse
    dxy = -(gamma**2) * (x1[:,0] - x2[:,0,np.newaxis]) * (x1[:,1] - x2[:,1,np.newaxis]) * kse
    dyy = -(gamma**2) * (x1[:,1] - x2[:,1,np.newaxis]) * (x1[:,1] - x2[:,1,np.newaxis]) * kse

    res = np.zeros(2*np.array(dyy.shape))*np.nan

    # K(x,x') = [[dxx, dxy]
    #            [dxy, dyy]]

    res[::2,::2] = dxx
    res[1::2,::2] = dxy
    res[::2,1::2] = dxy
    res[1::2,1::2] = dyy

    return res

class IncompressibleGP(object):
    def __init__(self, qxx, qyy, cfg):
        self.observations = []
        self.original_shape = qxx.shape
        # qxx = [[x_(0,0), x_(1,0), ... x_(m,0)]
        #        [x_(0,1), x_(1,1), ... x_(m,1)]
        #         ...
        #        [x_(0,n), x_(1,n), ... x_(m-0)]]

        # qyy = [[y_(0,0), y_(1,0), ... y_(m,0)]
        #        [y_(0,1), y_(1,1), ... y_(m,1)]
        #         ...
        #        [y_(0,n), y_(1,n), ... y_(m-0)]]

        # self.qx = [x_(0,0), y_(0,0), x_(0,1), x_(0,1), ..., x_(0,n), x_(0,n), x_(1,0), x_(1,0), ...,  x_(m,n), y_(m,n)]

        self.qx = np.vstack([qxx.flatten(), qyy.flatten()]).T.flatten()

        self.lengthscale = cfg['lengthscale']
        self.kernel_noise = cfg['kernel_noise']
        self.observation_noise = cfg['observation_noise']

    def fit(self):
        self.obs_locs = np.vstack([o.loc for o in self.observations])

        self.obs_data = np.hstack([o.data for o in self.observations])

        self.kdd = incompressibleRBFKernel(self.obs_locs, self.obs_locs, self.lengthscale, self.kernel_noise)
        self.kdd += np.eye(self.kdd.shape[0]) * self.observation_noise
        self.kdd_inv = np.linalg.inv(self.kdd)

    def predict(self):
        kdq = incompressibleRBFKernel(self.obs_locs, self.qx.reshape([-1, 2]), self.lengthscale, self.kernel_noise)
        kqq = incompressibleRBFKernel(self.qx.reshape([-1, 2]), self.qx.reshape([-1, 2]), self.lengthscale,
                                      self.kernel_noise)

        predicted_mean = np.matmul(np.matmul(kdq, self.kdd_inv), self.obs_data).reshape([-1, 2])

        predicted_u = predicted_mean[:, 0].reshape(self.original_shape)
        predicted_v = predicted_mean[:, 1].reshape(self.original_shape)

        predicted_cov = kqq - np.matmul(np.matmul(kdq, self.kdd_inv), kdq.T)

        return np.dstack([predicted_u, predicted_v]), predicted_cov

    def addObs(self, new_obs):
        self.observations.append(new_obs)

    def update(self, new_obs):
        self.addObs(new_obs)
        self.fit()


class CompressibleGP(object):
    def __init__(self, qxx, qyy, cfg):
        self.observations = []

        self.original_shape = qxx.shape

        self.qx = np.vstack([qxx.flatten(), qyy.flatten()]).T

        self.lengthscale = cfg['lengthscale']
        self.kernel_noise = cfg['kernel_noise']
        self.observation_noise = cfg['observation_noise']

    def fit(self):
        self.obs_locs = np.vstack([o.loc for o in self.observations])
        self.obs_u_data = np.hstack([o.u for o in self.observations])
        self.obs_v_data = np.hstack([o.v for o in self.observations])

        self.kdd = rbfKernel(self.obs_locs, self.obs_locs, self.lengthscale, self.kernel_noise)
        self.kdd += np.eye(self.kdd.shape[0]) * self.observation_noise
        self.kdd_inv = np.linalg.inv(self.kdd)

    def predict(self):
        kdq = rbfKernel(self.obs_locs, self.qx.reshape([-1, 2]), self.lengthscale, self.kernel_noise)
        kqq = rbfKernel(self.qx.reshape([-1, 2]), self.qx.reshape([-1, 2]), self.lengthscale, self.kernel_noise)

        predicted_u = np.matmul(np.matmul(kdq, self.kdd_inv), self.obs_u_data).reshape(self.original_shape)
        predicted_v = np.matmul(np.matmul(kdq, self.kdd_inv), self.obs_v_data).reshape(self.original_shape)
        predicted_cov = kqq - np.matmul(np.matmul(kdq, self.kdd_inv), kdq.T)

        return np.dstack([predicted_u, predicted_v]), predicted_cov

    def addObs(self, new_obs):
        self.observations.append(new_obs)

    def update(self, new_obs):
        self.addObs(new_obs)
        self.fit()
