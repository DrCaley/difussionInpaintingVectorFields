import datetime
import numpy as np

class FlowObservation(object):
    """docstring for CurrentObs"""
    def __init__(self, x, y, u, v, t=None):
        self.x = x
        self.y = y
        self.t = t
        self.u = u
        self.v = v

        self.loc = np.array([x, y])
        self.data = np.array([u, v])

    def __str__(self):
        if self.t is not None:
            return f"X: {self.x:.3f}, Y: {self.y:.3f}, U: {self.u:.3f}, V: {self.v:.3f}, T:{str(datetime.datetime.fromtimestamp(t))}"
        else:
            return f"X: {self.x:.3f}, Y: {self.y:.3f}, U: {self.u:.3f}, V: {self.v:.3f}"
