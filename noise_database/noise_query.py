import os.path
import pickle
from collections import OrderedDict

base_dir = os.path.dirname(os.path.abspath(__file__))

class QueryNoise:
    def __init__(self):
        self.cache = OrderedDict()
        pass

    def get(self, t):
        return self._fetch_noise(t)

    def set(self, key: int, value):
        self.cache[key] = value
        while len(self.cache) > 1000:
            self.cache.popitem(last=False)

    def delete(self, key):
        del self.cache[key]

    def _fetch_noise(self, t):
        noise = self.cache.get(t)

        if noise :
            return noise

        else:
            for i in range(100):
                key = t - i
                if key < 0:
                    break
                noise = self._fetch_noise_from_pickle(i)
                self.set(key, noise)
            return noise

    def _fetch_noise_from_pickle(self, t):
        path = f"./div_free_noise_{t}"
        if not os.path.exists(path) :
            raise f"Noise {t} does not exist"
        with open(path, 'rb') as f:
            noise = pickle.load(f)
        return noise
