import os
import pickle
import random
import re

class QueryNoise:
    def __init__(self, noise_dir):
        self.noise_dir = noise_dir
        self.cache = []
        self.cache_range = (None, None)  # (start_t, end_t)

    def _find_file_for_timestep(self, t: int):
        """Find the pickle file covering the given timestep."""
        for fname in os.listdir(self.noise_dir):
            match = re.match(r"(\d+)-(\d+)\.pickle", fname)
            if match:
                start, end = map(int, match.groups())
                if start <= t + 1 <= end:  # +1 since filenames are 1-based
                    return os.path.join(self.noise_dir, fname), start
        raise FileNotFoundError(f"No pickle file found containing timestep {t + 1}")

    def _load_if_needed(self, t: int):
        """Load correct pickle file if t isn't in current cache."""
        start, end = self.cache_range
        if start is None or not (start <= t + 1 <= end):  # +1 since filenames are 1-based
            file_path, file_start = self._find_file_for_timestep(t)
            with open(file_path, 'rb') as f:
                self.cache = pickle.load(f)
                self.cache_range = (file_start, file_start + len(self.cache) - 1)

    def get(self, t: int):
        self._load_if_needed(t)
        local_index = t - (self.cache_range[0] - 1)  # convert global t to index in cache
        return random.choice(self.cache[local_index])
