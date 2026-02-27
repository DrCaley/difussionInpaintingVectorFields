from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox
from scipy.io import loadmat
import h5py
import pickle

# ── Select which dataset to load: 'mat' | 'h5' | 'pickle' ──────────────────
SOURCE = 'pickle'
PICKLE_PATH = "data/stjohn_subset_ocean.pkl"   # path to your pickle file

# 44 x 94
def load_mat():
    file_path = "data/rams_head/stjohn_hourly_5m_velocity_ramhead_v2.mat"
    mat_data = loadmat(file_path)
    
    u = np.transpose(mat_data['u'], (1, 0, 2))   # (W, H, T) → (H, W, T)
    v = np.transpose(mat_data['v'], (1, 0, 2))
    lat = mat_data['lat'].T                        # (W, H) → (H, W)
    lon = mat_data['lon'].T

    H, W, T = u.shape
    print(f"Shape: {H} x {W} x {T}")
    return u, v, lat, lon, T
# 242 x 329
def load_h5():
    file_path = "data/stjohn_hourly_surface_velocity_20250718.mat"
    with h5py.File(file_path, "r") as f:
        print("All keys and their shapes:")
        for key in f.keys():
            data = f[key]
            try:
                shape = data.shape
            except AttributeError:
                shape = "Unknown / not a dataset"
            print(f"{key} : shape = {shape}")
        # Example: read u and v
        u_raw = np.array(f['us'])
        v_raw = np.array(f['vs'])
        lat = np.array(f['lat_rho'])
        lon = np.array(f['lon_rho'])
        u = np.transpose(u_raw, (1, 2, 0))
        v = np.transpose(v_raw, (1, 2, 0))

    
    T = u.shape[2]
    H, W = u.shape[:2]
    print(f"Shape: {H} x {W} x {T}")
    return u, v, lat, lon, T


def load_pickle(file_path=PICKLE_PATH):
    """Load a pickle file produced by create_subset.py.

    Expects either:
      - a dict with keys 'u'/'us', 'v'/'vs', and optionally 'lat'/'lat_rho', 'lon'/'lon_rho'
      - a numpy array of shape (H, W, T, 2) or (2, T, H, W) etc.
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    if isinstance(data, dict):
        print(f"Pickle keys: {list(data.keys())}")

        # Resolve u
        u_raw = data.get('u', data.get('us', None))
        v_raw = data.get('v', data.get('vs', None))
        if u_raw is None or v_raw is None:
            raise KeyError(f"Could not find u/v arrays in pickle keys: {list(data.keys())}")

        # Pickle stores raw (T, H, W) — transpose to (H, W, T)
        u = np.transpose(u_raw, (1, 2, 0))
        v = np.transpose(v_raw, (1, 2, 0))

        lat_raw = data.get('lat', data.get('lat_rho', None))
        lon_raw = data.get('lon', data.get('lon_rho', None))

    elif isinstance(data, np.ndarray):
        raise ValueError(
            f"Raw array shape {data.shape} — update load_pickle() for your specific layout."
        )
    else:
        raise TypeError(f"Unexpected pickle type: {type(data)}")

    H, W, T = u.shape
    print(f"Shape: {H} x {W} x {T}")

    # Build synthetic lat/lon grids if not present
    if lat_raw is None or lon_raw is None:
        print("No lat/lon found in pickle — using index grids.")
        lon_raw, lat_raw = np.meshgrid(np.arange(W), np.arange(H))
    elif lat_raw.ndim == 1:
        lon_raw, lat_raw = np.meshgrid(lon_raw, lat_raw)

    return u, v, lat_raw, lon_raw, T


# ── Load selected dataset ─────────────────────────────────────────────────────
if SOURCE == 'mat':
    u, v, lat, lon, T = load_mat()
elif SOURCE == 'h5':
    u, v, lat, lon, T = load_h5()
elif SOURCE == 'pickle':
    u, v, lat, lon, T = load_pickle()
else:
    raise ValueError(f"Unknown SOURCE '{SOURCE}'. Choose 'mat', 'h5', or 'pickle'.")

step = 2  # downsample vectors to reduce clutter
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.22)

def draw_frame(t):
    ax.clear()

    U = u[:, :, t]
    V = v[:, :, t]
    mag = np.sqrt(U**2 + V**2)
    land_mask = np.isnan(U)

    cmap = ListedColormap(['white', 'green'])
    ax.imshow(
        land_mask,
        cmap=cmap,
        origin='lower',
        extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        zorder=0
    )
    # Downsample vectors
    U_ds = U[::step, ::step]
    V_ds = V[::step, ::step]
    lat_ds = lat[::step, ::step]
    lon_ds = lon[::step, ::step]
    mag_ds = mag[::step, ::step]

    # Quiver arrows
    ax.quiver(
        lon_ds, lat_ds, U_ds, V_ds, mag_ds,
        cmap="viridis",
        scale=None,
        width=0.003,
        headwidth=3,
        headlength=4,
        alpha=0.9,
        zorder=2
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Vector Field — Time {t}")
    ax.set_aspect('equal', adjustable='box')

    # Reserve space for the hover annotation
    ax._index_label = ax.text(
        0.01, 0.99, "", transform=ax.transAxes,
        va='top', ha='left', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7),
        zorder=5
    )

# Initial draw
draw_frame(0)

# Mouse hover — show nearest [row, col] index
def on_mouse_move(event):
    if event.inaxes is not ax:
        return
    # Find nearest grid point to cursor (lon, lat)
    dist = (lon - event.xdata)**2 + (lat - event.ydata)**2
    row, col = np.unravel_index(np.argmin(dist), dist.shape)
    label = getattr(ax, '_index_label', None)
    if label:
        label.set_text(f"row={row}  col={col}\nlat={lat[row,col]:.4f}  lon={lon[row,col]:.4f}")
        fig.canvas.draw_idle()

fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)

# Slider
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
slider = Slider(ax_slider, 'Time', 0, T - 1, valinit=0, valstep=1)

def slider_update(val):
    t = int(slider.val)
    textbox.set_val(str(t))
    draw_frame(t)
    fig.canvas.draw_idle()

slider.on_changed(slider_update)

# Textbox
ax_text = plt.axes([0.4, 0.05, 0.2, 0.04])
textbox = TextBox(ax_text, "Go to:", initial="0")

def text_submit(text):
    try:
        t = int(text)
        if 0 <= t < T:
            slider.set_val(t)
    except:
        pass

textbox.on_submit(text_submit)

plt.show()