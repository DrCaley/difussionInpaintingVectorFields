Salutations.

You may be wondering why I've called you here.

Your task is to build a neural net capable of inpainting water currents,
using the data set and code base provided.

We didn't test much of this, and our results are dubious so far.

The paper about this will be groundbreaking if you are able to get things working. 
Like Nobel Peace Prize worthy.

Collect data, train some different models with the parameters in the .yaml file,
and - most importantly - have fun!

- Volunteer

---

## Testing

Run the test suite with:
```bash
source env/bin/activate
python -m pytest unit_tests/ -v
```

### Test Coverage

The test suite covers:
- **DDPM**: Forward/backward diffusion, noise scheduling, alpha/beta calculations
- **Noise Strategies**: Gaussian, divergence-free, and HH decomposition noise
- **Loss Functions**: MSE and physical loss strategies with divergence constraints
- **UNet**: Architecture tests for the 64×128 off-the-shelf model
- **Data Loading**: OceanImageDataset, DataLoader compatibility
- **Masks**: Random, coverage, and border mask generators
- **Helmholtz-Hodge Decomposition**: Solenoidal/irrotational field decomposition
- **Integration**: End-to-end DDPM+UNet pipeline tests
- **Edge Cases**: Boundary conditions, device handling, gradient flow

### Tests Not Implemented

The following test categories are **not feasible** in this environment:

1. **ModelInpainter Tests** - The `ModelInpainter` class depends on `tkinter` for GUI visualization. The `tkinter` module requires a display server and Tcl/Tk libraries that are not available in headless or minimal Python environments.

2. **Arbitrary Spatial Dimension Tests** - The off-the-shelf UNet architecture requires fixed 64×128 input dimensions. Ocean data (44×94) is padded to fit. Tests for arbitrary small spatial dimensions are not applicable.

3. **GPU Acceleration Tests** - This machine does not support GPU acceleration (no CUDA/MPS). All tests run on CPU.