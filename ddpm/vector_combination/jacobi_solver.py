"""Stub for jacobi_solver — original file was removed.

The analytical_seam_projection function is imported by vector_combiner.py
but is not used in the current inference pipeline (repaint_gp_init_adaptive).
This stub prevents ImportError while keeping the import chain intact.
"""

def analytical_seam_projection(known, inpainted, mask):
    raise NotImplementedError(
        "analytical_seam_projection was removed. "
        "Use naive blending or the combination UNet instead."
    )
