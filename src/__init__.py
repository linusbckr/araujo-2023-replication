"""
Standalone replication of Araujo et al. (2023), PNAS 120(46) e2312451120.

Three modules, one concern each:

    matrices     the spatial weight matrices W_t^[k] = shell_k AND C_t, their
                 time averages, and the Omega^(Delta+1) margins
    data_loader  the synthetic demo panel, the full-mode checkpoint reader, and
                 the LAI gap-fill / season-matched baseline
    estimation   FD-2SLS by thin QR + SVD with pixel-clustered errors, the exact
                 count/composition decomposition, and the two specifications

Entry point is ``run_replication.py`` in the parent directory.
"""

__all__ = ["matrices", "data_loader", "estimation"]
__version__ = "1.0.0"
