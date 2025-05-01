import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from .plotting.plot_rough_surface import plot_rough_surface_debug


def gaussian_correlated_surface(
    nx: int,
    ny: int,
    dx: float,
    corr_len_x: float,
    corr_len_y: float,
    height_rms: float,
    seed: int = None,
    debug_plots: bool = False,
) -> np.ndarray:
    """
    Create a 2D surface height array with Gaussian autocorrelation.

    Args:
        nx (int): Number of sample points in X direction
        ny (int): Number of sample points in Y direction
        dx (float): Distance between samples (assumed square pixels) [m]
        corr_len_x (float): Correlation length in X direction [m]
        corr_len_y (float): Correlation length in Y direction [m]
        height_rms (float): RMS height of the surface [m]
        seed (int): Optional seed for random number generator
        debug_plots (bool): Create debug plots

    Returns:
        np.ndarray: Numpy array of surface height with shape (ny, nx).
    """
    # Create a Gaussian kernel
    kernel_size_x = int(6.0 * round(corr_len_x / dx))
    kernel_size_y = int(6.0 * round(corr_len_y / dx))
    kernel_x = np.arange(-kernel_size_x // 2, kernel_size_x // 2) * dx
    kernel_y = np.arange(-kernel_size_y // 2, kernel_size_y // 2) * dx
    xx, yy = np.meshgrid(kernel_x, kernel_y)
    kernel = np.exp(-(xx**2 / corr_len_x**2 + yy**2 / corr_len_y**2))

    # Generate white noise
    rng = np.random.default_rng(seed=seed)
    noise_shape = (ny + kernel_size_y - 1, nx + kernel_size_x - 1)
    white_noise = rng.normal(0, 1, size=noise_shape)

    # Convolve the noise with the kernel
    gauss_corr = scipy.signal.fftconvolve(white_noise, kernel, mode="valid")

    # Normalize the data
    rough_surf = gauss_corr - np.mean(gauss_corr)
    rough_surf /= np.std(rough_surf)
    rough_surf *= height_rms

    if debug_plots:
        plot_rough_surface_debug(
            nx, ny, dx, corr_len_x, corr_len_y, height_rms, rough_surf, kernel
        )
        plt.show()

    return rough_surf
