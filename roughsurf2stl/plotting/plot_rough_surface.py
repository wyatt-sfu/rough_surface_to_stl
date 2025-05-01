import matplotlib.pyplot as plt
import numpy as np
import scipy.signal
import scipy.optimize


def plot_rough_surface_debug(
    nx, ny, dx, corr_len_x, corr_len_y, height_rms, rough_surf, kernel
):
    """Validation plots for validating the Gaussian rough surface generation."""
    surf_rms_height = np.sqrt(np.mean(rough_surf**2))
    print(
        f"Specified rms height: {height_rms}, "
        f"measured rms height: {surf_rms_height}"
    )

    # Measured autocorrelation
    autocorr_meas = scipy.signal.correlate(
        rough_surf / height_rms, rough_surf / height_rms, mode="same"
    )
    autocorr_meas /= nx * ny

    # Theoretical autocorrelation
    x_vals = np.arange(-nx // 2, nx // 2) * dx
    y_vals = np.arange(-ny // 2, ny // 2) * dx
    xx, yy = np.meshgrid(x_vals, y_vals)
    autocorr_theory = np.exp(-0.5 * (xx**2 / corr_len_x**2 + yy**2 / corr_len_y**2))

    # Fit a Gaussian function to the autocorrelation
    def gauss_func(x, sigma):
        return np.exp(-0.5 * x**2 / sigma**2)

    x_fit_bounds = (
        (nx // 2) - int(corr_len_x / dx) * 2,
        (nx // 2) + int(corr_len_x / dx) * 2,
    )
    autocorr_x_profile = autocorr_meas[ny // 2, x_fit_bounds[0] : x_fit_bounds[1]]
    res = scipy.optimize.curve_fit(
        gauss_func,
        x_vals[x_fit_bounds[0] : x_fit_bounds[1]],
        autocorr_x_profile,
        p0=[corr_len_x],
    )
    estimated_corr_len_x = res[0][0]
    print(f"Corr len x: {corr_len_x}, measured: {estimated_corr_len_x}")

    y_fit_bounds = (
        (ny // 2) - int(corr_len_y / dx) * 2,
        (ny // 2) + int(corr_len_y / dx) * 2,
    )
    autocorr_y_profile = autocorr_meas[y_fit_bounds[0] : y_fit_bounds[1], nx // 2]
    res = scipy.optimize.curve_fit(
        gauss_func,
        y_vals[y_fit_bounds[0] : y_fit_bounds[1]],
        autocorr_y_profile,
        p0=[corr_len_y],
    )
    estimated_corr_len_y = res[0][0]
    print(f"Corr len y: {corr_len_y}, measured: {estimated_corr_len_y}")

    autocorr_fit_x = np.exp(-0.5 * x_vals**2 / estimated_corr_len_x**2)
    autocorr_fit_y = np.exp(-0.5 * y_vals**2 / estimated_corr_len_y**2)

    # Plot all of the above
    fig, ((ax_kern, ax_rough, ax_corr), (ax_corr_x, ax_corr_y, ax_corr_valid)) = (
        plt.subplots(2, 3, constrained_layout=True, figsize=(10, 5))
    )

    im_kern = ax_kern.imshow(kernel)
    ax_kern.set_xlabel("X")
    ax_kern.set_ylabel("Y")
    ax_kern.set_title("Gaussian kernel")
    fig.colorbar(im_kern, ax=ax_kern)

    im_rough = ax_rough.imshow(rough_surf)
    ax_rough.set_xlabel("X")
    ax_rough.set_ylabel("Y")
    ax_rough.set_title("Rough surface")
    fig.colorbar(im_rough, ax=ax_rough)

    im_corr = ax_corr.imshow(autocorr_meas)
    ax_corr.set_xlabel("X")
    ax_corr.set_ylabel("Y")
    ax_corr.set_title("Rough surface autocorrelation")
    fig.colorbar(im_corr, ax=ax_corr)

    ax_corr_x.plot(x_vals, autocorr_meas[ny // 2, :], label="Actual")
    ax_corr_x.plot(x_vals, autocorr_theory[ny // 2, :], label="Theory")
    ax_corr_x.plot(x_vals, autocorr_fit_x, label="Fit")
    ax_corr_x.legend()
    ax_corr_x.grid(True)
    ax_corr_x.set_title("Correlation comparison (x axis)")
    ax_corr_x.set_xlabel("X Shift (m)")

    ax_corr_y.plot(y_vals, autocorr_meas[:, nx // 2], label="Actual")
    ax_corr_y.plot(y_vals, autocorr_theory[:, nx // 2], label="Theory")
    ax_corr_y.plot(y_vals, autocorr_fit_y, label="Fit")
    ax_corr_y.legend()
    ax_corr_y.grid(True)
    ax_corr_y.set_title("Correlation comparison (y axis)")
    ax_corr_y.set_xlabel("Y Shift (m)")

    im_corr_valid = ax_corr_valid.imshow(autocorr_meas - autocorr_theory)
    ax_corr_valid.set_xlabel("X")
    ax_corr_valid.set_ylabel("Y")
    ax_corr_valid.set_title("Difference between theoretical and actual")
    fig.colorbar(im_corr_valid, ax=ax_corr_valid)
