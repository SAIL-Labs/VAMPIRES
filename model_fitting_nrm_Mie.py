import pickle
from make_dust_mie import geometric_star
import corner
import math
import emcee
import matplotlib.pyplot as plt
import shutil
from jax import numpy as np
import numpy as onp
import equinox as eqx
import sys
sys.path.append('../all_projects/')
import useful_functions as uf
import os
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
from jax import devices
import jax
from scipy import stats
import statistics
from jax.tree_util import tree_map
import interpax as ipx
from scipy.ndimage import rotate
import time
from jax import vmap

import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")
# import tensorflow as tf
#
# gpu_mem_limit = True
#
# if gpu_mem_limit:
#     gpus = tf.config.list_physical_devices('GPU')
#     tf.config.experimental.set_memory_growth(gpus[0], True)




import jax.numpy as jnp
import jax.scipy.linalg as jla
from jax import jit


import jax.numpy as jnp
from jax.scipy.linalg import det, inv

def log_evidence_laplace(samples, log_probs, log_prob_fn, prior_fn):
    """
    Computes Bayesian evidence (log Z) using the Laplace approximation
    from MCMC samples using JAX.

    Args:
        samples: (n_samples, n_params) array of MCMC samples.
        log_probs: (n_samples,) array of log posterior probabilities.
        log_prob_fn: Function to compute log-likelihood (log P(data | theta)).
        prior_fn: Function to compute log-prior (log P(theta)).

    Returns:
        log_evidence: Log of the Bayesian evidence.
    """
    # Convert to JAX arrays
    samples = jnp.asarray(samples)
    log_probs = jnp.asarray(log_probs)

    # MAP estimate: Find the sample with the highest posterior probability
    idx_max = jnp.argmax(log_probs)
    theta_map = samples[idx_max]

    # Compute covariance matrix from MCMC samples
    cov_matrix = jnp.cov(samples, rowvar=False)  # Ensures correct shape (d, d)
    hessian_approx = inv(cov_matrix)  # Approximate inverse Hessian

    # Compute log-likelihood and log-prior at MAP estimate
    logL_map = log_prob_fn(theta_map)
    logPi_map = prior_fn(theta_map)

    # Compute log evidence using Laplace approximation
    d = samples.shape[1]  # Number of parameters
    logZ = logL_map + logPi_map - 0.5 * d * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(det(cov_matrix))

    return logZ


from sklearn.mixture import GaussianMixture
import jax.numpy as jnp
from jax.numpy.linalg import det, inv
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks


def detect_multimodal_parameters_kde(samples, threshold=0.05):
    """
    Detects multimodal parameters using Kernel Density Estimation (KDE).

    Args:
        samples: (n_samples, n_params) array of MCMC samples.
        threshold: Minimum relative prominence to consider separate modes.

    Returns:
        multimodal_params: List of parameter indices that are multimodal.
    """
    multimodal_params = []

    for i in range(samples.shape[1]):  # Loop over each parameter
        param_values = samples[:, i]

        # KDE estimation
        kde = gaussian_kde(param_values)
        x_vals = np.linspace(param_values.min(), param_values.max(), 1000)
        density = kde(x_vals)

        # Find peaks
        peaks, properties = find_peaks(density, prominence=threshold * np.max(density))

        if len(peaks) > 1:  # More than one peak → multimodal
            multimodal_params.append(i)

    return multimodal_params


import jax
import jax.numpy as jnp
from jax.scipy.linalg import det

def log_evidence_laplace_mixed(samples, log_probs, log_prob_fn, prior_fn, threshold=0.05):
    """
    Computes Bayesian evidence (log Z) using the Laplace approximation
    but only around the mode with the highest likelihood.

    Args:
        samples: (n_samples, n_params) array of MCMC samples.
        log_probs: (n_samples,) array of log posterior probabilities.
        log_prob_fn: Function to compute log-likelihood (log P(data | theta)).
        prior_fn: Function to compute log-prior (log P(theta)).
        threshold: KDE peak prominence threshold for detecting multimodality.

    Returns:
        log_evidence: Log of the Bayesian evidence.
    """
    # Convert to JAX arrays
    samples = jnp.asarray(samples)
    log_probs = jnp.asarray(log_probs)

    # Detect multimodal parameters
    multimodal_params = detect_multimodal_parameters_kde(np.array(samples), threshold)
    print(f"Detected multimodal parameters: {multimodal_params}")

    # Find Maximum Likelihood Estimate (MLE) sample
    idx_max = jnp.argmax(log_probs)
    theta_mle = samples[idx_max]

    # Compute covariance only for the mode with highest likelihood
    cov_matrix = jnp.cov(samples, rowvar=False)  # Full covariance matrix

    # Compute log-likelihood and log-prior at MLE estimate
    logL_mle = log_prob_fn(theta_mle)
    logPi_mle = prior_fn(theta_mle)

    # Compute log evidence using Laplace approximation
    d = samples.shape[1]  # Number of parameters
    logZ = logL_mle + logPi_mle - 0.5 * d * jnp.log(2 * jnp.pi) - 0.5 * jnp.log(det(cov_matrix))

    return logZ

def super_gaussian(x, sigma, m, amp=1, x0=0):
    sigma = float(sigma)
    m = float(m)

    return amp * ((np.exp(-(2 ** (2 * m - 1)) * np.log(2) * (((x - x0) ** 2) / ((sigma) ** 2)) ** (m))) ** 2)


def compute_errors(number, errors):
    sigma_std = (errors / np.std(number)) * np.sqrt(1 + (np.std(number) / np.sqrt(len(number))) ** 2 + (
            (number - np.mean(number)) * (np.std(number) / np.sqrt(2 * (len(number) - 1)))) ** 2)

    return sigma_std


def window_amical(datacube, window, m=3):
    isz = datacube.shape[1]  # Assuming square images
    xx, yy = np.arange(isz), np.arange(isz)
    xx2 = xx - isz // 2
    yy2 = isz // 2 - yy
    distance = np.sqrt(xx2 ** 2 + yy2[:, None] ** 2)

    w = super_gaussian(distance, sigma=window * 2, m=m)

    def apply_window(img):
        return img * w

    cleaned_array = vmap(apply_window)(datacube)

    return cleaned_array


def rotator(theta=0):
    """Return the Mueller matrix for rotation clockwise about the optical axis.

    Parameters
    ----------
    theta : float, optional
        Angle of rotation, in radians. By default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix

    Examples
    --------
    >>> rotator(0)
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0.,  0.],
           [ 0., -0.,  1.,  0.],
           [ 0.,  0.,  0.,  1.]])

    >>> rotator(np.deg2rad(45))
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  1.,  0.],
           [ 0., -1.,  0.,  0.],
           [ 0.,  0.,  0.,  1.]])
    """
    cos2t = np.cos(2 * theta)
    sin2t = np.sin(2 * theta)
    M = np.array(((1, 0, 0, 0), (0, cos2t, sin2t, 0), (0, -sin2t, cos2t, 0), (0, 0, 0, 1)))
    return M


def nd_coords(npixels, pixel_scales = 1.0, offsets = 0.0, indexing = "xy", ):

    if indexing not in ["xy", "ij"]:
        raise ValueError("indexing must be either 'xy' or 'ij'.")

    # Promote npixels to tuple to handle 1d case
    if not isinstance(npixels, tuple):
        npixels = (npixels,)

    # Assume equal pixel scales if not given
    if not isinstance(pixel_scales, tuple):
        pixel_scales = (pixel_scales,) * len(npixels)

    # Assume no offset if not given
    if not isinstance(offsets, tuple):
        offsets = (offsets,) * len(npixels)

    def pixel_fn(n, offset, scale):
        start = -(n - 1) / 2 * scale - offset
        end = (n - 1) / 2 * scale - offset
        return np.linspace(start, end, n)

    # Generate the linear edges of each axes
    # TODO: tree_flatten()[0] to avoid squeeze?
    lin_pixels = tree_map(pixel_fn, npixels, offsets, pixel_scales)

    # output (x, y) for 2d, else in order
    positions = np.array(np.meshgrid(*lin_pixels, indexing=indexing))

    # Squeeze for empty axis removal in 1d case
    return np.squeeze(positions)


def return_interpolator(data):
    coords = nd_coords(data.shape, indexing="ij")
    # Get the array coordinate vectors, probs a more elegant way to do this but eh
    ys = coords[0, :, 0]
    xs = coords[1, 0, :]

    interpolator = ipx.Interpolator2D(xs, ys, data, extrap=True)

    return interpolator, coords


def load_average_fixed_grids_multi_jax(materials, grain_size, grids_loc):
    types = [
        ('scat_V45_', 'H_scat_m'),
        ('scat_H45_', 'V_scat_m'),
        ('scat_V_', 'H45_scat_m'),
        ('scat_H_', 'V45_scat_m')
    ]

    wavelengths = ['610', '670', '720', '760']
    averaged_grids = {}

    for prefix, filename_base in types:
        for wl in wavelengths:
            # Load and stack grids from all materials
            stacked = []
            for mat in materials:
                arr = np.load(f"{grids_loc}{filename_base}{mat}_r{grain_size}_w{wl}.npy")[0:308, 0:308, 0:308]
                stacked.append(jnp.array(arr))  # convert to JAX array
            # Stack and average
            stacked_array = jnp.stack(stacked, axis=0)  # shape (N, 308, 308, 308)
            avg_grid = jnp.mean(stacked_array, axis=0)
            averaged_grids[f"{prefix}{wl}"] = avg_grid

    return averaged_grids


def prior(theta):
    for i in range(len(theta)):
        if theta[i] > model_params['priors'][i][0] and theta[i] < model_params['priors'][i][1]:
            test = 3
        else:
            test = -4
            return -np.inf
    return 0


@eqx.filter_jit
def make_star(model_params):

    stellar_object = geometric_star(model_params)
    stellar_object.make_3D_model()
    y_model_this = stellar_object.simulate_nrm()

    # === Segment 1 ===
    q_real_vis_1 = stellar_object.ydata_real[0:153].copy()
    u_real_vis_1 = stellar_object.ydata_real[153:153 * 2].copy()
    q_real_cp_1 = stellar_object.ydata_real[153 * 2:153 * 2 + 816].copy()
    u_real_cp_1 = stellar_object.ydata_real[153 * 2 + 816:153 * 2 + 816 * 2].copy()

    q_real_vis_norm_1 = (q_real_vis_1 - np.mean(q_real_vis_1)) / np.std(q_real_vis_1)
    u_real_vis_norm_1 = (u_real_vis_1 - np.mean(u_real_vis_1)) / np.std(u_real_vis_1)
    q_real_cp_norm_1 = (q_real_cp_1 - np.mean(q_real_cp_1)) / np.std(q_real_cp_1)
    u_real_cp_norm_1 = (u_real_cp_1 - np.mean(u_real_cp_1)) / np.std(u_real_cp_1)

    norm_obs_1 = np.concatenate((q_real_vis_norm_1,
                                 u_real_vis_norm_1,
                                 np.sin(np.deg2rad(q_real_cp_norm_1)),
                                 np.cos(np.deg2rad(q_real_cp_norm_1)),
                                 np.sin(np.deg2rad(u_real_cp_norm_1)),
                                 np.cos(np.deg2rad(u_real_cp_norm_1))))

    q_real_vis_err_1 = stellar_object.ydata_real_err[0:153]
    u_real_vis_err_1 = stellar_object.ydata_real_err[153:153 * 2]
    q_real_cp_err_1 = stellar_object.ydata_real_err[153 * 2:153 * 2 + 816]
    u_real_cp_err_1 = stellar_object.ydata_real_err[153 * 2 + 816:153 * 2 + 816 * 2]

    q_real_vis_err_norm_1 = compute_errors(q_real_vis_1, q_real_vis_err_1)
    u_real_vis_err_norm_1 = compute_errors(u_real_vis_1, u_real_vis_err_1)
    q_real_cp_err_norm_1 = compute_errors(q_real_cp_1, q_real_cp_err_1)
    u_real_cp_err_norm_1 = compute_errors(u_real_cp_1, u_real_cp_err_1)

    norm_obs_err_1 = np.abs(np.concatenate((q_real_vis_err_norm_1,
                                            u_real_vis_err_norm_1,
                                            np.sin(np.deg2rad(q_real_cp_err_norm_1)),
                                            np.cos(np.deg2rad(q_real_cp_err_norm_1)),
                                            np.sin(np.deg2rad(u_real_cp_err_norm_1)),
                                            np.cos(np.deg2rad(u_real_cp_err_norm_1)))))

    q_model_vis_1 = y_model_this[0:153]
    u_model_vis_1 = y_model_this[153:153 * 2]
    q_model_cp_1 = y_model_this[153 * 2:153 * 2 + 816]
    u_model_cp_1 = y_model_this[153 * 2 + 816:153 * 2 + 816 * 2]

    q_model_vis_norm_1 = (q_model_vis_1 - np.mean(q_real_vis_1)) / np.std(q_real_vis_1)
    u_model_vis_norm_1 = (u_model_vis_1 - np.mean(u_real_vis_1)) / np.std(u_real_vis_1)
    q_model_cp_norm_1 = (q_model_cp_1 - np.mean(q_real_cp_1)) / np.std(q_real_cp_1)
    u_model_cp_norm_1 = (u_model_cp_1 - np.mean(u_real_cp_1)) / np.std(u_real_cp_1)

    norm_model_1 = np.concatenate((q_model_vis_norm_1,
                                   u_model_vis_norm_1,
                                   np.sin(np.deg2rad(q_model_cp_norm_1)),
                                   np.cos(np.deg2rad(q_model_cp_norm_1)),
                                   np.sin(np.deg2rad(u_model_cp_norm_1)),
                                   np.cos(np.deg2rad(u_model_cp_norm_1))))

    # === Segment 2 ===
    start_2 = 153 * 2 + 816 * 2
    q_real_vis_2 = stellar_object.ydata_real[start_2:start_2 + 153].copy()
    u_real_vis_2 = stellar_object.ydata_real[start_2 + 153:start_2 + 153 * 2].copy()
    q_real_cp_2 = stellar_object.ydata_real[start_2 + 153 * 2:start_2 + 153 * 2 + 816].copy()
    u_real_cp_2 = stellar_object.ydata_real[start_2 + 153 * 2 + 816:start_2 + 153 * 2 + 816 * 2].copy()

    q_real_vis_norm_2 = (q_real_vis_2 - np.mean(q_real_vis_2)) / np.std(q_real_vis_2)
    u_real_vis_norm_2 = (u_real_vis_2 - np.mean(u_real_vis_2)) / np.std(u_real_vis_2)
    q_real_cp_norm_2 = (q_real_cp_2 - np.mean(q_real_cp_2)) / np.std(q_real_cp_2)
    u_real_cp_norm_2 = (u_real_cp_2 - np.mean(u_real_cp_2)) / np.std(u_real_cp_2)

    norm_obs_2 = np.concatenate((q_real_vis_norm_2,
                                 u_real_vis_norm_2,
                                 np.sin(np.deg2rad(q_real_cp_norm_2)),
                                 np.cos(np.deg2rad(q_real_cp_norm_2)),
                                 np.sin(np.deg2rad(u_real_cp_norm_2)),
                                 np.cos(np.deg2rad(u_real_cp_norm_2))))

    q_real_vis_err_2 = stellar_object.ydata_real_err[start_2:start_2 + 153]
    u_real_vis_err_2 = stellar_object.ydata_real_err[start_2 + 153:start_2 + 153 * 2]
    q_real_cp_err_2 = stellar_object.ydata_real_err[start_2 + 153 * 2:start_2 + 153 * 2 + 816]
    u_real_cp_err_2 = stellar_object.ydata_real_err[start_2 + 153 * 2 + 816:start_2 + 153 * 2 + 816 * 2]

    q_real_vis_err_norm_2 = compute_errors(q_real_vis_2, q_real_vis_err_2)
    u_real_vis_err_norm_2 = compute_errors(u_real_vis_2, u_real_vis_err_2)
    q_real_cp_err_norm_2 = compute_errors(q_real_cp_2, q_real_cp_err_2)
    u_real_cp_err_norm_2 = compute_errors(u_real_cp_2, u_real_cp_err_2)

    norm_obs_err_2 = np.abs(np.concatenate((q_real_vis_err_norm_2,
                                            u_real_vis_err_norm_2,
                                            np.sin(np.deg2rad(q_real_cp_err_norm_2)),
                                            np.cos(np.deg2rad(q_real_cp_err_norm_2)),
                                            np.sin(np.deg2rad(u_real_cp_err_norm_2)),
                                            np.cos(np.deg2rad(u_real_cp_err_norm_2)))))

    q_model_vis_2 = y_model_this[start_2:start_2 + 153]
    u_model_vis_2 = y_model_this[start_2 + 153:start_2 + 153 * 2]
    q_model_cp_2 = y_model_this[start_2 + 153 * 2:start_2 + 153 * 2 + 816]
    u_model_cp_2 = y_model_this[start_2 + 153 * 2 + 816:start_2 + 153 * 2 + 816 * 2]

    q_model_vis_norm_2 = (q_model_vis_2 - np.mean(q_real_vis_2)) / np.std(q_real_vis_2)
    u_model_vis_norm_2 = (u_model_vis_2 - np.mean(u_real_vis_2)) / np.std(u_real_vis_2)
    q_model_cp_norm_2 = (q_model_cp_2 - np.mean(q_real_cp_2)) / np.std(q_real_cp_2)
    u_model_cp_norm_2 = (u_model_cp_2 - np.mean(u_real_cp_2)) / np.std(u_real_cp_2)

    norm_model_2 = np.concatenate((q_model_vis_norm_2,
                                   u_model_vis_norm_2,
                                   np.sin(np.deg2rad(q_model_cp_norm_2)),
                                   np.cos(np.deg2rad(q_model_cp_norm_2)),
                                   np.sin(np.deg2rad(u_model_cp_norm_2)),
                                   np.cos(np.deg2rad(u_model_cp_norm_2))))

    # === Segment 3 ===
    # === Segment 3 ===
    start_3 = 153 * 2 + 816 * 2 + 153 * 2 + 816 * 2
    q_real_vis_3 = stellar_object.ydata_real[start_3:start_3 + 153].copy()
    u_real_vis_3 = stellar_object.ydata_real[start_3 + 153:start_3 + 153 * 2].copy()
    q_real_cp_3 = stellar_object.ydata_real[start_3 + 153 * 2:start_3 + 153 * 2 + 816].copy()
    u_real_cp_3 = stellar_object.ydata_real[start_3 + 153 * 2 + 816:start_3 + 153 * 2 + 816 * 2].copy()

    q_real_vis_norm_3 = (q_real_vis_3 - np.mean(q_real_vis_3)) / np.std(q_real_vis_3)
    u_real_vis_norm_3 = (u_real_vis_3 - np.mean(u_real_vis_3)) / np.std(u_real_vis_3)
    q_real_cp_norm_3 = (q_real_cp_3 - np.mean(q_real_cp_3)) / np.std(q_real_cp_3)
    u_real_cp_norm_3 = (u_real_cp_3 - np.mean(u_real_cp_3)) / np.std(u_real_cp_3)

    norm_obs_3 = np.concatenate((q_real_vis_norm_3,
                                 u_real_vis_norm_3,
                                 np.sin(np.deg2rad(q_real_cp_norm_3)),
                                 np.cos(np.deg2rad(q_real_cp_norm_3)),
                                 np.sin(np.deg2rad(u_real_cp_norm_3)),
                                 np.cos(np.deg2rad(u_real_cp_norm_3))))

    q_real_vis_err_3 = stellar_object.ydata_real_err[start_3:start_3 + 153]
    u_real_vis_err_3 = stellar_object.ydata_real_err[start_3 + 153:start_3 + 153 * 2]
    q_real_cp_err_3 = stellar_object.ydata_real_err[start_3 + 153 * 2:start_3 + 153 * 2 + 816]
    u_real_cp_err_3 = stellar_object.ydata_real_err[start_3 + 153 * 2 + 816:start_3 + 153 * 2 + 816 * 2]

    q_real_vis_err_norm_3 = compute_errors(q_real_vis_3, q_real_vis_err_3)
    u_real_vis_err_norm_3 = compute_errors(u_real_vis_3, u_real_vis_err_3)
    q_real_cp_err_norm_3 = compute_errors(q_real_cp_3, q_real_cp_err_3)
    u_real_cp_err_norm_3 = compute_errors(u_real_cp_3, u_real_cp_err_3)

    norm_obs_err_3 = np.abs(np.concatenate((q_real_vis_err_norm_3,
                                            u_real_vis_err_norm_3,
                                            np.sin(np.deg2rad(q_real_cp_err_norm_3)),
                                            np.cos(np.deg2rad(q_real_cp_err_norm_3)),
                                            np.sin(np.deg2rad(u_real_cp_err_norm_3)),
                                            np.cos(np.deg2rad(u_real_cp_err_norm_3)))))

    q_model_vis_3 = y_model_this[start_3:start_3 + 153]
    u_model_vis_3 = y_model_this[start_3 + 153:start_3 + 153 * 2]
    q_model_cp_3 = y_model_this[start_3 + 153 * 2:start_3 + 153 * 2 + 816]
    u_model_cp_3 = y_model_this[start_3 + 153 * 2 + 816:start_3 + 153 * 2 + 816 * 2]

    q_model_vis_norm_3 = (q_model_vis_3 - np.mean(q_real_vis_3)) / np.std(q_real_vis_3)
    u_model_vis_norm_3 = (u_model_vis_3 - np.mean(u_real_vis_3)) / np.std(u_real_vis_3)
    q_model_cp_norm_3 = (q_model_cp_3 - np.mean(q_real_cp_3)) / np.std(q_real_cp_3)
    u_model_cp_norm_3 = (u_model_cp_3 - np.mean(u_real_cp_3)) / np.std(u_real_cp_3)

    norm_model_3 = np.concatenate((q_model_vis_norm_3,
                                   u_model_vis_norm_3,
                                   np.sin(np.deg2rad(q_model_cp_norm_3)),
                                   np.cos(np.deg2rad(q_model_cp_norm_3)),
                                   np.sin(np.deg2rad(u_model_cp_norm_3)),
                                   np.cos(np.deg2rad(u_model_cp_norm_3))))

    # === Segment 4 ===
    start_4 = start_3 + 153 * 2 + 816 * 2
    q_real_vis_4 = stellar_object.ydata_real[start_4:start_4 + 153].copy()
    u_real_vis_4 = stellar_object.ydata_real[start_4 + 153:start_4 + 153 * 2].copy()
    q_real_cp_4 = stellar_object.ydata_real[start_4 + 153 * 2:start_4 + 153 * 2 + 816].copy()
    u_real_cp_4 = stellar_object.ydata_real[start_4 + 153 * 2 + 816:start_4 + 153 * 2 + 816 * 2].copy()

    q_real_vis_norm_4 = (q_real_vis_4 - np.mean(q_real_vis_4)) / np.std(q_real_vis_4)
    u_real_vis_norm_4 = (u_real_vis_4 - np.mean(u_real_vis_4)) / np.std(u_real_vis_4)
    q_real_cp_norm_4 = (q_real_cp_4 - np.mean(q_real_cp_4)) / np.std(q_real_cp_4)
    u_real_cp_norm_4 = (u_real_cp_4 - np.mean(u_real_cp_4)) / np.std(u_real_cp_4)

    norm_obs_4 = np.concatenate((q_real_vis_norm_4,
                                 u_real_vis_norm_4,
                                 np.sin(np.deg2rad(q_real_cp_norm_4)),
                                 np.cos(np.deg2rad(q_real_cp_norm_4)),
                                 np.sin(np.deg2rad(u_real_cp_norm_4)),
                                 np.cos(np.deg2rad(u_real_cp_norm_4))))

    q_real_vis_err_4 = stellar_object.ydata_real_err[start_4:start_4 + 153]
    u_real_vis_err_4 = stellar_object.ydata_real_err[start_4 + 153:start_4 + 153 * 2]
    q_real_cp_err_4 = stellar_object.ydata_real_err[start_4 + 153 * 2:start_4 + 153 * 2 + 816]
    u_real_cp_err_4 = stellar_object.ydata_real_err[start_4 + 153 * 2 + 816:start_4 + 153 * 2 + 816 * 2]

    q_real_vis_err_norm_4 = compute_errors(q_real_vis_4, q_real_vis_err_4)
    u_real_vis_err_norm_4 = compute_errors(u_real_vis_4, u_real_vis_err_4)
    q_real_cp_err_norm_4 = compute_errors(q_real_cp_4, q_real_cp_err_4)
    u_real_cp_err_norm_4 = compute_errors(u_real_cp_4, u_real_cp_err_4)

    norm_obs_err_4 = np.abs(np.concatenate((q_real_vis_err_norm_4,
                                            u_real_vis_err_norm_4,
                                            np.sin(np.deg2rad(q_real_cp_err_norm_4)),
                                            np.cos(np.deg2rad(q_real_cp_err_norm_4)),
                                            np.sin(np.deg2rad(u_real_cp_err_norm_4)),
                                            np.cos(np.deg2rad(u_real_cp_err_norm_4)))))

    q_model_vis_4 = y_model_this[start_4:start_4 + 153]
    u_model_vis_4 = y_model_this[start_4 + 153:start_4 + 153 * 2]
    q_model_cp_4 = y_model_this[start_4 + 153 * 2:start_4 + 153 * 2 + 816]
    u_model_cp_4 = y_model_this[start_4 + 153 * 2 + 816:start_4 + 153 * 2 + 816 * 2]

    q_model_vis_norm_4 = (q_model_vis_4 - np.mean(q_real_vis_4)) / np.std(q_real_vis_4)
    u_model_vis_norm_4 = (u_model_vis_4 - np.mean(u_real_vis_4)) / np.std(u_real_vis_4)
    q_model_cp_norm_4 = (q_model_cp_4 - np.mean(q_real_cp_4)) / np.std(q_real_cp_4)
    u_model_cp_norm_4 = (u_model_cp_4 - np.mean(u_real_cp_4)) / np.std(u_real_cp_4)

    norm_model_4 = np.concatenate((q_model_vis_norm_4,
                                   u_model_vis_norm_4,
                                   np.sin(np.deg2rad(q_model_cp_norm_4)),
                                   np.cos(np.deg2rad(q_model_cp_norm_4)),
                                   np.sin(np.deg2rad(u_model_cp_norm_4)),
                                   np.cos(np.deg2rad(u_model_cp_norm_4))))

    norm_obs = np.concatenate((norm_obs_1, norm_obs_2, norm_obs_3, norm_obs_4), axis = 0) #
    norm_model = np.concatenate((norm_model_1, norm_model_2, norm_model_3, norm_model_4), axis = 0) #)
    norm_obs_err = np.concatenate((norm_obs_err_1, norm_obs_err_2, norm_obs_err_3, norm_obs_err_4), axis = 0) #

    # not fitting to 610 at all.
    # norm_obs = np.concatenate((norm_obs_2, norm_obs_3, norm_obs_4), axis = 0) #
    # norm_model = np.concatenate((norm_model_2, norm_model_3, norm_model_4), axis = 0) #)
    # norm_obs_err = np.concatenate((norm_obs_err_2, norm_obs_err_3, norm_obs_err_4), axis = 0) #

    # norm_obs = norm_obs_4            #np.concatenate((norm_obs_2, norm_obs_3, norm_obs_4), axis = 0) #
    # norm_model = norm_model_4        #np.concatenate((norm_model_2, norm_model_3, norm_model_4), axis = 0) #)
    # norm_obs_err = norm_obs_err_4    #np.concatenate((norm_obs_err_2, norm_obs_err_3, norm_obs_err_4), axis = 0) #



    # lnL_model = -0.5*uf.chi_squ(model_params['ydata_real'], y_model_this, model_params['ydata_real_err'])/2
    lnL_model = -0.5 * uf.chi_squ(norm_obs, norm_model, norm_obs_err) / 2


    return lnL_model


def lnprob_model(theta):


    for i in range(len(model_params['changing_by_theta'])):
        vari = model_params['changing_by_theta'][i]
        model_params[vari] = theta[i]
    for i in range(len(model_params['changing_by_setting'])):
        vari = model_params['changing_by_setting'][i]
        model_params[vari[0]] = model_params[vari[1]]

    check_prior = prior(theta)#, model_params['star_radius'])#, model_params)

    if not np.isfinite(check_prior):
        return -np.inf


    lnL_model = make_star(model_params)

    if np.isnan(check_prior + lnL_model): #math.isnan(check_prior + lnL_model):
        for i in range(len(model_params['changing_by_theta'])):
            vari = model_params['changing_by_theta'][i]
            print(model_params[vari])

    return check_prior + lnL_model


def compute_DFTM1(x, y, uv, wavel):
    '''Compute a direct Fourier transform matrix, from coordinates x and y (milliarcsec) to uv (metres) at a given wavelength wavel.'''

    # Convert to radians
    x = x * np.pi / 180.0 / 3600.0 / 1000.0
    y = y * np.pi / 180.0 / 3600.0 / 1000.0

    # get uv in nondimensional units
    uv = uv / wavel
    # Compute the matrix
    dftm = np.exp(-2j * np.pi * (np.outer(uv[:, 0], x) + np.outer(uv[:, 1], y)))
    return dftm

import logging

# Configure logging
logging.basicConfig(
    filename="script.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]: %(message)s"
)

def main_computation(step):
    # Example function that might raise an error
    if step == 3:
        raise ValueError("An intentional error occurred!")
    return step * 2

# Main loop
for step in range(10):
    try:
        logging.info(f"Processing step {step}")
        result = main_computation(step)
        logging.info(f"Result for step {step}: {result}")
    except Exception as e:
        logging.error(f"Error on step {step}: {e}")
        # Optional: You can handle the error further, such as skipping to the next step
        continue

logging.info("Script completed.")
loc = os.getcwd()
image_size = 308
star_radius = 10.5 #10.5
pixel_ratio = 1
wavelength = 750 * 10 ** (-9)
field_of_view = pixel_ratio  * image_size
size_biggest_baseline_m = (wavelength) / (pixel_ratio * 2 * (4.8481368 * 10 ** (-9)))

tags = ['muCep_2017',  'muCep_2018_02', 'muCep_2018_05', 'muCep_2020']

models = ['Model_A',
          'Model_B',
          'Model_C',
          'Model_G',
          'Model_Hcirc',
          'Model_I',

          'Model_J',
          "Model_B1",
          'Model_H1',
          'Model_I1',
          'Model_plellipse1',


          'Model_shellthickcircle',
          'Model_twothickcircles',
          "Model_thinshellplshell",
          "Model_threethinshells",
          "Model_onethin_onethick",

          'Model_pl_1',
          "Model_Dtest",
          "Model_shellthickcircle_1",
          "Model_twothincircles",
          "Model_thinellipse_1",


          "Model_plellipse_2",
          "Model_twothinellipses",
          "Model_F",  # model F # 23
          'Model_ellipse_twoblobs',
          'Model_ellipse_twoblobs_prior',
          'Model_E',  # Model E
          'Model_bright_spot_thin_ellipse',
          'Model_offset_thin_ellipse', 'Model_offset_thick_elliipse_blob',
          'Model_O',
          'Model_shellthickellipse',  # _21',
            'Model_O3circ',
          'Model_thinellipseblob',
          'Model_E',
        'Model_thincircblob',
          'Model_thincllipseblob',
          'Model_Q ',
          'Model_enhance'
          ]

p0s = [[15, 4], # A# Model A thin spherical shell
       [15, 2,  4], # Model B power law spherical shell
       [15, 15, 4, 4], # Model C two thin shells
       [15, 15,45, 4, np.pi*3/2 , 80, 0.5 ], # Model G - Ellipitcal shell with blob
       [15,  4, np.pi/4 , 80, 0.5, 2 ], # Model H
       [15,    4, 0, 120, 0.5],  # Model I

       [15, 12, 45, 4, 0, 120, 0.5],  # Model J
       [12, 4, 3],  # Model B.1
       [15, 15,45, 4, np.pi*3/2 , 120, 0.5, 2 ],     # Model H1
       [15, 2, 4, np.pi * 3 / 2, 120, 0.5],           # Model I1
       [12, 4,  12,   0 ],


       [12, 4, 3],
       [12, 1, 16, 1, 4, 5],
       [12,  12, 4, 1.5],
       [12,  12, 12, 4, 1.5, 1.5],
       [12,  12, 2, 4, 1.5],

       [12, 4, 3],
       [15, 15,  0,  3],                            # Model D
       [12, 4, 3],
       [12, 12, 4, 4],
       [12, 12,  12, 45,  3],

       [12, 4,  12,  0, 2],
       [12, 12, 12, 12, 0, 4, 4],
       [15,   4, np.pi*3/2 , 120, 0.5],              # model F
       [12, 12, 0,  4, np.pi*3/2 , 120, 0.5, 4.7 , 120, 0.5],
       [15, 15,  0, 4, np.pi * 3 / 2, 120, 0.5, 4.7, 120, 0.5],

       [15, 15, 45, 4, 0, 0 ],                         # Model E
       [12, 12, 0, 4, np.pi/2, np.pi/8, 10, np.pi/8],
       [12, 12, 0, 4, 0, 0 ] ,
       [12, 12, 0, 4, np.pi * 3 / 2, 120, 0.5, 0, 0, 2],
       [15, 15, 0,  4, 0, 80, 0.5, 0, 0],
       [12, 12, 25, 4, 3],
       [15,  4, np.pi /4, 80, 0.5, 0, 0],
       [15, 15, 0, 4, np.pi * 1 / 2, 80, 0.5], # was pi times 3/2
       [15,  4, 0, 0 , 15, 25],
       [15, 4, 0 , 80, 0.5],
       [15, 4, 0, 80, 0.5 , 15, 25],
       [15, 15, 0, 4,  np.pi /4, 80, 0.5, 360, 2],
       [15, 15, 0, 4, 360, 2]
       ]


changing_by_thetas =  [['a',  'dust_star_contrast'], # Model A thin spherical shell
                       ['a', 'dust_pl_exp',  'dust_star_contrast'], # Model B power law spherical shell
                       ['a', 'a2', 'dust_star_contrast', 'ellipse_contrast_ratio'], # Model C
                       ['a', 'b', 'alpha', 'dust_star_contrast',  'phi_blob',  'blob_radial_distance', 'blob_contrast'], # Model G
                       ['a',  'dust_star_contrast',  'phi_blob',  'blob_radial_distance', 'blob_contrast', 'dust_pl_exp'],# Model H
                       ['a',  'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast'],  # Model I

                       ['a', 'b', 'alpha',  'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast'],     # Model J
                       ['a',  'dust_star_contrast', 'dust_pl_exp'],    # Model B.1
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast', 'dust_pl_exp'],  # Model H1
                       ['a', 'thicka', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast'], # Model I1
                       ['a', 'dust_star_contrast', 'b',  'alpha' ],

                       ['a', 'dust_star_contrast', 'thicka'],
                       ['a', 'thick1',  'a2', 'thick2', 'dust_star_contrast', 'ellipse_contrast_ratio'],
                       ['a',  'a2',  'dust_star_contrast', 'ellipse_contrast_ratio'],
                       ['a',  'a2', 'a3', 'dust_star_contrast', 'ellipse_contrast_ratio', 'ellipse_contrast_ratio2'],
                       ['a', 'a2',  'thicka2', 'dust_star_contrast', 'ellipse_contrast_ratio'],

                       ['a', 'dust_star_contrast', 'dust_pl_exp'],
                       ['a', 'b',   'alpha', 'dust_star_contrast'], # Model D
                       ['a', 'dust_star_contrast', 'thicka'],
                       ['a', 'a2','dust_star_contrast', 'ellipse_contrast_ratio' ],
                       ['a', 'b','c', 'alpha', 'dust_star_contrast'],

                       ['a', 'dust_star_contrast', 'b',    'alpha', 'dust_pl_exp'],
                       ['a', 'b', 'a2', 'b2', 'alpha',  'dust_star_contrast', 'ellipse_contrast_ratio'],


                       ['a', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast'], # Model F
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast', 'phi_blob2', 'blob_radial_distance2', 'blob_contrast2'],
                       ['a', 'b',  'alpha', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast', 'phi_blob2', 'blob_radial_distance2', 'blob_contrast2'],
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'h', 'k'],  # Model E
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'bright_location_theta', 'bright_size', 'bright_mag',  'bright_location_phi'],
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'h', 'k'],
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast', 'h', 'k', 'thicka'],
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast', 'h', 'k'],
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'thicka'],
                       ['a',  'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast', 'h', 'k'],
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast'],
                       ['a', 'dust_star_contrast', 'h', 'k', 'b', 'alpha'],
                       ['a', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast'],
                       ['a', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast' , 'b', 'alpha'],
                       ['a',  'b', 'alpha', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast',  'enhancement_loc', 'enhancement_amp'],
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'enhancement_loc', 'enhancement_amp']
                       ]




priors =  [[[1, 50],  [0.01,  50]],                      # Model A thin spherical shell
           [[1, 100], [0.1, 10], [0.01, 50]],             # Model B power law spherical shell
           [[1, 200], [12, 200], [0.01, 50], [0.01, 50]], # Model C two thin shells
           [[1, 100], [1, 100], [0, 90], [0.01, 50],   [np.pi , 2*np.pi],    [0, 500],  [0.001, 500]], # Model G -
           [[1, 100],   [0.01, 50],  [-np.pi/4, np.pi / 2],    [0, 500],  [0.001, 500], [0.1, 9.5]], # Model H
           [[1, 100],   [0.01, 50],   [-np.pi/2, np.pi/2],    [0, 500],  [0.001, 500]], # Model I


           [[12, 100], [12, 100], [-45, 45],  [0.01, 50],    [-np.pi/2, np.pi/2],    [0, 500],  [0.001, 500]],  # Model J
           [[1, 200], [0.01, 50], [0.5, 9]],  # Model B.1
           [[0, 100], [0, 100], [0, 90], [0.01, 50],   [np.pi , 2*np.pi],    [0, 500],  [0.001, 500], [0.1, 9.5]],     #Model H1
           [[1, 100],  [1, 100], [0.01, 50],   [np.pi , 2*np.pi],    [0, 500],  [0.001, 500]],                       # Model I1
           [[1 ,150], [0.01, 50],  [1, 50], [0, 90]],

           [[12, 200], [0.01, 10], [1, 150]], # ellipse pl
           [[12, 15], [1, 150],   [12, 150], [1, 300],   [0.01, 10], [0.00001, 500]],
           [[12, 150],  [12, 150], [0.01, 10], [0.00001, 100]],
           [[11, 150], [11, 150], [11, 150], [0.01, 10], [0.00001, 100], [0.00001, 100]],
           [[11, 150], [11, 150],  [1, 150], [0.00001, 10], [0.00001, 500]],


           [[1, 200], [0.01, 10], [0.1, 15]],
           [[5, 25], [5, 25],   [0, 45], [0.01, 50]],                        # Model D
           [[1 , 200], [0.01, 10], [1, 150]],
           [[12, 200], [12, 200], [0.01, 10], [0.01, 50]],
           [[1, 25], [1, 25], [1, 50],  [-45, 45], [0.01, 10]],

           [[12, 150], [0.01, 10], [12, 50],  [0, 90], [0.1, 4]],
           [[12, 250], [12, 250], [12, 250], [12, 250], [0, 90],  [0.01, 10], [0.0001, 750]],
           [[12, 100],  [0.01, 10], [np.pi, 2 * np.pi], [0, 500], [0.001, 500]], # Model F
           [[12, 100], [12, 100], [-45, 45], [0.01, 10], [np.pi, 2*np.pi], [0, 500], [0.001, 500], [4.52, 4.70], [0, 500], [0.001, 500]],
           [[15, 100], [15, 100],  [-45, 45], [0.01, 10], [np.pi, 2*np.pi], [0, 500], [0.001, 500],[np.pi, 2*np.pi], [0, 500], [0.001, 500]],

           [[12, 25], [12, 25], [0,90], [0.01, 10], [-15, 15], [-15, 15]],                           # Model E
           [[12, 100], [12, 100], [-45, 45], [0.01, 10], [0, np.pi*2], [0.01, np.pi], [0.01, 200], [0, np.pi]],
           [[12, 100], [12, 100], [0, 90], [0.01, 10], [-9, 9], [-9, 9]],
           [[12, 100], [12, 100], [0, 90], [0.01, 10], [np.pi, 2 * np.pi], [0, 500], [0.001, 500], [-9, 9], [-9, 9], [1, 40]],
           [[1 , 100], [1 , 100], [-25, 25], [0.01, 10], [-np.pi/2, np.pi/2], [0,200], [0.001, 500],  [-9, 9], [-9, 9]] ,


           [[1, 200], [1, 200],  [0, 45], [0.01, 20], [1, 200]],
           [[1, 100],  [0.01, 10],  [-np.pi/4, np.pi / 2], [0, 200], [0.001, 500], [-10,10], [-10,10]],
           [[12, 100], [12, 100], [-25, 25], [0.01, 10], [-np.pi/2, np.pi/2], [80, 500], [0.001, 500]], # np.pi, 2 * np.pi
           [[12, 50], [0.01, 10], [-50, 50], [-50, 50], [12, 50], [0, 45]],
           [[1, 100],  [0.01, 10],  [-np.pi/2, np.pi/2],    [0, 500],  [0.001, 500]],
           [[1, 100], [0.01, 10], [-np.pi/4, np.pi / 2], [0, 500], [0.001, 500], [1, 100], [0, 45]],
           [[1, 100], [0, 100], [-45, 45], [0.01, 10],   [-np.pi/4, np.pi / 2], [0, 250], [0.001, 500], [270, 450], [0.001, 15]] ,
           [[1, 100], [1, 100], [-45, 45], [0.01, 20], [270, 450], [0.001, 15]]]  # Model J1]

changing_by_settings =  [[['c', 'a'], ['b', 'a']],                                         # Model A thin spherical shell
                         [['b', 'a'], ['c', 'a']],                                         # Model B power law spherical shell
                         [['c', 'a'], ['b', 'a'], ['c2', 'a2'],  ['b2', 'a2']],                 # C
                         [['c', 'a']],                                                        # Model G
                         [['c', 'a'], ['b', 'a']],                                                           # Model H
                         [['c', 'a'], ['b', 'a']],                                                          # Model I

                         [['c', 'a']] ,                                                                     # Model J
                         [['c', 'a'],['b', 'a']],
                         [['c', 'a']],                                                                      # Model H1
                         [['c', 'a'], ['b', 'a']],                                                              # Model I1
                         [['c', 'a']],

                         [['c', 'a'], ['b', 'a']],
                         [['c', 'a'], ['b', 'a'], ['c2', 'a2'], ['b2', 'a2']],
                         [['c', 'a'], ['b', 'a'], ['c2', 'a2'], ['b2', 'a2']],
                         [['c', 'a'], ['b', 'a'], ['c2', 'a2'], ['b2', 'a2'], ['c3', 'a3'], ['b3', 'a3']],
                         [['c', 'a'], ['b', 'a'], ['c2', 'a2'], ['b2', 'a2']],

                         [['c', 'a'], ['b', 'a']],
                         [['c', 'a']], # Model D

                         [['c', 'a'], ['b', 'a']],
                         [['c', 'a'], ['b', 'a'], ['c2', 'a2'], ['b2', 'a2']],
                         [],

                         [['c', 'a']],
                         [['c', 'a'], ['c2', 'a2']],

                         [['c', 'a'], ['b', 'a']], # Model F
                         [['c', 'a']],
                         [['c', 'a']],

                         [['c', 'a']], # Model E
                         [['c', 'a']],
                         [['c', 'a']],
                         [['c', 'a']],
                         [['c', 'a']],
                         [['c', 'a']],
                         [['c', 'a'], ['b', 'a']],
                         [['c', 'a']],
                         [['c', 'a']],
                         [['c', 'a'], ['b', 'a']],
                          [['c', 'a']],
                         [['c', 'a'] ],
                         [['c', 'a']]
                         ] # E



#['a',  'a2', 'a3', 'dust_star_contrast', 'ellipse_contrast_ratio', 'ellipse_contrast_ratio2']]

num_steps_run_list =     [4000, 5000, 5000, 5000,   5000, 5000 , 6000, 5000, 8000, 6000, 8000, 5000, 5000, 5000, 5000, 5000, 5000, 3000, 5000, 5000, 5000, 5000, 10000, 4000, 10000, 10000, 5000, 7000, 5000, 8000, 8000, 8000, 9000, 12000, 8000, 9000, 8000, 9000, 9000]
num_steps_discard_list = [2000, 2000,  2000,  2000, 3000, 3000 , 4000, 1000, 4000, 4000,   3000, 2000, 3000, 2000, 1000, 1000, 1000, 1000, 1000, 2000, 2000, 2000, 7000, 2000, 4000, 5000, 2000, 3000, 2000, 4000, 5000, 4000, 4000, 6000, 4000, 4000, 4000, 4000, 4000]

                                                                                          # this was two_thick_circles

model_class = ['ellipse_thin', # Model A thin spherical shell
               'ellipse', # Model B power law spherical shell
               'two_thin_circles', # Model C - two thin shells
               'ellipse_blob_bright', # Model G  #this was a power law and i didnt realise
               'ellipse_and_blob',  # Model H
               'ellipse_blob_bright',    # Model I

               'ellipse_blob_bright', # Model J
               'ellipse',
               'ellipse_and_blob',  # Model H1
                'ellipse_blob_bright',  # Model Id1
               'ellipse',

               'thick_ellipse',
               'two_thick_circles',
               'two_thin_circles',
               'three_thin_circles',
               'one_thin_one_thick_circle',

               'ellipse',
               'ellipse_thin',
               'thick_ellipse',
               'two_thin_circles',
               'ellipse_thin',

               'ellipse',
               'two_thin_circles',
               'ellipse_blob_bright', # Model F
               'ellipse_and_twoblob',
               'ellipse_and_twoblob',


               'ellipse_thin', # Model E
               'bright_spot_ellipse',
               'ellipse_thin',
               'ellipse_blob_bright',
               'ellipse_and_blob',
               'thick_ellipse',
              'ellipse_and_blob',
               'ellipse_blob_bright',
               'ellipse_thin',
               'ellipse_blob_bright',
               'ellipse_blob_bright',
               'ellipse_enhance_and_blob',
               'ellipse_enhance'
               ]


observing_run = 'muCep_2023'
savedir = '/geometric_models_data/final_thesis_chem/images/'
savedir_fits = '/geometric_models_data/final_thesis_chem/fits/'
meta_data = '/geometric_models_data/{}/'.format(observing_run)
grids_loc = '/scattering_grids/muCep_2023/'

 

mod = 30
tag = tags[0]


if observing_run == 'muCep_2020':
    PA = 133
elif observing_run == 'muCep_2018_02':
    PA = 175
elif observing_run == 'muCep_2018_05':
    PA = 139.5
elif observing_run == 'muCep_2017':
    PA = -121 #
elif observing_run == 'muCep_2023':
    PA = 154
else:
    print('OBSERVING EPOCH NOT IDENTIFIED - STOP')
    cat = dog


PA = PA - 78.9 + 180
PA = -PA


indx_of_cp = np.load(meta_data + 'indx_of_cp.npy')

##### BUG
ucoords_610 = np.load(meta_data + '610/u_coords.npy')
vcoords_610 = np.load(meta_data + '610/v_coords.npy')
xdata_610 = np.arctan(vcoords_610 / ucoords_610)
zdata_610 = np.sqrt(ucoords_610 ** 2 + vcoords_610** 2)
uv_concat_610 = np.concatenate((np.expand_dims(ucoords_610, axis=1), np.expand_dims(vcoords_610, axis=1)), axis=1)
x, y, z = np.ogrid[-154:154, -154:154, -154:154]
xx, yy = np.meshgrid(x.flatten(), y.flatten())
dftm_grid_610 = compute_DFTM1(xx.flatten(), yy.flatten(), uv_concat_610, 610e-9)


ucoords_670 = np.load(meta_data + '670/u_coords.npy')
vcoords_670 = np.load(meta_data + '670/v_coords.npy')
xdata_670 = np.arctan(vcoords_670 / ucoords_670)
zdata_670 = np.sqrt(ucoords_670 ** 2 + vcoords_670** 2)
uv_concat_670 = np.concatenate((np.expand_dims(ucoords_670, axis=1), np.expand_dims(vcoords_670, axis=1)), axis=1)
x, y, z = np.ogrid[-154:154, -154:154, -154:154]
xx, yy = np.meshgrid(x.flatten(), y.flatten())
dftm_grid_670 = compute_DFTM1(xx.flatten(), yy.flatten(), uv_concat_670, 670e-9)


ucoords_720 = np.load(meta_data + '720/u_coords.npy')
vcoords_720 = np.load(meta_data + '720/v_coords.npy')
xdata_720 = np.arctan(vcoords_720 / ucoords_720)
zdata_720 = np.sqrt(ucoords_720 ** 2 + vcoords_720** 2)
uv_concat_720 = np.concatenate((np.expand_dims(ucoords_720, axis=1), np.expand_dims(vcoords_720, axis=1)), axis=1)
x, y, z = np.ogrid[-154:154, -154:154, -154:154]
xx, yy = np.meshgrid(x.flatten(), y.flatten())
dftm_grid_720 = compute_DFTM1(xx.flatten(), yy.flatten(), uv_concat_720, 720e-9)


ucoords_760 = np.load(meta_data + '760/u_coords.npy')
vcoords_760 = np.load(meta_data + '760/v_coords.npy')
xdata_760 = np.arctan(vcoords_760 / ucoords_760)
zdata_760 = np.sqrt(ucoords_760 ** 2 + vcoords_760** 2)
uv_concat_760 = np.concatenate((np.expand_dims(ucoords_760, axis=1), np.expand_dims(vcoords_760, axis=1)), axis=1)
x, y, z = np.ogrid[-154:154, -154:154, -154:154]
xx, yy = np.meshgrid(x.flatten(), y.flatten())
dftm_grid_760 = compute_DFTM1(xx.flatten(), yy.flatten(), uv_concat_760, 760e-9)


plt.figure()
plt.scatter(ucoords_610, vcoords_610)
plt.savefig(savedir + 'uvcoords__{}_{}.pdf'.format(610, tag))
plt.close()

plt.figure()
plt.scatter(ucoords_670, vcoords_670)
plt.savefig(savedir + 'uvcoords__{}_{}.pdf'.format(670, tag))
plt.close()

plt.figure()
plt.scatter(ucoords_720, vcoords_720)
plt.savefig(savedir + 'uvcoords__{}_{}.pdf'.format(720, tag))
plt.close()

plt.figure()
plt.scatter(ucoords_760, vcoords_760)
plt.savefig(savedir + 'uvcoords__{}_{}.pdf'.format(760, tag))
plt.close()


diff_Q_610 =   np.load(meta_data + '610/averageq.npy')**2
diff_U_610 =    np.load(meta_data + '610/averageu.npy')**2
Q_cp_610   =   np.load(meta_data + '610/averageqcp.npy')
U_cp_610   =    np.load(meta_data + '610/averageucp.npy')

diff_Q_670 =   np.load(meta_data + '670/averageq.npy')**2
diff_U_670 =   np.load(meta_data + '670/averageu.npy')**2
Q_cp_670   =   np.load(meta_data + '670/averageqcp.npy')
U_cp_670   =   np.load(meta_data + '670/averageucp.npy')

diff_Q_720 =   np.load(meta_data + '720/averageq.npy')**2
diff_U_720  =   np.load(meta_data + '720/averageu.npy')**2
Q_cp_720    =   np.load(meta_data + '720/averageqcp.npy')
U_cp_720    =   np.load(meta_data + '720/averageucp.npy')

diff_Q_760 =   np.load(meta_data + '760/averageq.npy')**2
diff_U_760 =   np.load(meta_data + '760/averageu.npy')**2
Q_cp_760   =   np.load(meta_data + '760/averageqcp.npy')
U_cp_760    =   np.load(meta_data + '760/averageucp.npy')
 
y_model_this = np.concatenate((diff_Q_610, diff_U_610, Q_cp_610, U_cp_610,
                               diff_Q_670, diff_U_670, Q_cp_670, U_cp_670,
                               diff_Q_720, diff_U_720, Q_cp_720, U_cp_720,
                               diff_Q_760, diff_U_760, Q_cp_760, U_cp_760), axis = 0)



diff_Q_610_err =   np.load(meta_data + '610/errorq.npy')*2.5
diff_U_610_err =   np.load(meta_data + '610/erroru.npy')*2.5
Q_cp_610_err   =    np.load(meta_data + '610/errorqcp.npy')
U_cp_610_err   =    np.load(meta_data + '610/errorucp.npy')

diff_Q_670_err =   np.load(meta_data + '670/errorq.npy')*2.5
diff_U_670_err =   np.load(meta_data + '670/erroru.npy')*2.5
Q_cp_670_err   =   np.load(meta_data + '670/errorqcp.npy')
U_cp_670_err   =   np.load(meta_data + '670/errorucp.npy')

diff_Q_720_err =   np.load(meta_data + '720/errorq.npy')*2.5
diff_U_720_err  =   np.load(meta_data + '720/erroru.npy')*2.5
Q_cp_720_err    =   np.load(meta_data + '720/errorqcp.npy')
U_cp_720_err    =   np.load(meta_data + '720/errorucp.npy')

diff_Q_760_err =   np.load(meta_data + '760/errorq.npy')*2.5
diff_U_760_err =   np.load(meta_data + '760/erroru.npy')*2.5
Q_cp_760_err   =   np.load(meta_data + '760/errorqcp.npy')
U_cp_760_err    =   np.load(meta_data + '760/errorucp.npy')

y_model_err = jnp.concatenate(( diff_Q_610_err , diff_U_610_err,  Q_cp_610_err,   U_cp_610_err ,
                                diff_Q_670_err , diff_U_670_err,  Q_cp_670_err ,  U_cp_670_err  ,
                                diff_Q_720_err,  diff_U_720_err,  Q_cp_720_err  , U_cp_720_err ,
                                diff_Q_760_err , diff_U_760_err , Q_cp_760_err,   U_cp_760_err ), axis = 0)

ydatar = y_model_this
ydatarerr = y_model_err

save =  True
run =    True
plot =  True

# enstatite single, pl
# Forsterite single, pl
#Al2O3 single, pl
# EnstatiteCrystal single, pl

# Run from ForsteriteCrystal - single first

# chemicals = ['Spinel']
# chemicals =  ['Enstatite',   'Olivine', 'pyroxene', 'EnstatiteCrystal', 'ForsteriteCrystal', 'Spinel',
#         'Silica', 'CorundumCrystal', 'mg60_fe40', 'mg70_fe30', 'mg80_fe20', 'mg95_fe05', 'mg0.95_fe0.05_olivine',
#         'mg0.8_fe0.2_olivine', 'mg0.7_fe0.3_olivine', 'mg0.6_fe0.4_olivine']
# chemicals = ['ForsteriteCrystal', 'Spinel']

#'Enstatite', 'Forsterite', 'Al2O3', ,

# chemicals = ['Forsterite', 'EnstatiteCrystal',
#              'Enstatite','ForsteriteCrystal',
#              'Spinel','Silica', 'mg95_fe05',
#              'Al2O3', 'CorundumCrystal' ] # will need to do Enstatite afterwards - you forgot it
#
chemicals = [ 'Enstatite'  ] # will need to do Enstatite afterwards - you forgot it






radius_specs = [(1,   1.1,     -0.1),
                (5,   5.1,     -0.1),
                (10,  10.1,    -0.1),
                (25,  25.1,    -0.1),
                (50,  50.1,    -0.1),
                (100, 100.1,   -0.1),
                (150, 150.1,   -0.1),
                (200, 200.1,   -0.1),
                (250, 250.1,   -0.1),
                (300, 300.1,   -0.1),
                (350, 350.1,   -0.1),
                (400, 400.1,   -0.1),
                (450, 450.1,   -0.1),
                (500, 500.1,   -0.1),
                (550, 550.1,   -0.1),
                (600, 600.1,   -0.1),
                (650, 650.1,   -0.1),
                (700, 700.1,   -0.1),
                (750, 750.1,   -0.1),
                (800, 800.1,   -0.1),
                (850, 850.1,   -0.1),
                (900, 900.1,   -0.1),
                (950, 950.1,   -0.1),
                (1000, 1000.1, -0.1),
                (1100, 1100.1, -0.1),
                (1200, 1200.1, -0.1),
                (1300, 1300.1, -0.1),
                (1400, 1400.1, -0.1),
                (1500, 1500.1, -0.1),
                (1600, 1600.1, -0.1),
                (1700, 1700.1, -0.1),
                (1800, 1800.1, -0.1),
                (1900, 1900.1, -0.1),
                (2000, 2000.1, -0.1),
                 (1,300,-4),
                (1,300,-3),
                (1,300,-2),

                (1,500,-4),
                (1,500,-3),
                (1,500,-2),

                (100, 1000, -4),
                (100, 1000, -3),
                (100, 1000, -2),

                (200, 1000, -4),
                (200, 1000, -3),
                (200, 1000, -2),

                (250, 1000, -4),
                (250, 1000, -3),
                (250, 1000, -2),

                (300, 1000, -4),
                (300, 1000, -3),
                (300, 1000, -2),


                (400, 1000, -4),
                (400, 1000, -3),
                (400, 1000, -2),

                (500, 1000, -4),
                (500, 1000, -3),
                (500, 1000, -2),



                (100, 2000, -4),
                (100, 2000, -3),
                (100, 2000, -2),

                (200, 2000, -4),
                (200, 2000, -3),
                (200, 2000, -2),

                (250, 2000, -4),
                (250, 2000, -3),
                (250, 2000, -2),

                (300, 2000, -4),
                (300, 2000, -3),
                (300, 2000, -2),

                (400, 2000, -4),
                (400, 2000, -3),
                (400, 2000, -2),

                (500, 2000, -4),
                (500, 2000, -3),
                (500, 2000, -2),]


# radius_specs = [(10,  10.1,    -0.1)]

save_rad_keys = [f"{a}-{b}-{abs(p)}" for a,b,p in radius_specs]
wvs = [610, 670, 720, 760]
chi_map = np.zeros((len(chemicals), len(save_rad_keys)))
bayes_map = np.zeros((len(chemicals), len(save_rad_keys)))
 
for mod in [38]:


    for kkkk in range(0, 1):
        chem_i = 0

        for chem in chemicals:
            grain_i = 0
            material = chem

            for grain_size in save_rad_keys:



                # tag = '_{}_{}_SINGLE760FIT'.format(chem, grain_size)
                # tag = '2023_mucep_SINGLE760FIT'
                tag = '_{}_{}_'.format(chem, grain_size)
                tag = '2023_mucep_'
                observing_run =  'muCep_2023'
                print(observing_run)
                print(tag)

                tag = tag + '_' + chem + '_' + str(grain_size) + '_'
                print('******************')
                print(tag)
                print(observing_run)
                print(model_class[mod])
                print(models[mod])
                print('******************')

                if chem == 'FE50':

                    materials = ['Enstatite', 'Forsterite']
                    grids = load_average_fixed_grids_multi_jax(materials, grain_size, grids_loc)


                    scat_V45_610 = grids['scat_V45_610']
                    scat_H45_610 = grids['scat_H45_610']
                    scat_V_610 = grids['scat_V_610']
                    scat_H_610 = grids['scat_H_610']

                    scat_V45_670 = grids['scat_V45_670']
                    scat_H45_670 = grids['scat_H45_670']
                    scat_V_670 = grids['scat_V_670']
                    scat_H_670 = grids['scat_H_670']

                    scat_V45_720 = grids['scat_V45_720']
                    scat_H45_720 = grids['scat_H45_720']
                    scat_V_720 = grids['scat_V_720']
                    scat_H_720 = grids['scat_H_720']

                    scat_V45_760 = grids['scat_V45_760']
                    scat_H45_760 = grids['scat_H45_760']
                    scat_V_760 = grids['scat_V_760']
                    scat_H_760 = grids['scat_H_760']

                elif chem == 'FC50':

                    materials = ['Al2O3', 'Forsterite']
                    grids = load_average_fixed_grids_multi_jax(materials, grain_size, grids_loc)

                    scat_V45_610 = grids['scat_V45_610']
                    scat_H45_610 = grids['scat_H45_610']
                    scat_V_610 = grids['scat_V_610']
                    scat_H_610 = grids['scat_H_610']

                    scat_V45_670 = grids['scat_V45_670']
                    scat_H45_670 = grids['scat_H45_670']
                    scat_V_670 = grids['scat_V_670']
                    scat_H_670 = grids['scat_H_670']

                    scat_V45_720 = grids['scat_V45_720']
                    scat_H45_720 = grids['scat_H45_720']
                    scat_V_720 = grids['scat_V_720']
                    scat_H_720 = grids['scat_H_720']

                    scat_V45_760 = grids['scat_V45_760']
                    scat_H45_760 = grids['scat_H45_760']
                    scat_V_760 = grids['scat_V_760']
                    scat_H_760 = grids['scat_H_760']



                elif chem == 'EC50':

                    materials = ['Al2O3', 'Enstatite']
                    grids = load_average_fixed_grids_multi_jax(materials, grain_size, grids_loc)

                    scat_V45_610 = grids['scat_V45_610']
                    scat_H45_610 = grids['scat_H45_610']
                    scat_V_610 = grids['scat_V_610']
                    scat_H_610 = grids['scat_H_610']

                    scat_V45_670 = grids['scat_V45_670']
                    scat_H45_670 = grids['scat_H45_670']
                    scat_V_670 = grids['scat_V_670']
                    scat_H_670 = grids['scat_H_670']

                    scat_V45_720 = grids['scat_V45_720']
                    scat_H45_720 = grids['scat_H45_720']
                    scat_V_720 = grids['scat_V_720']
                    scat_H_720 = grids['scat_H_720']

                    scat_V45_760 = grids['scat_V45_760']
                    scat_H45_760 = grids['scat_H45_760']
                    scat_V_760 = grids['scat_V_760']
                    scat_H_760 = grids['scat_H_760']



                elif chem == 'FEC33':

                    materials = ['Al2O3', 'Enstatite', 'Forsterite']
                    grids = load_average_fixed_grids_multi_jax(materials, grain_size, grids_loc)

                    scat_V45_610 = grids['scat_V45_610']
                    scat_H45_610 = grids['scat_H45_610']
                    scat_V_610 = grids['scat_V_610']
                    scat_H_610 = grids['scat_H_610']

                    scat_V45_670 = grids['scat_V45_670']
                    scat_H45_670 = grids['scat_H45_670']
                    scat_V_670 = grids['scat_V_670']
                    scat_H_670 = grids['scat_H_670']

                    scat_V45_720 = grids['scat_V45_720']
                    scat_H45_720 = grids['scat_H45_720']
                    scat_V_720 = grids['scat_V_720']
                    scat_H_720 = grids['scat_H_720']

                    scat_V45_760 = grids['scat_V45_760']
                    scat_H45_760 = grids['scat_H45_760']
                    scat_V_760 = grids['scat_V_760']
                    scat_H_760 = grids['scat_H_760']




                else:

                    scat_V45_610 = np.load( grids_loc + 'H_scat_m{}_r{}_w610.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #
                    scat_H45_610 = np.load(grids_loc + 'V_scat_m{}_r{}_w610.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #
                    scat_V_610 = np.load(grids_loc + 'H45_scat_m{}_r{}_w610.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #
                    scat_H_610 = np.load(grids_loc + 'V45_scat_m{}_r{}_w610.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #

                    scat_V45_670 = np.load(grids_loc + 'H_scat_m{}_r{}_w670.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #
                    scat_H45_670 = np.load(grids_loc + 'V_scat_m{}_r{}_w670.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #
                    scat_V_670 = np.load(grids_loc + 'H45_scat_m{}_r{}_w670.npy'.format(
                        material, grain_size))[0:308, 0:308, 0:308]  #
                    scat_H_670 = np.load(grids_loc + 'V45_scat_m{}_r{}_w670.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #

                    scat_V45_720 = np.load(grids_loc + 'H_scat_m{}_r{}_w720.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #
                    scat_H45_720 = np.load(grids_loc + 'V_scat_m{}_r{}_w720.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #
                    scat_V_720 = np.load(grids_loc + 'H45_scat_m{}_r{}_w720.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #
                    scat_H_720 = np.load(grids_loc + 'V45_scat_m{}_r{}_w720.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #

                    scat_V45_760 = np.load(grids_loc + 'H_scat_m{}_r{}_w760.npy'.format(
                          material, grain_size))[0:308, 0:308, 0:308]  #
                    scat_H45_760 = np.load(grids_loc + 'V_scat_m{}_r{}_w760.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #
                    scat_V_760 = np.load(grids_loc + 'H45_scat_m{}_r{}_w760.npy'.format(
                            material, grain_size))[0:308, 0:308, 0:308]  #
                    scat_H_760 = np.load(grids_loc + 'V45_scat_m{}_r{}_w760.npy'.format(
                            material , grain_size))[0:308, 0:308, 0:308]

                    #
                # extract 2D slices
                slice_H_610 = scat_H_610.sum(axis=2)
                slice_V_610 = scat_V_610.sum(axis=2)
                slice_H45_610 = scat_H45_610.sum(axis=2)
                slice_V45_610 = scat_V45_610.sum(axis=2)

                slice_H_670 = scat_H_670.sum(axis=2)
                slice_V_670 = scat_V_670.sum(axis=2)
                slice_H45_670 = scat_H45_670.sum(axis=2)
                slice_V45_670 = scat_V45_670.sum(axis=2)

                slice_H_720 = scat_H_720.sum(axis=2)
                slice_V_720 = scat_V_720.sum(axis=2)
                slice_H45_720 = scat_H45_720.sum(axis=2)
                slice_V45_720 = scat_V45_720.sum(axis=2)

                slice_H_760 = scat_H_760.sum(axis=2)
                slice_V_760 = scat_V_760.sum(axis=2)
                slice_H45_760 = scat_H45_760.sum(axis=2)
                slice_V45_760 = scat_V45_760.sum(axis=2)



                fig, axes = plt.subplots(4, 4, figsize=(12, 12), sharex=True, sharey=True)

                # Row 0 → 610 nm
                im00 = axes[0, 0].imshow(slice_H_610 - slice_V_610, origin='lower', cmap='viridis')
                axes[0, 1].imshow(slice_V_610, origin='lower', cmap='viridis')
                axes[0, 2].imshow(slice_H45_610 - slice_V45_610 , origin='lower', cmap='viridis')
                axes[0, 3].imshow(slice_V45_610, origin='lower', cmap='viridis')

                # Row 1 → 670 nm
                axes[1, 0].imshow(slice_H_670 - slice_V_670, origin='lower', cmap='viridis')
                axes[1, 1].imshow(slice_V_670, origin='lower', cmap='viridis')
                axes[1, 2].imshow(slice_H45_670 - slice_V45_670, origin='lower', cmap='viridis')
                axes[1, 3].imshow(slice_V45_670, origin='lower', cmap='viridis')

                # Row 2 → 720 nm
                axes[2, 0].imshow(slice_H_720 - slice_V_720, origin='lower', cmap='viridis')
                axes[2, 1].imshow(slice_V_720, origin='lower', cmap='viridis')
                axes[2, 2].imshow(slice_H45_720 - slice_V45_720, origin='lower', cmap='viridis')
                axes[2, 3].imshow(slice_V45_720, origin='lower', cmap='viridis')

                # Row 3 → 760 nm
                axes[3, 0].imshow(slice_H_760 - slice_V_760, origin='lower', cmap='viridis')
                axes[3, 1].imshow(slice_V_760, origin='lower', cmap='viridis')
                axes[3, 2].imshow(slice_H45_760 - slice_V45_760, origin='lower', cmap='viridis')
                axes[3, 3].imshow(slice_V45_760, origin='lower', cmap='viridis')

                # annotate column titles (pol states) on top row
                axes[0, 0].set_title('H', pad=6)
                axes[0, 1].set_title('V', pad=6)
                axes[0, 2].set_title('H45', pad=6)
                axes[0, 3].set_title('V45', pad=6)

                # annotate row labels (wavelength) on first column
                axes[0, 0].set_ylabel('610 nm', fontsize=12)
                axes[1, 0].set_ylabel('670 nm', fontsize=12)
                axes[2, 0].set_ylabel('720 nm', fontsize=12)
                axes[3, 0].set_ylabel('760 nm', fontsize=12)

                # hide all axis ticks
                for i in range(4):
                    for j in range(4):
                        axes[i, j].set_xticks([])
                        axes[i, j].set_yticks([])

                # single colorbar on the right
                cbar = fig.colorbar(im00, ax=axes, orientation='vertical',
                                    fraction=0.02, pad=0.02)
                cbar.set_label('Scattering strength (arb. units)')

                plt.tight_layout()
                plt.savefig(savedir + 'scattering_grids.pdf')




                ######

                arrays = [
                    'scat_V_610', 'scat_H_610', 'scat_V45_610', 'scat_H45_610',
                    'scat_V_670', 'scat_H_670', 'scat_V45_670', 'scat_H45_670',
                    'scat_V_720', 'scat_H_720', 'scat_V45_720', 'scat_H45_720',
                    'scat_V_760', 'scat_H_760', 'scat_V45_760', 'scat_H45_760',
                ]


                # Replace any NaNs in these arrays with 0
                for name in arrays:
                    arr = globals()[name]
                    # jnp.nan_to_num will set NaNs to 0.0
                    cleaned = np.nan_to_num(arr, nan=0.0)
                    globals()[name] = cleaned


                if observing_run == 'muCep_2020':
                    PA = 133
                elif observing_run == 'muCep_2018_02':
                    PA = 175
                elif observing_run == 'muCep_2018_05':
                    PA = 139.5
                elif observing_run == 'muCep_2017':
                    PA = -121
                elif observing_run == 'muCep_2023':
                    PA = 154
                    print('Observing run is muCep_2023, PA is {}'.format(PA))

                else:
                    print('OBSERVING EPOCH NOT IDENTIFIED - STOP')
                    cat = dog

                PA = PA - 78.9 + 180
                PA = -PA
 
                meta_data = 'geometric_models_data/{}/'.format(observing_run)



                model_params = {'image_size': image_size,
                                'PA': PA,
                                'pixel_ratio': pixel_ratio,
                                'size_biggest_baseline_m': size_biggest_baseline_m,
                                'wavelength': wavelength,
                                'u_coords_610': ucoords_610,
                                'v_coords_610': vcoords_610,
                                'u_coords_670': ucoords_670,
                                'v_coords_670': vcoords_670,
                                'u_coords_720': ucoords_720,
                                'v_coords_720': vcoords_720,
                                'u_coords_760': ucoords_760,
                                'v_coords_760': vcoords_760,

                                'xdata_760': xdata_760,
                                'zdata_760': zdata_760,

                                'xdata_720': xdata_720,
                                'zdata_720': zdata_720,

                                'xdata_670': xdata_670,
                                'zdata_670': zdata_670,

                                'xdata_610': xdata_610,
                                'zdata_610': zdata_610,


                                'indx_of_cp':indx_of_cp,

                                'dftm_grid_610': dftm_grid_610,
                                'dftm_grid_670': dftm_grid_670,
                                'dftm_grid_720': dftm_grid_720,
                                'dftm_grid_760': dftm_grid_760,


                                'H_scat_610': scat_H_610,
                                'V_scat_610': scat_V_610,
                                'H45_scat_610': scat_H45_610,
                                'V45_scat_610': scat_V45_610,

                                'H_scat_670': scat_H_670,
                                'V_scat_670': scat_V_670,
                                'H45_scat_670': scat_H45_670,
                                'V45_scat_670': scat_V45_670,

                                'H_scat_720': scat_H_720,
                                'V_scat_720': scat_V_720,
                                'H45_scat_720': scat_H45_720,
                                'V45_scat_720': scat_V45_720,

                                'H_scat_760': scat_H_760,
                                'V_scat_760': scat_V_760,
                                'H45_scat_760': scat_H45_760,
                                'V45_scat_760': scat_V45_760,

                                'enhancement_amp': 2,
                                'enhancement_size': 90,
                                'enhancement_loc': 180,

                                'thick1': 1,
                                'thick2': 3,
                                'thicka2':2,
                                'thickb' :2,
                                'thicka':2,
                                'thickb':2,


                                'star_radius': star_radius,

                                'a': 15,
                                'a2': 11,
                                'a3': 11,
                                'b': 15,
                                'b2': 11,
                                'b3': 11,
                                'c2': 11,
                                'c3': 11,
                                'c': 12,
                                'h': 0,
                                'k': 0,
                                'n': 0,

                                'bright_location_theta': np.pi/8,
                                'bright_location_phi': np.pi / 8,
                                'bright_size': np.pi/8,
                                'bright_mag': 4,

                                'ellipse_scale': 2,
                                'ellipse_scale2': 4,
                                'ellipse_contrast_ratio2': 3,
                                'ellipse_contrast_ratio3': 3,
                                'alpha': 20,
                                'alpha2':20,
                                'dust_star_contrast': 8.5,
                                'ellipse_contrast_ratio': 0.5,


                                'ydata_real' : ydatar,
                                'ydata_real_err': ydatarerr,

                                'dust_shape': model_class[mod],
                                'dust_pl_const': 1,
                                'dust_pl_exp': 5,
                                'dust_pl_exp2': 8,
                                'dust_c': 1,
                                'shell_thickness': 3,

                                'blob_radial_distance': 10,
                                'theta_blob': 1.5,
                                'phi_blob': 1.5,
                                'r_blob': 20,
                                'blob_contrast':100,

                                'blob_radial_distance2': 10,
                                'theta_blob2': 1.5,
                                'phi_blob2': 1.5,
                                'r_blob2': 10,
                                'blob_contrast2': 0.5,

                                'p0': p0s[mod],
                                'changing_by_theta': changing_by_thetas[mod],
                                'priors': priors[mod],
                                'changing_by_setting': changing_by_settings[mod],
                                'filename': models[mod]}



                num_steps_run = num_steps_run_list[mod]
                discard = num_steps_discard_list[mod]

                stellar_object = geometric_star(model_params)
                stellar_object.make_3D_model()
                y_model_this = stellar_object.simulate_nrm()

        

                np.save(savedir + 'true_obs_{}_model_{}.npy'.format(tag, mod), stellar_object.ydata_real)
                np.save(savedir + 'true_obs_err_{}_model_{}.npy'.format(tag, mod), stellar_object.ydata_real_err)


                print(model_params['filename'])
                nwalkers = len(model_params['changing_by_theta']) * 3


                model = model_params['filename']
                star_radius = model_params['star_radius']
                ndim = len(model_params['p0'])
                pos = [model_params['p0'] + 1e-4*onp.random.randn(ndim) for i in range(nwalkers)]



                injection_test = False

                if injection_test == True:
                    num_steps = 17000
                    plt.figure(figsize=(16, 3))


                    for k in range(5):

                        for i in range(len(model_params['changing_by_theta'])): # this will change the values of the parameters
                            vari = model_params['changing_by_theta'][i]
                            priors_d =  model_params['priors'][i]


                            thing = onp.random.uniform(priors_d[0], priors_d[1])   #np.random.rand() * (priors_d[1] - priors_d[0])        #priors_d[0], priors_d[1])
                            model_params[vari] = thing

                            print('Changed thing is')
                            print(vari)
                            print(model_params[vari])


                        for i in range(len(model_params['changing_by_setting'])): # this will then update the relevant ones accordingly
                            vari = model_params['changing_by_setting'][i]
                            model_params[vari[0]] = model_params[vari[1]]



                        stellar_object = geometric_star(model_params)
                        stellar_object.make_3D_model()
                        model_params['ydata_real'] = stellar_object.simulate_nrm() # whats this doing


                        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_model ,
                        moves=[(emcee.moves.DEMove(), 0.8),
                                (emcee.moves.DESnookerMove(), 0.2),])

                        state = sampler.run_mcmc(pos, num_steps, progress = True)
                        not_flat_samples = sampler.get_chain(flat=False, discard = discard)#00)
                        flat_samples = sampler.get_chain(flat = False, discard = discard)#00)

                        fig = corner.corner(flat_samples, truth_color='b', labels=model_params['changing_by_theta'], show_titles=True, hist_bin_factor=2)
                        plt.savefig(savedir + 'corner_injection_{}.pdf'.format(k))

                        MLEs = np.zeros((len(model_params['p0'])))

                        for j in range(len(MLEs)):

                            MLEs = MLEs.at[j].set(stats.mode(not_flat_samples[:, :, j]).mode[0])

                        print('**************************************')

                        for o in range(len(MLEs)):
                            plt.subplot(1, len(MLEs), o+1)
                            var_name = model_params['changing_by_theta'][o]
                            print('The truth')
                            print(model_params[var_name])
                            print('MLEs')
                            print(MLEs[o])
                            plt.scatter(model_params[var_name], MLEs[o])
                            plt.title(var_name)
                            plt.xlabel('True')
                            plt.ylabel('MLEs')
                            plt.xlim(model_params['priors'][o][0], model_params['priors'][o][1])
                            plt.ylim(model_params['priors'][o][0], model_params['priors'][o][1])


                    plt.tight_layout()
                    plt.savefig(savedir + 'injection_testing_{}_{}.pdf'.format(model, tag))



                if run == True:

                    time_start = time.time()

                    num_steps = num_steps_run
                    original = os.getcwd() + '/emcee_fit_package.py'
                    target = savedir_fits + 'emcee_fit_package_' + model_params['filename'] + '_' + tag + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_walkers.py'
                    shutil.copyfile(original, target)

                    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_model ,
                    moves=[(emcee.moves.DEMove(), 0.8),
                            (emcee.moves.DESnookerMove(), 0.2),])

                    state = sampler.run_mcmc(pos, num_steps, progress = True)

                    time_end = time.time()
                    print('Total time taken is {} seconds'.format(time_end - time_start))


                else:
                    num_steps = num_steps_run
                    with open(savedir_fits + model_params['filename'] + '_' + tag + '_' + str(star_radius) + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_sampler.pkl', 'rb') as file:
                        sampler = pickle.load(file)
                    with open(savedir_fits + model_params['filename'] + '_' + tag + '_' + str(star_radius) + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_state.pkl', 'rb') as file:
                        state = pickle.load(file)
                    with open(savedir_fits + model_params['filename'] + '_' + tag + '_' + str(star_radius) + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_smodelparams.pkl', 'rb') as file:
                        model_params = pickle.load(file)



                if save == True:
                    filep = open(savedir_fits + model_params['filename'] + '_' + tag + '_' + str(star_radius) + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_sampler.pkl', 'wb')
                    pickle.dump(sampler, filep)
                    filep.close()
                    filep = open(savedir_fits + model_params['filename'] + '_' + tag + '_' + str(star_radius) + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_state.pkl', 'wb')
                    pickle.dump(state, filep)
                    filep.close()
                    filep = open(savedir_fits + model_params['filename'] + '_' + tag + '_' + str(star_radius) + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_smodelparams.pkl', 'wb')
                    pickle.dump(model_params, filep)
                    filep.close()





                log_likelihoods = sampler.get_log_prob(flat=True, discard = discard)  # From emcee sampler
                log_likelihood_max = np.max(log_likelihoods)      # Maximum log-likelihood
                num_params =  len(model_params['p0'])           # Number of free parameters
                num_data_points = len(ydatar)              # Size of your dataset

                # Compute BIC
                BIC = -2 * log_likelihood_max + num_params * np.log(num_data_points)
                print(f">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> BIC: {BIC}")

                flat_samples = sampler.get_chain(flat=True, discard=discard)




                logZ = log_evidence_laplace_mixed(flat_samples, log_likelihoods, lnprob_model,  prior)  #log_evidence_laplace_mixed(samples, log_probs, log_prob_fn, prior_fn, adaptive_mode_detection=True):

                bayes_map = bayes_map.at[chem_i, grain_i].set(logZ)





                not_flat_samples = sampler.get_chain(flat=False, discard=discard)

                # Initialize the MLEs array using JAX array
                MLEs = np.zeros((len(model_params['p0'])))
                perr = np.zeros((len(model_params['p0'])))

                # Iterate over each parameter dimension
                for i in range(len(MLEs)):
                    # Create a histogram from the samples
                    counts, bin_edges, patches = plt.hist(not_flat_samples[:, :, i].flatten(), bins=35, edgecolor='black')

                    # Find the bin with the maximum count
                    max_bin_idx = np.argmax(counts)

                    # Ensure the index is within bounds
                    if max_bin_idx < len(bin_edges) - 1:
                        # MLE is the center of the bin with the maximum count
                        mle_value = 0.5 * (bin_edges[max_bin_idx] + bin_edges[max_bin_idx + 1])
                    else:
                        # If the index is out of bounds, use the last valid bin edge
                        mle_value = bin_edges[max_bin_idx]

                    # Use the JAX .at[] method to set the value in the MLEs array
                    MLEs = MLEs.at[i].set(mle_value)
                    perr = perr.at[i].set(np.std(not_flat_samples[:, :, i].flatten()))



                flat_samples = sampler.get_chain(flat=True, discard = discard)
                num_params = np.shape(flat_samples)[1] - 1
                param_names = model_params['changing_by_theta']


                param_values = MLEs
                for i in range(len(param_names)):
                    vari = param_names[i]
                    model_params[vari] = param_values[i]

                for i in range(len(model_params['changing_by_setting'])):
                    vari = model_params['changing_by_setting'][i]
                    model_params[vari[0]] = model_params[vari[1]]



                print(model_params['dust_shape'])  #model_class[mod]
                stellar_object = geometric_star(model_params)
                stellar_object.make_3D_model()
                y_model_this = stellar_object.simulate_nrm()


                Q_vals_real = onp.concatenate((stellar_object.ydata_real[0:153], stellar_object.ydata_real[153*2:153*2 + 816]), axis = 0)
                U_vals_real = onp.concatenate((stellar_object.ydata_real[153:153*2], stellar_object.ydata_real[153*2 + 816:153*2 + 816*2]), axis = 0)

                Q_vals_model =  onp.concatenate((stellar_object.y_model[0:153], stellar_object.y_model[153*2:153*2 + 816]), axis = 0)
                U_vals_model = onp.concatenate((stellar_object.y_model[153:153*2], stellar_object.y_model[153*2 + 816:153*2 + 816*2]), axis = 0)

                Q_vals_error =  onp.concatenate((stellar_object.ydata_real_err[0:153], stellar_object.ydata_real_err[153*2:153*2 + 816]), axis = 0)
                U_vals_error = onp.concatenate((stellar_object.ydata_real_err[153:153*2], stellar_object.ydata_real_err[153*2 + 816:153*2 + 816*2]), axis = 0)

                chisqu_Q = uf.chi_squ_red(Q_vals_real, Q_vals_model, Q_vals_error, len(model_params['changing_by_theta']))
                chisqu_U = uf.chi_squ_red(U_vals_real, U_vals_model, U_vals_error, len(model_params['changing_by_theta']))


                np.save(savedir + 'modelMLE_obs_{}_model_{}.npy'.format(tag, mod), stellar_object.y_model)

                final_chi_red = (chisqu_Q + chisqu_U)/2

                numpoints = 153*2 + 816*2
                real_data = stellar_object.ydata_real[-numpoints:]
                model_data = stellar_object.y_model[-numpoints:]
                error_data = stellar_object.ydata_real_err[-numpoints:]


                chisqu_reduced = uf.chi_squ_red(real_data, model_data, error_data, len(model_params['changing_by_theta']))
                print('The chi squared reduced value is  {}'.format(chisqu_reduced))



                chisqu = uf.chi_squ(real_data, model_data, error_data)
                print('The chi squared value is  {}'.format(chisqu))


                chi_map = chi_map.at[chem_i, grain_i].set(chisqu_reduced)

                np.save(savedir + 'bayes_evidence_{}_{}.npy'.format(tag, mod), logZ)
                np.save(savedir + 'chisqu_reduced_{}_{}.npy'.format(tag, mod), chisqu_reduced)

                print('Log evidence is {}'.format(logZ))
                print('The chi squared value is  {}'.format(chisqu_reduced))


                if final_chi_red > 1:

                    nobs = len(stellar_object.ydata_real)
                    nparams = len(model_params['changing_by_theta'])
                    scaling_factor = np.sqrt(chisqu/(nobs - nparams))

                if plot:
                    plt.figure(figsize=(6, 6))
                    fig = corner.corner(flat_samples, truth_color = 'b', labels = param_names, show_titles = True, hist_bin_factor = 2, truths=MLEs)


                    if final_chi_red > 1:
                        for i in range(len(param_names)):
                            name = param_names[i]
                            err = perr[i] * scaling_factor
                            print('**************')
                            print(perr[i])
                            print(err)
                            print('**************')
                            ax = fig.axes[i * (len(param_names) + 1)]  # Get the diagonal axes
                            title = f'{name} = {MLEs[i]:.2f} ± {err:.2f}'
                            ax.set_title(title)
                            ax.set_title(title, fontsize=9)



                    fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

                    # plt.tight_layout()
                    plt.savefig(savedir + 'corner_plot_{}_{}.pdf'.format(model, tag))
                    plt.close()

                if plot:
                    fig, ax = plt.subplots(4, 2*4, gridspec_kw={'height_ratios': [1, 1, 1, 1]})
                    fig.set_size_inches(19, 9.5)

                    im3 = ax[1,0].scatter(stellar_object.xdata_610, stellar_object.ydata_real[0:153], c=abs(stellar_object.zdata_610), cmap='jet', marker='x', label = 'Real', s = 10, linewidths=0.5)
                    ax[1,0].errorbar(stellar_object.xdata_610, stellar_object.ydata_real[0:153], yerr = stellar_object.ydata_real_err[0:153], fmt = 'none', ecolor= 'lightgrey', alpha = 0.4, elinewidth=0.5)
                    ax[1,0].plot(stellar_object.xdata_610, np.mean(stellar_object.ydata_real[0:153]) * np.ones(np.shape(stellar_object.xdata_610)), label = 'Real $\mu$')
                    im4 = ax[1,0].scatter(stellar_object.xdata_610, stellar_object.y_model[0:153], c=abs(stellar_object.zdata_610), cmap='jet', marker='o', alpha=0.1, label='Model', vmin=np.min(stellar_object.zdata_610), vmax=np.max(stellar_object.zdata_610),  s = 5)
                    ax[1,0].plot(stellar_object.xdata_610, np.mean(stellar_object.y_model[0:153]) * np.ones(np.shape(stellar_object.y_model[0:153])), alpha=0.2, label='Model $\mu$')

                    chisqu_t = uf.chi_squ_red(stellar_object.ydata_real[0:153], stellar_object.y_model[0:153], stellar_object.ydata_real_err[0:153], len(model_params['changing_by_theta']))
                    ax[1, 0].set_title('Modelled and Real Stokes Q $V^2$, $\chi_{{red}}^2$ {:.2f}'.format(chisqu_t))
                    cbar = fig.colorbar(im3, ax=ax[1,0], label = "Baseline Length (m)")
                    cbar.set_label('Baseline Length (m)')
                    ax[1,0].set_ylabel('Differential $V^2$')
                    ax[1,0].set_xlabel('Azimuth Angle (rad)')


                    im4 = ax[1,1].scatter(stellar_object.xdata_610, stellar_object.ydata_real[153:153 * 2], c=abs(stellar_object.zdata_610), cmap='jet', marker='x', label = 'Real',  s = 10, linewidths=0.5)
                    ax[1,1].errorbar(stellar_object.xdata_610, stellar_object.ydata_real[153:153 * 2], yerr = stellar_object.ydata_real_err[153:153 * 2], fmt = 'none', ecolor = 'lightgrey', alpha = 0.4, elinewidth=0.5)
                    ax[1,1].plot(stellar_object.xdata_610, np.mean(stellar_object.ydata_real[153:153 * 2]) * np.ones(np.shape(stellar_object.xdata_610)), label = 'Real $\mu$')
                    im4 = ax[1,1].scatter(stellar_object.xdata_610, stellar_object.y_model[153:153 * 2], c=abs(stellar_object.zdata_610), cmap='jet', marker='o', alpha=0.1, label='Model', vmin=np.min(stellar_object.zdata_610), vmax=np.max(stellar_object.zdata_610), s = 5)
                    ax[1,1].plot(stellar_object.xdata_610, np.mean(stellar_object.y_model[153:153 * 2]) * np.ones(np.shape(stellar_object.y_model[153:153 * 2])), alpha=0.2, label='Model $\mu$')

                    chisqu_t = uf.chi_squ_red(stellar_object.ydata_real[153:153 * 2], stellar_object.y_model[153:153 * 2], stellar_object.ydata_real_err[153:153 * 2], len(model_params['changing_by_theta']))
                    ax[1, 1].set_title('Modelled and Real Stokes U $V^2$, $\chi_{{red}}^2$ {:.2f}'.format(chisqu_t))
                    cbar = fig.colorbar(im3, ax=ax[1,1], label = "Baseline Length (m)")
                    cbar.set_label('Baseline Length (m)')
                    ax[1,1].set_ylabel('Differential $V^2$')
                    ax[1,1].set_xlabel('Azimuth Angle (rad)')
                    # ax[1,1].legend(loc='best')


                    big_thing = np.max(np.array([np.abs(np.min(stellar_object.image_Q_610)), np.abs(np.max(stellar_object.image_Q_610))]))
                    im7 = ax[0,0].imshow((stellar_object.image_Q_610), cmap = 'seismic', clim = [-big_thing, big_thing])#.sum(axis=2)))
                    ax[0,0].set_title('Stokes Q Image 610')
                    ax[0,0].set_xlabel('x (mas)')
                    ax[0,0].set_ylabel('y (mas)')
                    plt.tight_layout()
                    cbar = fig.colorbar(im7, ax=ax[0,0], label = "Baseline Length (m)")
                    cbar.set_label('Normalised Polarised Flux')

                    big_thing = np.max(np.array([np.abs(np.min(stellar_object.image_U_610)), np.abs(np.max(stellar_object.image_U_610))]))
                    im77 = ax[0,1].imshow((stellar_object.image_U_610),  cmap = 'seismic', clim = [-big_thing, big_thing])
                    ax[0,1].set_title('Stokes U Image 610')
                    ax[0,1].set_xlabel('x (mas)')
                    ax[0,1].set_ylabel('y (mas)')
                    cbar = fig.colorbar(im7, ax=ax[0,1], label = "Baseline Length (m)")
                    cbar.set_label('Normalised Polarised Flux')


                    im5 = ax[2,0].scatter(stellar_object.ydata_real[0:153], stellar_object.y_model[0:153],
                                        c=abs(stellar_object.zdata_610), cmap='jet',
                                        marker='o',  label='M', vmin=np.min(stellar_object.zdata_610),
                                        vmax=np.max(stellar_object.zdata_610))
                    ax[2,0].set_title('Correlation between real and modelled $V^2$')
                    length_thing = np.arange(np.min(stellar_object.ydata_real[0:153]), np.max(stellar_object.ydata_real[0:153]), 0.0001)
                    ax[2,0].scatter(length_thing, length_thing, marker =  '.', c = 'k', vmin=np.min(stellar_object.zdata_610),
                                            vmax=np.max(stellar_object.zdata_610), label = 'x = y')
                    cbar = fig.colorbar(im5, ax=ax[2,0], label = "Baseline Length (m)")
                    ax[2,0].set_ylabel('Modelled Differential $V^2$')
                    ax[2,0].set_xlabel('Real Differential $V^2$')


                    im6 = ax[2,1].scatter(stellar_object.ydata_real[153:153 * 2], stellar_object.y_model[153:153 * 2],
                                        c=abs(stellar_object.zdata_610), cmap='jet',
                                        marker='o', label='M', vmin=np.min(stellar_object.zdata_610),
                                        vmax=np.max(stellar_object.zdata_610))
                    ax[2,1].set_title('Correlation between real and modelled $V^2$')
                    length_thing = np.arange(np.min(stellar_object.ydata_real[153:153 * 2]), np.max(stellar_object.ydata_real[153:153 * 2]), 0.0001)
                    ax[2,1].scatter(length_thing, length_thing, marker = '.', c = 'k', vmin=np.min(stellar_object.zdata_610),
                                            vmax=np.max(stellar_object.zdata_610), label = 'x = y')
                    ax[2,1].set_ylabel('Modelled Differential $V^2$')
                    ax[2,1].set_xlabel('Real Differential $V^2$')
                    # ax[2,1].legend(loc='best')
                    cbar = fig.colorbar(im6, ax=ax[2,1], label = "Baseline Length (m)")

                #############################################################


                    im5 = ax[3,0].scatter(stellar_object.ydata_real[153*2:153*2 + 816], stellar_object.y_model[153*2:153*2 + 816],  marker='o',  label='M' )
                    ax[3,0].set_title('Correlation between real and modelled $CP$')
                    length_thing = np.arange(np.min(stellar_object.ydata_real[153*2:153*2 + 816]), np.max(stellar_object.ydata_real[153*2:153*2 + 816]), 0.0001)
                    ax[3,0].scatter(length_thing, length_thing, marker =  '.', c = 'k',  label = 'x = y')
                    cbar = fig.colorbar(im5, ax=ax[3,0], label = "Baseline Length (m)")
                    ax[3,0].set_ylabel('Modelled Differential CP')
                    ax[3,0].set_xlabel('Real Differential CP')


                    im6 = ax[3,1].scatter(stellar_object.ydata_real[153*2 + 816:153*2 + 816*2], stellar_object.y_model[153*2 + 816:153*2 + 816*2],  marker='o', label='M' )
                    ax[3,1].set_title('Correlation between real and modelled CP')
                    length_thing = np.arange(np.min(stellar_object.ydata_real[153*2 + 816:153*2 + 816*2]), np.max(stellar_object.ydata_real[153*2 + 816:153*2 + 816*2]), 0.0001)
                    ax[3,1].scatter(length_thing, length_thing, marker = '.', c = 'k', label = 'x = y')
                    ax[3,1].set_ylabel('Modelled Differential CP')
                    ax[3,1].set_xlabel('Real Differential CP')





                    #########################################################

                    start = 153 * 2 + 816 * 2
                    q_slice = slice(start, start + 153)
                    u_slice = slice(start + 153, start + 153 * 2)

                    # Stokes Q plot for segment 2
                    im3 = ax[1, 2].scatter(stellar_object.xdata_670, stellar_object.ydata_real[q_slice], c=abs(stellar_object.zdata_670), cmap='jet', marker='x', label='Real',  s = 10, linewidths=0.5)
                    ax[1, 2].errorbar(stellar_object.xdata_670, stellar_object.ydata_real[q_slice], yerr=stellar_object.ydata_real_err[q_slice], fmt='none', ecolor='lightgrey', alpha = 0.4, elinewidth=0.5)
                    ax[1, 2].plot(stellar_object.xdata_670, np.mean(stellar_object.ydata_real[q_slice]) * np.ones_like(stellar_object.xdata_670),  label='Real $\mu$')
                    im4 = ax[1, 2].scatter(stellar_object.xdata_670, stellar_object.y_model[q_slice], c=abs(stellar_object.zdata_670),
                                           cmap='jet', marker='o', alpha=0.1, label='Model', vmin=np.min(stellar_object.zdata_670),
                                           vmax=np.max(stellar_object.zdata_670),  s = 5)
                    ax[1, 2].plot(stellar_object.xdata_670,
                                  np.mean(stellar_object.y_model[q_slice]) * np.ones_like(stellar_object.xdata_670), alpha=0.2,
                                  label='Model $\mu$')

                    chisqu_t = uf.chi_squ_red(stellar_object.ydata_real[q_slice], stellar_object.y_model[q_slice], stellar_object.ydata_real_err[q_slice], len(model_params['changing_by_theta']))
                    ax[1, 2].set_title('Modelled and Real Stokes Q $V^2$, $\chi_{{red}}^2$ {:.2f}'.format(chisqu_t))
                    cbar = fig.colorbar(im3, ax=ax[1, 2], label="Baseline Length (m)")
                    cbar.set_label('Baseline Length (m)')
                    ax[1, 2].set_ylabel('Differential $V^2$')
                    ax[1, 2].set_xlabel('Azimuth Angle (rad)')

                    # Stokes U plot for segment 2
                    im4 = ax[1, 3].scatter(stellar_object.xdata_670, stellar_object.ydata_real[u_slice], c=abs(stellar_object.zdata_670),
                                           cmap='jet', marker='x', label='Real',  s = 10,  linewidths=0.5)
                    ax[1, 3].errorbar(stellar_object.xdata_670, stellar_object.ydata_real[u_slice],
                                      yerr=stellar_object.ydata_real_err[u_slice], fmt='none', ecolor='lightgrey', alpha = 0.4, elinewidth=0.5)
                    ax[1, 3].plot(stellar_object.xdata_670,
                                  np.mean(stellar_object.ydata_real[u_slice]) * np.ones_like(stellar_object.xdata_670),
                                  label='Real $\mu$')
                    im4 = ax[1, 3].scatter(stellar_object.xdata_670, stellar_object.y_model[u_slice], c=abs(stellar_object.zdata_670),
                                           cmap='jet', marker='o', alpha=0.1, label='Model', vmin=np.min(stellar_object.zdata_670),
                                           vmax=np.max(stellar_object.zdata_670),  s = 5)
                    ax[1, 3].plot(stellar_object.xdata_670,
                                  np.mean(stellar_object.y_model[u_slice]) * np.ones_like(stellar_object.xdata_670), alpha=0.2,
                                  label='Model $\mu$')
                    chisqu_t = uf.chi_squ_red(stellar_object.ydata_real[u_slice], stellar_object.y_model[u_slice],
                                              stellar_object.ydata_real_err[u_slice],
                                              len(model_params['changing_by_theta']))

                    ax[1, 3].set_title('Modelled and Real Stokes U $V^2$, $\chi_{{red}}^2$ {:.2f}'.format(chisqu_t))
                    cbar = fig.colorbar(im4, ax=ax[1, 3], label="Baseline Length (m)")
                    cbar.set_label('Baseline Length (m)')
                    ax[1, 3].set_ylabel('Differential $V^2$')
                    ax[1, 3].set_xlabel('Azimuth Angle (rad)')
                    # ax[1, 3].legend(loc='best')

                    # Correlation Q
                    im5 = ax[2,2].scatter(stellar_object.ydata_real[q_slice], stellar_object.y_model[q_slice],
                                           c=abs(stellar_object.zdata_670), cmap='jet', marker='o', label='M',
                                           vmin=np.min(stellar_object.zdata_670), vmax=np.max(stellar_object.zdata_670))
                    ax[2,2].set_title('Correlation between real and modelled $V^2$')
                    length_thing = np.linspace(np.min(stellar_object.ydata_real[q_slice]),
                                               np.max(stellar_object.ydata_real[q_slice]), 100)
                    ax[2,2].scatter(length_thing, length_thing, marker='.', c='k', label='x = y')
                    cbar = fig.colorbar(im5, ax=ax[2,2], label="Baseline Length (m)")
                    ax[2,2].set_ylabel('Modelled Differential $V^2$')
                    ax[2,2].set_xlabel('Real Differential $V^2$')

                    # Correlation U
                    im6 = ax[2,3].scatter(stellar_object.ydata_real[u_slice], stellar_object.y_model[u_slice],
                                           c=abs(stellar_object.zdata_670), cmap='jet', marker='o', label='M',
                                           vmin=np.min(stellar_object.zdata_670), vmax=np.max(stellar_object.zdata_670))
                    ax[2,3].set_title('Correlation between real and modelled $V^2$')
                    length_thing = np.linspace(np.min(stellar_object.ydata_real[u_slice]),
                                               np.max(stellar_object.ydata_real[u_slice]), 100)
                    ax[2,3].scatter(length_thing, length_thing, marker='.', c='k', label='x = y')
                    ax[2,3].set_ylabel('Modelled Differential $V^2$')
                    ax[2,3].set_xlabel('Real Differential $V^2$')
                    # ax[2,3].legend(loc='best')
                    cbar = fig.colorbar(im6, ax=ax[2,3], label="Baseline Length (m)")



                    im5 = ax[3,2].scatter(stellar_object.ydata_real[2244:3060], stellar_object.y_model[2244:3060],  marker='o',  label='M' )
                    ax[3,2].set_title('Correlation between real and modelled $CP$')
                    length_thing = np.arange(np.min(stellar_object.ydata_real[2244:3060]), np.max(stellar_object.ydata_real[2244:3060]), 0.0001)
                    ax[3,2].scatter(length_thing, length_thing, marker =  '.', c = 'k',  label = 'x = y')
                    ax[3,2].set_ylabel('Modelled Differential CP')
                    ax[3,2].set_xlabel('Real Differential CP')


                    im6 = ax[3,3].scatter(stellar_object.ydata_real[3060:3876], stellar_object.y_model[3060:3876],  marker='o', label='M' )
                    ax[3,3].set_title('Correlation between real and modelled CP')
                    length_thing = np.arange(np.min(stellar_object.ydata_real[3060:3876]), np.max(stellar_object.ydata_real[3060:3876]), 0.0001)
                    ax[3,3].scatter(length_thing, length_thing, marker = '.', c = 'k', label = 'x = y')
                    ax[3,3].set_ylabel('Modelled Differential CP')
                    ax[3,3].set_xlabel('Real Differential CP')




                    big_thing = np.max( np.array([np.abs(np.min(stellar_object.image_Q_670)), np.abs(np.max(stellar_object.image_Q_670))]))

                    im7 = ax[0, 2].imshow((stellar_object.image_Q_670), cmap='seismic',  clim=[-big_thing, big_thing])  # .sum(axis=2)))
                    ax[0, 2].set_title('Stokes Q Image 670')
                    ax[0, 2].set_xlabel('x (mas)')
                    ax[0, 2].set_ylabel('y (mas)')
                    plt.tight_layout()
                    cbar = fig.colorbar(im7, ax=ax[0, 2], label="Baseline Length (m)")
                    cbar.set_label('Normalised Polarised Flux')

                    big_thing = np.max( np.array([np.abs(np.min(stellar_object.image_U_670)), np.abs(np.max(stellar_object.image_U_670))]))
                    im77 = ax[0, 3].imshow((stellar_object.image_U_670), cmap='seismic', clim=[-big_thing, big_thing])
                    ax[0, 3].set_title('Stokes U Image 670')
                    ax[0, 3].set_xlabel('x (mas)')
                    ax[0, 3].set_ylabel('y (mas)')
                    cbar = fig.colorbar(im7, ax=ax[0, 3], label="Baseline Length (m)")
                    cbar.set_label('Normalised Polarised Flux')

                    ################################


                    start3 = 153 * 4 + 816 * 4
                    q_slice3 = slice(start3, start3 + 153)
                    u_slice3 = slice(start3 + 153, start3 + 153 * 2)


                    # Stokes Q Segment 3
                    im3 = ax[1,4].scatter(stellar_object.xdata_720, stellar_object.ydata_real[q_slice3], c=abs(stellar_object.zdata_720),
                                           cmap='jet', marker='x', label='Real',  s = 10, linewidths=0.5)
                    ax[1,4].errorbar(stellar_object.xdata_720, stellar_object.ydata_real[q_slice3],
                                      yerr=stellar_object.ydata_real_err[q_slice3], fmt='none', ecolor='lightgrey', alpha = 0.4, elinewidth=0.5)
                    ax[1,4].plot(stellar_object.xdata_720,
                                  np.mean(stellar_object.ydata_real[q_slice3]) * np.ones_like(stellar_object.xdata_720),
                                  label='Real $\mu$')
                    im4 = ax[1,4].scatter(stellar_object.xdata_720, stellar_object.y_model[q_slice3], c=abs(stellar_object.zdata_720),
                                           cmap='jet', marker='o', alpha=0.1, label='Model', vmin=np.min(stellar_object.zdata_720),
                                           vmax=np.max(stellar_object.zdata_720),  s = 5)
                    ax[1,4].plot(stellar_object.xdata_720,np.mean(stellar_object.y_model[q_slice3]) * np.ones_like(stellar_object.xdata_720), alpha=0.2,
                                  label='Model $\mu$')
                    ax[1,4].set_title('Modelled and Real Stokes Q $V^2$, Segment 3')
                    cbar = fig.colorbar(im3, ax=ax[1,4], label="Baseline Length (m)")
                    cbar.set_label('Baseline Length (m)')
                    ax[1,4].set_ylabel('Differential $V^2$')
                    ax[1,4].set_xlabel('Azimuth Angle (rad)')

                    # Stokes U Segment 3

                    im4 = ax[1, 5].scatter(stellar_object.xdata_720, stellar_object.ydata_real[u_slice3], c=abs(stellar_object.zdata_720),
                                           cmap='jet', marker='x', label='Real', s = 10, linewidths=0.5)
                    ax[1, 5].errorbar(stellar_object.xdata_720, stellar_object.ydata_real[u_slice3],
                                      yerr=stellar_object.ydata_real_err[u_slice3], fmt='none', ecolor='lightgrey', alpha = 0.4, elinewidth=0.5)
                    ax[1, 5].plot(stellar_object.xdata_720,
                                  np.mean(stellar_object.ydata_real[u_slice3]) * np.ones_like(stellar_object.xdata_720),
                                  label='Real $\mu$')
                    im4 = ax[1, 5].scatter(stellar_object.xdata_720, stellar_object.y_model[u_slice3], c=abs(stellar_object.zdata_720),
                                           cmap='jet', marker='o', alpha=0.1, label='Model', vmin=np.min(stellar_object.zdata_720),
                                           vmax=np.max(stellar_object.zdata_720),  s = 10)
                    ax[1, 5].plot(stellar_object.xdata_720,
                                  np.mean(stellar_object.y_model[u_slice3]) * np.ones_like(stellar_object.xdata_720), alpha=0.2,
                                  label='Model $\mu$')
                    ax[1, 5].set_title('Modelled and Real Stokes U $V^2$, Segment 3')
                    cbar = fig.colorbar(im4, ax=ax[1, 5], label="Baseline Length (m)")
                    cbar.set_label('Baseline Length (m)')
                    ax[1, 5].set_ylabel('Differential $V^2$')
                    ax[1, 5].set_xlabel('Azimuth Angle (rad)')
                    # ax[1, 5].legend(loc='best')


                    im5 = ax[3,4].scatter(stellar_object.ydata_real[4182:4998], stellar_object.y_model[4182:4998],  marker='o',  label='M' )
                    ax[3,4].set_title('Correlation between real and modelled $CP$')
                    length_thing = np.arange(np.min(stellar_object.ydata_real[4182:4998]), np.max(stellar_object.ydata_real[4182:4998]), 0.0001)
                    ax[3,4].scatter(length_thing, length_thing, marker =  '.', c = 'k',  label = 'x = y')
                    ax[3,4].set_ylabel('Modelled Differential CP')
                    ax[3,4].set_xlabel('Real Differential CP')


                    im6 = ax[3,5].scatter(stellar_object.ydata_real[4998:5814], stellar_object.y_model[4998:5814],  marker='o', label='M' )
                    ax[3,5].set_title('Correlation between real and modelled CP')
                    length_thing = np.arange(np.min(stellar_object.ydata_real[4998:5814]), np.max(stellar_object.ydata_real[4998:5814]), 0.0001)
                    ax[3,5].scatter(length_thing, length_thing, marker = '.', c = 'k', label = 'x = y')
                    ax[3,5].set_ylabel('Modelled Differential CP')
                    ax[3,5].set_xlabel('Real Differential CP')




                    # Correlation Segment 3 Q
                    im5 = ax[2,4].scatter(stellar_object.ydata_real[q_slice3], stellar_object.y_model[q_slice3],
                                           c=abs(stellar_object.zdata_720), cmap='jet', marker='o', label='M',
                                           vmin=np.min(stellar_object.zdata_720), vmax=np.max(stellar_object.zdata_720))
                    ax[2,4].set_title('Correlation Q, Segment 3')
                    line = np.linspace(np.min(stellar_object.ydata_real[q_slice3]), np.max(stellar_object.ydata_real[q_slice3]),
                                       100)
                    ax[2,4].scatter(line, line, marker='.', c='k', label='x = y')
                    cbar = fig.colorbar(im5, ax=ax[2,4], label="Baseline Length (m)")
                    ax[2,4].set_ylabel('Modelled Differential $V^2$')
                    ax[2,4].set_xlabel('Real Differential $V^2$')

                    # Correlation Segment 3 U
                    im6 = ax[2,5].scatter(stellar_object.ydata_real[u_slice3], stellar_object.y_model[u_slice3],
                                           c=abs(stellar_object.zdata_720), cmap='jet', marker='o', label='M',
                                           vmin=np.min(stellar_object.zdata_720), vmax=np.max(stellar_object.zdata_720))
                    ax[2,5].set_title('Correlation U, Segment 3')
                    line = np.linspace(np.min(stellar_object.ydata_real[u_slice3]), np.max(stellar_object.ydata_real[u_slice3]),
                                       100)
                    ax[2,5].scatter(line, line, marker='.', c='k', label='x = y')
                    ax[2,5].set_ylabel('Modelled Differential $V^2$')
                    ax[2,5].set_xlabel('Real Differential $V^2$')
                    # ax[2,5].legend(loc='best')
                    cbar = fig.colorbar(im6, ax=ax[2,5], label="Baseline Length (m)")





                    big_thing = np.max( np.array([np.abs(np.min(stellar_object.image_Q_720)), np.abs(np.max(stellar_object.image_Q_720))]))

                    im7 = ax[0, 4].imshow((stellar_object.image_Q_720), cmap='seismic',  clim=[-big_thing, big_thing])  # .sum(axis=2)))
                    ax[0, 4].set_title('Stokes Q Image 720')
                    ax[0, 4].set_xlabel('x (mas)')
                    ax[0, 4].set_ylabel('y (mas)')
                    plt.tight_layout()
                    cbar = fig.colorbar(im7, ax=ax[0, 4], label="Baseline Length (m)")
                    cbar.set_label('Normalised Polarised Flux')

                    big_thing = np.max( np.array([np.abs(np.min(stellar_object.image_U_720)), np.abs(np.max(stellar_object.image_U_720))]))
                    im77 = ax[0, 5].imshow((stellar_object.image_U_720), cmap='seismic', clim=[-big_thing, big_thing])
                    ax[0, 5].set_title('Stokes U Image 720')
                    ax[0, 5].set_xlabel('x (mas)')
                    ax[0, 5].set_ylabel('y (mas)')
                    cbar = fig.colorbar(im7, ax=ax[0, 5], label="Baseline Length (m)")
                    cbar.set_label('Normalised Polarised Flux')

                    ################################



                    start4 = 153 * 6 + 816 * 6
                    q_slice4 = slice(start4, start4 + 153)
                    u_slice4 = slice(start4 + 153, start4 + 153 * 2)


                    ##################################################################################



                    # Stokes Q Segment 4_760
                    im3 = ax[1,6].scatter(stellar_object.xdata_760, stellar_object.ydata_real[q_slice4], c=abs(stellar_object.zdata_760),
                                           cmap='jet', marker='x', label='Real',  s = 10, linewidths=0.5)
                    ax[1,6].errorbar(stellar_object.xdata_760, stellar_object.ydata_real[q_slice4],
                                      yerr=stellar_object.ydata_real_err[q_slice4], fmt='none', ecolor='lightgrey', alpha = 0.4, elinewidth=0.5)
                    ax[1, 6].plot(stellar_object.xdata_760,
                                  np.mean(stellar_object.ydata_real[q_slice4]) * np.ones_like(stellar_object.xdata_760),
                                  label='Real $\mu$')
                    im4 = ax[1, 6].scatter(stellar_object.xdata_760, stellar_object.y_model[q_slice4], c=abs(stellar_object.zdata_760),
                                           cmap='jet', marker='o', alpha=0.1, label='Model', vmin=np.min(stellar_object.zdata_760),
                                           vmax=np.max(stellar_object.zdata_760),  s = 5)
                    ax[1, 6].plot(stellar_object.xdata_760,
                                  np.mean(stellar_object.y_model[q_slice4]) * np.ones_like(stellar_object.xdata_760), alpha=0.2,
                                  label='Model $\mu$')
                    ax[1, 6].set_title('Modelled and Real Stokes Q $V^2$, Segment 4')
                    cbar = fig.colorbar(im3, ax=ax[1, 6], label="Baseline Length (m)")
                    cbar.set_label('Baseline Length (m)')
                    ax[1, 6].set_ylabel('Differential $V^2$')
                    ax[1, 6].set_xlabel('Azimuth Angle (rad)')

                    # Stokes U Segment 4
                    im4 = ax[1, 7].scatter(stellar_object.xdata_760, stellar_object.ydata_real[u_slice4], c=abs(stellar_object.zdata_760),
                                           cmap='jet', marker='x', label='Real', s = 10, linewidths=0.5)
                    ax[1, 7].errorbar(stellar_object.xdata_760, stellar_object.ydata_real[u_slice4],
                                      yerr=stellar_object.ydata_real_err[u_slice4], fmt='none', ecolor='lightgrey', alpha = 0.4, elinewidth=0.5)
                    ax[1, 7].plot(stellar_object.xdata_760,
                                  np.mean(stellar_object.ydata_real[u_slice4]) * np.ones_like(stellar_object.xdata_760),
                                  label='Real $\mu$')
                    im4 = ax[1, 7].scatter(stellar_object.xdata_760, stellar_object.y_model[u_slice4], c=abs(stellar_object.zdata_760),
                                           cmap='jet', marker='o', alpha=0.1, label='Model', vmin=np.min(stellar_object.zdata_760),
                                           vmax=np.max(stellar_object.zdata_760),  s = 5)
                    ax[1, 7].plot(stellar_object.xdata_760,
                                  np.mean(stellar_object.y_model[u_slice4]) * np.ones_like(stellar_object.xdata_760), alpha=0.2,
                                  label='Model $\mu$')
                    ax[1, 7].set_title('Modelled and Real Stokes U $V^2$, Segment 4')
                    cbar = fig.colorbar(im4, ax=ax[1, 7], label="Baseline Length (m)")
                    cbar.set_label('Baseline Length (m)')
                    ax[1, 7].set_ylabel('Differential $V^2$')
                    ax[1, 7].set_xlabel('Azimuth Angle (rad)')
                    # ax[1, 7].legend(loc='best')

                    # Correlation Segment 4 Q
                    im5 = ax[2,6].scatter(stellar_object.ydata_real[q_slice4], stellar_object.y_model[q_slice4],
                                           c=abs(stellar_object.zdata_760), cmap='jet', marker='o', label='M',
                                           vmin=np.min(stellar_object.zdata_760), vmax=np.max(stellar_object.zdata_760))
                    ax[2,6].set_title('Correlation Q, Segment 4')
                    line = np.linspace(np.min(stellar_object.ydata_real[q_slice4]), np.max(stellar_object.ydata_real[q_slice4]),
                                       100)
                    ax[2,6].scatter(line, line, marker='.', c='k', label='x = y')
                    cbar = fig.colorbar(im5, ax=ax[2,6], label="Baseline Length (m)")
                    ax[2,6].set_ylabel('Modelled Differential $V^2$')
                    ax[2,6].set_xlabel('Real Differential $V^2$')

                    # Correlation Segment 4 U
                    im6 = ax[2,7].scatter(stellar_object.ydata_real[u_slice4], stellar_object.y_model[u_slice4],
                                           c=abs(stellar_object.zdata_760), cmap='jet', marker='o', label='M',
                                           vmin=np.min(stellar_object.zdata_760), vmax=np.max(stellar_object.zdata_760))
                    ax[2,7].set_title('Correlation U, Segment 4')
                    line = np.linspace(np.min(stellar_object.ydata_real[u_slice4]), np.max(stellar_object.ydata_real[u_slice4]),
                                       100)
                    ax[2,7].scatter(line, line, marker='.', c='k', label='x = y')
                    ax[2,7].set_ylabel('Modelled Differential $V^2$')
                    ax[2,7].set_xlabel('Real Differential $V^2$')
                    # ax[2,7].legend(loc='best')
                    cbar = fig.colorbar(im6, ax=ax[2,7], label="Baseline Length (m)")




                    im5 = ax[3,6].scatter(stellar_object.ydata_real[6120:6936], stellar_object.y_model[6120:6936],  marker='o',  label='M' )
                    ax[3,6].set_title('Correlation between real and modelled $CP$')
                    length_thing = np.arange(np.min(stellar_object.ydata_real[6120:6936]), np.max(stellar_object.ydata_real[6120:6936]), 0.0001)
                    ax[3,6].scatter(length_thing, length_thing, marker =  '.', c = 'k',  label = 'x = y')
                    ax[3,6].set_ylabel('Modelled Differential CP')
                    ax[3,6].set_xlabel('Real Differential CP')


                    im6 = ax[3,7].scatter(stellar_object.ydata_real[6936:7752], stellar_object.y_model[6936:7752],  marker='o', label='M' )
                    ax[3,7].set_title('Correlation between real and modelled CP')
                    length_thing = np.arange(np.min(stellar_object.ydata_real[6936:7752]), np.max(stellar_object.ydata_real[6936:7752]), 0.0001)
                    ax[3,7].scatter(length_thing, length_thing, marker = '.', c = 'k', label = 'x = y')
                    ax[3,7].set_ylabel('Modelled Differential CP')
                    ax[3,7].set_xlabel('Real Differential CP')




                    big_thing = np.max( np.array([np.abs(np.min(stellar_object.image_Q_760)), np.abs(np.max(stellar_object.image_Q_760))]))

                    im7 = ax[0, 6].imshow((stellar_object.image_Q_760), cmap='seismic',  clim=[-big_thing, big_thing])  # .sum(axis=2)))
                    ax[0, 6].set_title('Stokes Q Image 760')
                    ax[0, 6].set_xlabel('x (mas)')
                    ax[0, 6].set_ylabel('y (mas)')
                    plt.tight_layout()
                    cbar = fig.colorbar(im7, ax=ax[0, 6], label="Baseline Length (m)")
                    cbar.set_label('Normalised Polarised Flux')

                    big_thing = np.max( np.array([np.abs(np.min(stellar_object.image_U_760)), np.abs(np.max(stellar_object.image_U_760))]))
                    im77 = ax[0, 7].imshow((stellar_object.image_U_760), cmap='seismic', clim=[-big_thing, big_thing])
                    ax[0, 7].set_title('Stokes U Image 760')
                    ax[0, 7].set_xlabel('x (mas)')
                    ax[0, 7].set_ylabel('y (mas)')
                    cbar = fig.colorbar(im7, ax=ax[0, 7], label="Baseline Length (m)")
                    cbar.set_label('Normalised Polarised Flux')

                    ################################







                    plt.tight_layout()
                    plt.savefig(savedir + 'model_plots_{}_{}.pdf'.format(model, tag))
                    plt.close()






                pol_P = np.sqrt(stellar_object.image_U_760 ** 2 + stellar_object.image_Q_760 ** 2)
                phi_P = np.arctan2(stellar_object.image_U_760, stellar_object.image_Q_760) / 2
                xE = np.arange(0, np.shape(phi_P)[0], 1)
                yE = np.arange(0, np.shape(phi_P)[0], 1)
                X, Y = np.meshgrid(xE, yE)

                if plot:
                    plt.figure()
                    plt.quiver(X, Y, pol_P / pol_P.max(), pol_P / pol_P.max(), angles=np.rad2deg(phi_P + np.pi / 2), pivot='mid',
                               headwidth=0, headlength=0, scale=25)
                    plt.title('Polarization Vectors')
                    plt.xlabel('mas')
                    plt.ylabel('mas')
                    plt.savefig(savedir + 'polarisation_map_{}_{}.pdf'.format(model, tag))

                Q_vals_real = onp.concatenate((stellar_object.ydata_real[0:153], stellar_object.ydata_real[153 * 2:153 * 2 + 816]),  axis=0)
                U_vals_real = onp.concatenate( (stellar_object.ydata_real[153:153 * 2], stellar_object.ydata_real[153 * 2 + 816:153 * 2 + 816 * 2]), axis=0)

                Q_vals_model = onp.concatenate((stellar_object.y_model[0:153], stellar_object.y_model[153 * 2:153 * 2 + 816]), axis=0)
                U_vals_model = onp.concatenate(  (stellar_object.y_model[153:153 * 2], stellar_object.y_model[153 * 2 + 816:153 * 2 + 816 * 2]), axis=0)

                Q_vals_error = onp.concatenate(  (stellar_object.ydata_real_err[0:153], stellar_object.ydata_real_err[153 * 2:153 * 2 + 816]), axis=0)
                U_vals_error = onp.concatenate( (stellar_object.ydata_real_err[153:153 * 2], stellar_object.ydata_real_err[153 * 2 + 816:153 * 2 + 816 * 2]),  axis=0)



                bl_things = zdata_610[indx_of_cp]
                bl_cp = np.max(bl_things, axis = 0)
 
                chisqu_reducedQ = uf.chi_squ_red(stellar_object.ydata_real[-(153*2 + 816*2):],
                                                 stellar_object.y_model[-(153*2 + 816*2):],
                                                 stellar_object.ydata_real_err[-(153*2 + 816*2):],
                                                 len(model_params['changing_by_theta']))


                chisqu_reducedU = uf.chi_squ_red(stellar_object.ydata_real[-(153*2 + 816*2):],
                                                 stellar_object.y_model[-(153*2 + 816*2):],
                                                 stellar_object.ydata_real_err[-(153*2 + 816*2):],
                                                 len(model_params['changing_by_theta']))

               
                def radial_average(image, center=None):
                    """Computes the radial average of an image."""
                    y, x = np.indices(image.shape)

                    if center is None:
                        center = (image.shape[1] // 2, image.shape[0] // 2)  # (x_center, y_center)

                    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)  # Compute radial distances
                    r = r.astype(np.int32)  # Convert to integers for binning

                    radial_sum = np.bincount(r.ravel(), weights=image.ravel())  # Sum intensities per radius
                    radial_count = np.bincount(r.ravel())  # Count pixels per radius
                    radial_mean = radial_sum / np.maximum(radial_count, 1)  # Avoid division by zero

                    return radial_mean


                QonI = stellar_object.image_Q_760/stellar_object.image_I_760
                PolP_ = np.sqrt(stellar_object.image_Q_760**2 + stellar_object.image_U_760**2)/(stellar_object.image_I_760 + 1e-36)
                pol_P = np.sqrt(stellar_object.image_U_760 ** 2 + stellar_object.image_Q_760 ** 2)

                if plot:
                    plt.figure()
                    plt.imshow(PolP_)
                    plt.title('Pol P {:.4f}, fracP {:.4f}, phi P {:.4f}'.format(np.sum(pol_P), np.sum(PolP_), np.mean(phi_P)))
                    plt.colorbar()
                    plt.savefig(savedir + 'POLP_{}_{}.pdf'.format(model, tag))




                radial_profile = radial_average(np.abs(QonI * 100))
                if plot:
                    plt.figure(figsize = (14,4))
                    plt.subplot(1,3,1)
                    plt.imshow(QonI*100)
                    plt.colorbar()
                    plt.xlabel('x (mas)')
                    plt.ylabel('y (mas)')
                    plt.title('Q/I % (full FOV)')
                    plt.subplot(1,3,2)
                    imsize = np.shape(QonI)[0]
                    middle = int(imsize/2)
                    plt.imshow(QonI[middle-20:middle+20, middle-20:middle+20]*100)
                    plt.colorbar()
                    plt.xlabel('x (mas)')
                    plt.ylabel('y (mas)')
                    plt.title('Q/I % (reduced FOV)')

                    plt.subplot(1,3,3)
                    plt.plot(radial_profile[0:50])
                    plt.xlabel("Radius (pixels, mas)")
                    plt.ylabel("Average abs(Q/I) %")
                    plt.title("Radial Q/I Profile")
                    plt.tight_layout()
                    plt.savefig(savedir + 'QonI_{}_{}.pdf'.format(model, tag))

                if plot:
                    plt.figure(figsize=(16, 16))

                    sto_I = scat_H_610 + scat_V_610
                    Q = scat_H_610 - scat_V_610
                    U = scat_H45_610 - scat_V45_610
                    I = scat_H_610 + scat_V_610

                    P = np.sqrt(Q**2 + U**2)/(I + 1e-36)

                    dust_shell_final = stellar_object.shell#*P
                    dust_shell_final = dust_shell_final/dust_shell_final.sum()

                    plt.subplot(4, 3, 1)
                    plt.imshow(rotate(dust_shell_final.sum(axis=2), angle=PA, reshape=False))
                    plt.title('Rotated, BIC {:.2f}'.format(BIC))
                    plt.colorbar()

                    plt.subplot(4, 3, 2)

                    dust_shell_final = stellar_object.shell*P
                    dust_shell_final = dust_shell_final/dust_shell_final.sum()
                    plt.imshow(rotate(dust_shell_final.sum(axis=2), angle=PA, reshape=False))
                    # plt.imshow(dust_shell_final.sum(axis=2))#, angle=PA, reshape=False))
                    plt.title('Rotated P, chisq {:.2f}'.format(chisqu_reduced))
                    plt.colorbar()

                    plt.subplot(4, 3, 3)

                    Irot = np.expand_dims(rotate(stellar_object.image_I_610, angle=PA, reshape=False), axis=0)
                    Qrot = np.expand_dims(rotate(stellar_object.image_Q_610, angle=PA, reshape=False), axis=0)
                    Urot = np.expand_dims(rotate(stellar_object.image_U_610, angle=PA, reshape=False), axis=0)
                    Vrot = np.zeros(np.shape(Urot))

                    input_stokes = np.concatenate((Irot, Qrot, Urot, Vrot), axis=0)

                    mmrotQ = uf.comp_higher_matrix_mult(rotator(-PA), input_stokes)[1,]
                    tep = np.max(np.array([np.min(mmrotQ), np.max(mmrotQ)]))
                    plt.imshow(mmrotQ, cmap='seismic', clim=[-tep*0.1, tep*0.1])
                    # plt.title('Log Prob is {:.10f}'.format( logZ ))
                    plt.colorbar()


                    ################

                    sto_I = scat_H_670 + scat_V_670
                    Q = scat_H_670 - scat_V_670
                    U = scat_H45_670 - scat_V45_670
                    I = scat_H_670 + scat_V_670

                    P = np.sqrt(Q**2 + U**2)/(I + 1e-36)

                    dust_shell_final = stellar_object.shell#*P
                    dust_shell_final = dust_shell_final/dust_shell_final.sum()



                    plt.subplot(4, 3, 4)
                    plt.imshow(rotate(dust_shell_final.sum(axis=2), angle=PA, reshape=False))
                    # plt.imshow(dust_shell_final.sum(axis=2))
                    plt.title('Rotated, BIC {:.2f}'.format(BIC))
                    plt.colorbar()

                    plt.subplot(4, 3, 5)

                    dust_shell_final = stellar_object.shell*P
                    dust_shell_final = dust_shell_final/dust_shell_final.sum()
                    plt.imshow(rotate(dust_shell_final.sum(axis=2), angle=PA, reshape=False))
                    # plt.imshow(dust_shell_final.sum(axis=2))#, angle=PA, reshape=False))
                    plt.title('Rotated P, {:.2f}'.format(chisqu_reduced))
                    plt.colorbar()

                    plt.subplot(4, 3, 6)

                    Irot = np.expand_dims(rotate(stellar_object.image_I_670, angle=PA, reshape=False), axis=0)
                    Qrot = np.expand_dims(rotate(stellar_object.image_Q_670, angle=PA, reshape=False), axis=0)
                    Urot = np.expand_dims(rotate(stellar_object.image_U_670, angle=PA, reshape=False), axis=0)
                    Vrot = np.zeros(np.shape(Urot))

                    input_stokes = np.concatenate((Irot, Qrot, Urot, Vrot), axis=0)

                    mmrotQ = uf.comp_higher_matrix_mult(rotator(-PA), input_stokes)[1,]
                    tep = np.max(np.array([np.min(mmrotQ), np.max(mmrotQ)]))
                    plt.imshow(mmrotQ, cmap='seismic', clim=[-tep*0.1, tep*0.1])
                    # plt.title('Log Prob is {:.10f}'.format( logZ ))
                    plt.colorbar()

                    #####################

                    sto_I = scat_H_720 + scat_V_720
                    Q = scat_H_720 - scat_V_720
                    U = scat_H45_720 - scat_V45_720
                    I = scat_H_720 + scat_V_720

                    P = np.sqrt(Q**2 + U**2)/(I + 1e-36)

                    dust_shell_final = stellar_object.shell#*P
                    dust_shell_final = dust_shell_final/dust_shell_final.sum()


                    plt.subplot(4, 3, 7)
                    plt.imshow(rotate(dust_shell_final.sum(axis=2), angle=PA, reshape=False))
                    # plt.imshow(dust_shell_final.sum(axis=2))
                    plt.title('Rotated, BIC {:.2f}'.format(BIC))
                    plt.colorbar()

                    plt.subplot(4, 3, 8)

                    dust_shell_final = stellar_object.shell*P
                    dust_shell_final = dust_shell_final/dust_shell_final.sum()
                    plt.imshow(rotate(dust_shell_final.sum(axis=2), angle=PA, reshape=False))
                    # plt.imshow(dust_shell_final.sum(axis=2))#, angle=PA, reshape=False))
                    plt.title('Rotated P, {:.2f}'.format(chisqu_reduced))
                    plt.colorbar()

                    plt.subplot(4, 3, 9)

                    Irot = np.expand_dims(rotate(stellar_object.image_I_720, angle=PA, reshape=False), axis=0)
                    Qrot = np.expand_dims(rotate(stellar_object.image_Q_720, angle=PA, reshape=False), axis=0)
                    Urot = np.expand_dims(rotate(stellar_object.image_U_720, angle=PA, reshape=False), axis=0)
                    Vrot = np.zeros(np.shape(Urot))

                    input_stokes = np.concatenate((Irot, Qrot, Urot, Vrot), axis=0)

                    mmrotQ = uf.comp_higher_matrix_mult(rotator(-PA), input_stokes)[1,]
                    tep = np.max(np.array([np.min(mmrotQ), np.max(mmrotQ)]))
                    plt.imshow(mmrotQ, cmap='seismic', clim=[-tep*0.1, tep*0.1])
                    # plt.title('Log Prob is {:.10f}'.format( logZ ))
                    plt.colorbar()


                    #######################

                    sto_I = scat_H_760 + scat_V_760
                    Q = scat_H_760 - scat_V_760
                    U = scat_H45_760 - scat_V45_760
                    I = scat_H_760 + scat_V_760

                    P = np.sqrt(Q**2 + U**2)/(I + 1e-36)

                    dust_shell_final = stellar_object.shell#*P
                    dust_shell_final = dust_shell_final/dust_shell_final.sum()

                    plt.subplot(4, 3, 10)
                    plt.imshow(rotate(dust_shell_final.sum(axis=2), angle=PA, reshape=False))
                    # plt.imshow(dust_shell_final.sum(axis=2))
                    plt.title('Rotated, BIC {:.4f}'.format(BIC))
                    plt.colorbar()

                    plt.subplot(4, 3, 11)

                    dust_shell_final = stellar_object.shell*P
                    dust_shell_final = dust_shell_final/dust_shell_final.sum()
                    plt.imshow(rotate(dust_shell_final.sum(axis=2), angle=PA, reshape=False))
                    # plt.imshow(dust_shell_final.sum(axis=2))#, angle=PA, reshape=False))
                    plt.title('Rotated P, chisqu {:.4f}'.format(chisqu_reduced))
                    plt.colorbar()

                    plt.subplot(4, 3, 12)

                    Irot = np.expand_dims(rotate(stellar_object.image_I_760, angle=PA, reshape=False), axis=0)
                    Qrot = np.expand_dims(rotate(stellar_object.image_Q_760, angle=PA, reshape=False), axis=0)
                    Urot = np.expand_dims(rotate(stellar_object.image_U_760, angle=PA, reshape=False), axis=0)
                    Vrot = np.zeros(np.shape(Urot))

                    input_stokes = np.concatenate((Irot, Qrot, Urot, Vrot), axis=0)

                    mmrotQ = uf.comp_higher_matrix_mult(rotator(-PA), input_stokes)[1,]
                    tep = np.max(np.array([np.min(mmrotQ), np.max(mmrotQ)]))
                    plt.imshow(mmrotQ, cmap='seismic', clim=[-tep*0.1, tep*0.1])
                    # plt.title('Log Prob is {:.10f}'.format( logZ ))
                    plt.colorbar()
                    np.save(savedir + 'dust_shell_{}_{}.npy'.format(model, tag), rotate(dust_shell_final.sum(axis=2), angle=PA, reshape=False))
                    plt.savefig( savedir + 'rotatedmodel_plots_dustonly_{}_{}.pdf'.format(  model, tag))

                    plt.close('all')

                try:
                    autocorr_time = sampler.get_autocorr_time()[0]
                    # print('************************************************************************************************************')
                    print(autocorr_time)
                except Exception as e:
                    print(f"Error calculating autocorrelation time: {e}")
                    autocorr_time = None


                del scat_V45_610, scat_H45_610, scat_V_610, scat_H_610
                del scat_V45_670, scat_H45_670, scat_V_670, scat_H_670
                del scat_V45_720, scat_H45_720, scat_V_720, scat_H_720
                del scat_V45_760, scat_H45_760, scat_V_760, scat_H_760

                grain_i = grain_i + 1
            chem_i = chem_i + 1


        print('Log evidence is {}'.format(logZ))
        print('The chi squared value is  {}'.format(chisqu_reduced))
        np.save(savedir + 'chi_map_{}.npy'.format(mod), chi_map )
        np.save(savedir + 'bayes_map_{}.npy'.format(mod), bayes_map)
