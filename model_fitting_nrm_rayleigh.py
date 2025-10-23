import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_FLAGS"] = "--xla_gpu_force_compilation_parallelism=1"

import jax

print(jax.devices())

import jaxlib
import jax
print("jaxlib:", jaxlib.__version__)
print("jax:", jax.__version__)

import pickle
from JIT2make_3D_geom_stars import geometric_star
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

warnings.filterwarnings("ignore")
 

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
    stellar_object.make_dust()
    y_model_this = stellar_object.simulate_nrm()

    q_real_vis = stellar_object.ydata_real[0:153].copy()
    u_real_vis = stellar_object.ydata_real[153:153 * 2].copy()
    q_real_cp = stellar_object.ydata_real[153 * 2:153 * 2 + 816].copy()
    u_real_cp = stellar_object.ydata_real[153 * 2 + 816:153 * 2 + 816 * 2].copy()

    q_real_vis_norm = (q_real_vis - np.mean(q_real_vis)) / np.std(q_real_vis)
    u_real_vis_norm = (u_real_vis - np.mean(u_real_vis)) / np.std(u_real_vis)
    q_real_cp_norm = (q_real_cp - np.mean(q_real_cp)) / np.std(q_real_cp)
    u_real_cp_norm = (u_real_cp - np.mean(u_real_cp)) / np.std(u_real_cp) ### this is the mistake


    norm_obs = np.concatenate(( q_real_vis_norm,
                                u_real_vis_norm,
                                np.sin(np.deg2rad(q_real_cp_norm)),
                                np.cos(np.deg2rad(q_real_cp_norm)),
                                np.sin(np.deg2rad(u_real_cp_norm)),
                                np.cos(np.deg2rad(u_real_cp_norm))))

    q_real_vis_err = stellar_object.ydata_real_err[0:153]
    u_real_vis_err = stellar_object.ydata_real_err[153:153 * 2]
    q_real_cp_err = stellar_object.ydata_real_err[153 * 2:153 * 2 + 816]
    u_real_cp_err = stellar_object.ydata_real_err[153 * 2 + 816: 153 * 2 + 816 * 2]

    q_real_vis_err_norm = compute_errors(q_real_vis, q_real_vis_err)
    u_real_vis_err_norm = compute_errors(u_real_vis, u_real_vis_err)
    q_real_cp_err_norm = compute_errors(q_real_cp, q_real_cp_err)
    u_real_cp_err_norm = compute_errors(u_real_cp, u_real_cp_err)

    norm_obs_err = np.abs(np.concatenate(( q_real_vis_err_norm,
                                           u_real_vis_err_norm,
                                          np.sin(np.deg2rad(q_real_cp_err_norm)),
                                          np.cos(np.deg2rad(q_real_cp_err_norm)),
                                          np.sin(np.deg2rad(u_real_cp_err_norm)),
                                          np.cos(np.deg2rad(u_real_cp_err_norm)))))

    q_model_vis = y_model_this[0:153]
    u_model_vis = y_model_this[153:153 * 2]
    q_model_cp = y_model_this[153 * 2:153 * 2 + 816]
    u_model_cp = y_model_this[153 * 2 + 816:153 * 2 + 816 * 2]

    q_model_vis_norm = (q_model_vis - np.mean(q_real_vis)) / np.std(q_real_vis)
    u_model_vis_norm = (u_model_vis - np.mean(u_real_vis)) / np.std(u_real_vis)
    q_model_cp_norm = (q_model_cp - np.mean(q_real_cp)) / np.std(q_real_cp)
    u_model_cp_norm = (u_model_cp - np.mean(u_real_cp)) / np.std(u_real_cp) ### this is the mistake


    norm_model = np.concatenate((q_model_vis_norm,
                                 u_model_vis_norm,
                                 np.sin(np.deg2rad(q_model_cp_norm)),
                                 np.cos(np.deg2rad(q_model_cp_norm)),
                                 np.sin(np.deg2rad(u_model_cp_norm)),
                                 np.cos(np.deg2rad(u_model_cp_norm))))




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

    check_prior = prior(theta)

    if not np.isfinite(check_prior):
        return -np.inf


    lnL_model = make_star(model_params) # this is nan


    if np.isnan(check_prior + lnL_model): #math.isnan(check_prior + lnL_model):
        for i in range(len(model_params['changing_by_theta'])):
            vari = model_params['changing_by_theta'][i]
            # print(model_params[vari])
            # print('nans')

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
          'Model_B1',
          'Model_C',
          'Model_G',
          'Model_H',
          'Model_I',


          'Model_thincircblob',
          "Model_B1",
          'Model_H1',
          'Model_I1',
          'Model_plellipse',

          
          'Model_shellthickcircle',
         'Model_shellthickcircle1',
          'Model_shellthickellipse',#_21',
          'Model_twothickcircles',
          "Model_thinshellplshell",

          "Model_threethinshells",
          "Model_onethin_onethick",
          'Model_pl_1',
          "Model_Dt2",
          "Model_shellthickcircle_1",

          "Model_twothincircles",
          "Model_thinellipse_1",
          "Model_plellipse_2",
          "Model_twothinellipses",
          "Model_F", # model F


          'Model_ellipse_twoblobs',
          'Model_ellipse_twoblobs_prior',
          'Model_Ecirc',
            'Model_E',
          'Model_bright_spot_thin_circle',  #### model P
          'Model_offset_thin_ellipse',

          'Model_offset_thick_elliipse_blob',
          'Model_thinellipseblob',
          'Model_J',
          'Model_O3_offcirc',
          'Model_Q_circ',
          'Model_enhancecircle']

p0s = [[15, 4], # A# Model A thin spherical shell
       [15, 2,  4], # Model B power law spherical shell
       [15, 15, 4, 4], # Model C two thin shells
       [15, 15,45, 4, np.pi*3/2 , 80, 0.5 ], # Model G - Ellipitcal shell with blob
       [15, 15, 25, 3, 4, np.pi*3/2 , 80, 0.5 ], # Model H #15,45,
       [15,   4, np.pi * 3 / 2, 80, 0.5],  # Model I

       [15, 4, np.pi * 3 / 2, 80, 0.5],  # Model J
       [12, 4, 3],  # Model B.1
       [15, 15,45, 4, np.pi*3/2 , 120, 0.5, 2 ],     # Model H1
       [15, 2, 4, np.pi * 3 / 2, 120, 0.5],           # Model I1
       [12, 4, 12,  45,  2],


       [12, 4, 3],
       [12, 4, 3],
       [12, 12, 25, 4, 3],
       [12, 1, 16, 1, 4, 5],
       [12,  12, 4, 1.5],

       [12,  12, 12, 4, 1.5, 1.5],
       [12,  12, 2, 4, 1.5],
       [12, 4, 3],
       [15, 15,  34,  3],                            # Model D
       [12, 4, 3],

       [12, 12, 4, 4],
       [12, 12,  12, 45,  3],
       [12, 4,  12,  0, 2],
       [12, 12, 12, 12, 0, 4, 4],
       [15,   4, np.pi*3/2 , 120, 0.5],              # model F

       [12, 12, 0,  4, np.pi*3/2 , 80, 0.5, 4.7 , 120, 0.5],
       [15, 15,  0, 4, np.pi * 3 / 2, 120, 0.5, 4.7, 120, 0.5],
       [15,  4, 0, 0 ],
       [15, 15, 25, 4, 0, 0],
       # Model E
        #['a', 'dust_pl_exp', 'dust_star_contrast',  'bright_mag' , 'bright_location_phi']
       [12,  4, 4, np.pi], #### model P np.pi/2,10,



       [12, 12, 0, 4, 0, 0 ] ,
       [12, 12, 0, 4, np.pi * 3 / 2, 120, 0.5, 0, 0, 2],
       [15, 15, 0, 4, np.pi * 3 / 2, 80, 0.5],
       [15, 15, 0,4, np.pi * 3 / 2, 80, 0.5],  # Model I
       [15,  4, np.pi*3/2 , 80, 0.5 , 0, 0],


       [15,  4, np.pi*3/2, 80, 0.5,  360, 2] ,
       [15, 4,   360, 2]
       ]


changing_by_thetas =  [['a',  'dust_star_contrast'], # Model A thin spherical shell
                       ['a', 'dust_pl_exp',  'dust_star_contrast'], # Model B power law spherical shell
                       ['a', 'a2', 'dust_star_contrast', 'ellipse_contrast_ratio'], # Model C
                       ['a', 'b', 'alpha', 'dust_star_contrast',  'phi_blob',  'blob_radial_distance', 'blob_contrast'], # Model G
                       ['a', 'b', 'alpha', 'dust_pl_exp', 'dust_star_contrast',  'phi_blob',  'blob_radial_distance', 'blob_contrast'],# Model H # 'b', 'alpha',
                       ['a',   'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast'],  # Model I

                       ['a',  'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast'],     # Model J
                       ['a',  'dust_star_contrast', 'dust_pl_exp'],    # Model B.1
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast', 'dust_pl_exp'],  # Model H1
                       ['a', 'thicka', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast'], # Model I1
                       ['a', 'dust_star_contrast', 'b',  'alpha' , 'dust_pl_exp'], #

                       ['a', 'dust_star_contrast', 'thicka'],
                        ['a', 'dust_star_contrast', 'thicka'],
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'thicka'],
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
                       ['a', 'dust_star_contrast', 'h', 'k'],  # Model E,
                        ['a', 'b', 'alpha', 'dust_star_contrast', 'h', 'k'],
                       ['a',  'dust_star_contrast',  'bright_mag' ,'bright_location_phi'],#### model P'bright_location_theta','bright_mag',
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'h', 'k'],

                       ['a', 'b', 'alpha', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast', 'h', 'k', 'thicka'],
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast'],
                       ['a', 'b', 'alpha', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast'],  # Model I
                       ['a', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast', 'h', 'k'],
                        ['a',  'dust_star_contrast',  'phi_blob',  'blob_radial_distance', 'blob_contrast',  'enhancement_loc', 'enhancement_amp'],
                       ['a', 'dust_star_contrast',  'enhancement_loc', 'enhancement_amp']
                       ]

priors =  [[[12, 200],  [0.01,  10]],                      # Model A thin spherical shell
           [[1, 100], [0.1, 10], [0.01, 10]],             # Model B power law spherical shell
           [[12, 200], [12, 200], [0.01, 10], [0.01, 50]], # Model C two thin shells
           [[12, 100], [12, 100], [0, 90], [0.01, 10],   [np.pi , 2*np.pi],    [0, 100],  [0.001, 500]], # Model G -
           [[12, 100], [12, 100], [0, 45] ,[0.05, 5], [0.01, 10],   [np.pi , 2*np.pi],    [0,100],  [0.001, 500]], # Model H # [12, 100], [-40, 45],
           [[1, 100],   [0.01, 10],   [np.pi , 2*np.pi],    [0, 100],  [0.001, 500]], # Model I

           [[1, 100],  [0.01, 10],   [np.pi , 2*np.pi],    [0, 500],  [0.001, 500]],  # Model J1
           [[1, 200], [0.01, 10], [0.5, 9]],  # Model B.1
           [[0, 100], [0, 100], [0, 90], [0.01, 10],   [np.pi , 2*np.pi],    [0, 500],  [0.001, 500], [0.1, 9.5]],     #Model H1
           [[1, 100],  [1, 100], [0.01, 10],   [np.pi , 2*np.pi],    [0, 500],  [0.001, 500]],                       # Model I1
           [[1,50],  [0.01, 10],  [1,50],  [25, 75], [0.1,5.0]], #

           [[12, 200], [0.01, 10], [1, 150]],
           [[1, 200],  [0.01, 10], [1, 150]],
           [[1, 200], [1, 200],  [0, 45], [0.01, 20], [1, 200]],
           [[12, 15], [1, 150],   [12, 150], [1, 300],   [0.01, 10], [0.00001, 500]],
           [[12, 150],  [12, 150], [0.01, 10], [0.00001, 100]],


           [[11, 150], [11, 150], [11, 150], [0.01, 10], [0.00001, 100], [0.00001, 100]],
           [[11, 150], [11, 150],  [1, 150], [0.00001, 10], [0.00001, 500]],
           [[1, 200], [0.01, 10], [0.1, 15]],
           [[12, 25], [12, 15],   [30, 45], [0.01, 10]],                        # Model D
           [[1, 200], [0.01, 10], [1, 150]],

           [[12, 200], [12, 200], [0.01, 10], [0.01, 50]],
           [[1, 25], [1, 25], [1, 50],  [-45, 45], [0.01, 10]],
           [[12, 150], [0.01, 10], [12, 50],  [0, 90], [0.1, 4]],
           [[12, 250], [12, 250], [12, 250], [12, 250], [0, 90],  [0.01, 10], [0.0001, 750]],
           [[12, 100],  [0.01, 10], [np.pi, 2 * np.pi], [0, 500], [0.001, 500]], # Model F

#
     #    ['a', 'b', 'alpha', 'dust_star_contrast', 'phi_blob', 'blob_radial_distance', 'blob_contrast', 'phi_blob2', 'blob_radial_distance2', 'blob_contrast2'],
           [[12, 100], [12, 100], [-45, 45], [0.01, 10], [np.pi, 2*np.pi], [0, 100], [0.001, 500], [np.pi, 2*np.pi], [0, 500], [0.001, 500]],
           [[15, 100], [15, 100],  [-45, 45], [0.01, 10], [np.pi, 2*np.pi], [0, 500], [0.001, 500],[np.pi, 2*np.pi], [0, 500], [0.001, 500]],
           [[12, 50], [0.01, 10],  [-50, 50], [-50, 50]],      #                # Model E
           [[12, 50], [12, 50],  [0, 45], [0.01, 10], [-50, 50], [-50, 50]],
           [[12, 100],  [0.01, 10],  [0.01, 400],  [-np.pi, np.pi]], #### model P[0, np.pi*2],[0.01, 400],
           [[12, 100], [12, 100], [0, 90], [0.01, 10], [-9, 9], [-9, 9]],

           [[12, 100], [12, 100], [0, 90], [0.01, 10], [np.pi, 2 * np.pi], [0, 500], [0.001, 500], [-9, 9], [-9, 9], [1, 40]],
           [[12, 100], [12, 100], [-25, 25], [0.01, 10], [np.pi, 2 * np.pi], [0, 500], [0.001, 500]],
           [[1, 100], [1, 100], [-25, 25], [0.01, 10], [np.pi, 2 * np.pi], [0, 100], [0.001, 500]],  # Model I
           [[1, 100],  [0.01, 10],   [np.pi , 2*np.pi],    [0,150],  [0.001, 500],  [-9, 9], [-9, 9]] ,
           [[1, 100],   [0.01, 10], [np.pi, 2 * np.pi], [0, 150], [0.001, 500],  [270, 450], [0.001, 15]],
            [[1, 100],   [0.01, 10],  [270, 450], [0.001, 15]]]




changing_by_settings =  [[['c', 'a'], ['b', 'a']],                                         # Model A thin spherical shell
                         [['b', 'a'], ['c', 'a']],                                         # Model B power law spherical shell
                         [['c', 'a'], ['b', 'a'], ['c2', 'a2'],  ['b2', 'a2']],                 # C
                         [['c', 'a']],                                                        # Model G
                         [['c', 'a']],        #, ['b', 'a']                                                   # Model H
                         [ ['c', 'a'], ['b', 'a']],                                                          # Model I

                         [['c', 'a'],['b', 'a']] ,                                                                     # Model J
                         [['c', 'a'],['b', 'a']],
                         [['c', 'a']],                                                                      # Model H1
                         [['c', 'a'], ['b', 'a']],                                                              # Model I1
                         [ ],   # just try for now.

                         [['c', 'a'], ['b', 'a']], # thick circle
                         [['c', 'a'], ['b', 'a']],
                         [['c', 'a']],
                         [['c', 'a'], ['b', 'a'], ['c2', 'a2'], ['b2', 'a2']],
                         [['c', 'a'], ['b', 'a'], ['c2', 'a2'], ['b2', 'a2']],


                         [['c', 'a'], ['b', 'a'], ['c2', 'a2'], ['b2', 'a2'], ['c3', 'a3'], ['b3', 'a3']],
                         [['c', 'a'], ['b', 'a'], ['c2', 'a2'], ['b2', 'a2']],
                         [['c', 'a'], ['b', 'a']],
                         [['c', 'a'] ], # Model D
                         [['c', 'a'], ['b', 'a']],


                         [['c', 'a'], ['b', 'a'], ['c2', 'a2'], ['b2', 'a2']],
                         [],
                         [['c', 'a']],
                         [['c', 'a'], ['c2', 'a2']],
                         [['c', 'a'], ['b', 'a']], # Model F


                         [['c', 'a']],
                         [['c', 'a']],
                         [['c', 'a'], ['b', 'a'] ], # Model E
                         [['c', 'a']],
                         [['c', 'a'], ['b', 'a']], #### model P
                         [['c', 'a']],

                         [['c', 'a']],
                         [['c', 'a']],
                         [['c', 'a']],
                         [['c', 'a'], ['b', 'a']],
                         [['c', 'a'], ['b', 'a']],
                         [['c', 'a'], ['b', 'a']]
                         ] # E 

num_steps_run_list =     [5000, 5000, 5000, 8000, 10000, 8000,
                          8000, 5000, 8000, 6000, 5000,
                          5000, 5000, 8000,5000, 5000,
                          5000, 5000, 12000, 8000, 5000,
                          7000, 5000, 5000, 10000, 4000,
                          15000, 10000, 15000, 15000, 10000, 5000,
                          8000, 12000, 12000, 8000, 10000, 8000]
num_steps_discard_list = [2500, 3000, 2000, 5000, 6000, 5000,
                          4000, 2000, 4000, 4000, 3000,
                          2000, 2000, 4000, 3000, 2000,
                          1000, 1000, 8000, 6000, 1000,
                          3000, 2000, 2000, 7000, 2000,
                          8000, 5000, 10000,10000, 3000, 2000,
                          4000, 8000, 8000, 5000, 5000, 4000]

                                                                                          # this was two_thick_circles

model_class = ['ellipse_thin', # Model A thin spherical shell
               'ellipse', # Model B p ower law spherical shell
               'two_thin_circles', # Model C - two thin shells
               'ellipse_blob_bright', # Model G  #this was a power law and i didnt realise
               'ellipse_and_blob',  # Model H
               'ellipse_blob_bright',    # Model I

               'ellipse_blob_bright', # Model J
               'ellipse',
               'ellipse_and_blob',  # Model H1
                'ellipse_blob_bright',  # Model I1
               'ellipse',


               'thick_ellipse',
                'thick_ellipse',
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
                'ellipse_thin',
               'bright_spot_ellipse',

               'ellipse_thin',
               'ellipse_blob_bright',
               'ellipse_blob_bright',
               'ellipse_blob_bright',
               'ellipse_and_blob',
               'ellipse_enhance_and_blob',
               'ellipse_enhance'
               ]

save = True
run = False
plot = True



 


for mod in range(0,1):#37, 38):
    print(mod)
    print(len(tags))
    for i in [0]:


        tag = tags[i]
        observing_run =  tags[i]
        print(tag)
        print(observing_run)
        print(model_class[mod])
        print(models[mod])



        if observing_run == 'muCep_2020':
            PA = 133
        elif observing_run == 'muCep_2018_02':
            PA = 175
        elif observing_run == 'muCep_2018_05':
            PA = 139.5
        elif observing_run == 'muCep_2017':
            PA = -121
        else:
            print('OBSERVING EPOCH NOT IDENTIFIED - STOP')
            cat = dog

        PA = PA - 78.9
        PA = -PA

        meta_data =  'geometric_models_data/{}/'.format(observing_run)
        save_dir = 'geometric_models_data/final_thesis/images/'
        save_dir_fit = 'geometric_models_data/final_thesis/fits/'
 
        scat_H = np.load(meta_data    + 'stokesH_scattering_8.0.npy')#
        scat_V = np.load(meta_data    + 'stokesV_scattering_8.0.npy')
        scat_H45 = np.load(meta_data  + 'stokesH45_scattering_8.0.npy')
        scat_V45 = np.load(meta_data  + 'stokesV45_scattering_8.0.npy')

        if models[mod] == "Model_B1" or 'Model_J1' or "Model_H1" or "Model_I1" or 'Model_shellthickcircle1' or 'Model_shellthickellipse1' or 'Model_shellthickellipse_21' or 'Model_plellipse1' or 'Model_plellipse1_1' or 'Model_G1' or 'Model_O1':

            scat_H = np.load(meta_data + 'stokesH_scattering_1.0.npy')  #
            scat_V = np.load(meta_data + 'stokesV_scattering_1.0.npy')
            scat_H45 = np.load(meta_data + 'stokesH45_scattering_1.0.npy')
            scat_V45 = np.load(meta_data + 'stokesV45_scattering_1.0.npy')



        ucoords = np.load(meta_data   + 'u_coords.npy')
        vcoords = np.load(meta_data   + 'v_coords.npy')
        print("NaNs in u:", jnp.isnan(ucoords).any())
        print("NaNs in v:", jnp.isnan(vcoords).any())
        print("Infs in u:", jnp.isinf(ucoords).any())
        print("Infs in v:", jnp.isinf(vcoords).any())


        xdata = np.arctan(vcoords / ucoords)
        zdata = np.sqrt(ucoords**2 + vcoords**2)
        uv_concat = np.concatenate((np.expand_dims(ucoords, axis = 1), np.expand_dims(vcoords, axis = 1)), axis = 1)
        indx_of_cp = np.load(meta_data  + 'indx_of_cp.npy')

        x, y, z = np.ogrid[-154:154, -154:154, -154:154]
        xx, yy = np.meshgrid(x.flatten(), y.flatten())
        dftm_grid = compute_DFTM1(xx.flatten(), yy.flatten(), uv_concat, wavelength)



        qpolr =   np.load(meta_data + 'q_vis.npy')
        upolr =   np.load(meta_data + 'u_vis.npy')
        qpolerr = np.load(meta_data + 'q_vis_err.npy')*2.5
        upolerr = np.load(meta_data + 'u_vis_err.npy')*2.5



        qpolr_cp =   np.load(meta_data + 'q_cp.npy')
        upolr_cp =   np.load(meta_data + 'u_cp.npy')
        qpolerr_cp = np.load(meta_data + 'q_cp_err.npy')
        upolerr_cp = np.load(meta_data + 'u_cp_err.npy')



        ydatar = np.concatenate((qpolr, upolr, qpolr_cp, upolr_cp))
        ydataerr = np.concatenate((qpolerr, upolerr, qpolerr_cp, upolerr_cp))

        if plot == True:
            plt.figure()
            plt.scatter(ucoords, vcoords)
            plt.savefig(save_dir + 'uvcoords_{}.pdf'.format(tag))
            plt.close()

        model_params = {'image_size': image_size,
                        'PA': PA,
                        'pixel_ratio': pixel_ratio,
                        'size_biggest_baseline_m': size_biggest_baseline_m,
                        'wavelength': wavelength,
                        'u_coords': ucoords,
                        'v_coords': vcoords,
                        'xdata': xdata,
                        'zdata': zdata,
                        'indx_of_cp':indx_of_cp,


                        'H_scat': scat_H,
                        'V_scat' : scat_V,
                        'H45_scat' : scat_H45,
                        'V45_scat' : scat_V45,
                        'thick1': 1,
                        'thick2': 3,
                        'thicka2':1,
                        'thickb' :1,
                        'thicka':10,
                        'thickb': 2,

                        'dftm_grid': dftm_grid,
                        'star_radius': star_radius,

                        'a': 35,
                        'b': 50,

                        'a2': 11,
                        'a3': 11,

                        'b2': 11,
                        'b3': 11,
                        'c2': 11,
                        'c3': 11,
                        'c': 60,
                        'h': 0,
                        'k': 0,
                        'n': 0,

                        'enhancement_amp': 2,
                        'enhancement_size': 90,
                        'enhancement_loc': 180,

                        'bright_location_theta': 2,
                        'bright_location_phi': np.pi / 8,
                        'bright_size': 0.45,#np.pi/8,
                        'bright_mag': 4,

                        'ellipse_scale': 2,
                        'ellipse_scale2': 4,
                        'ellipse_contrast_ratio2': 3,
                        'ellipse_contrast_ratio3': 3,
                        'alpha': 45,
                        'alpha2':20,
                        'dust_star_contrast': 8.5,
                        'ellipse_contrast_ratio': 0.5,


                        'ydata_real' : ydatar,
                        'ydata_real_err': ydataerr,

                        'dust_shape': model_class[mod],
                        'dust_pl_const': 1,
                        'dust_pl_exp': 3,
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
        stellar_object.make_dust()
        y_model_this = stellar_object.simulate_nrm()



        np.save(save_dir + 'true_obs_{}.npy'.format(tag), stellar_object.ydata_real)
        np.save(save_dir + 'true_obs_err_{}.npy'.format(tag), stellar_object.ydata_real_err)
        np.save(save_dir + 'ucoords_{}.npy'.format(tag), ucoords)
        np.save(save_dir + 'vcoords_{}.npy'.format(tag), vcoords)




        if plot == True:
            plt.figure(figsize = (10,8))
            plt.subplot(2,2,1)
            plt.title('Real Visibilities - Stokes Q')
            sc = plt.scatter(stellar_object.xdata, stellar_object.ydata_real[0:153], c=abs(stellar_object.zdata), cmap='jet')
            plt.xlabel('Azimuth Angle (radians)')
            plt.ylabel('Differential Visibility')
            cbar = plt.colorbar(sc)
            cbar.set_label('Baseline Length (m)')

            plt.subplot(2,2,2)
            plt.title('Model Visibilities - Stokes Q')
            sc = plt.scatter(stellar_object.xdata, stellar_object.y_model[0:153], c=abs(stellar_object.zdata), cmap='jet')
            plt.xlabel('Azimuth Angle (radians)')
            plt.ylabel('Differential Visibility')
            cbar = plt.colorbar(sc)
            cbar.set_label('Baseline Length (m)')

            plt.subplot(2,2,3)
            plt.title('Real Visibilities - Stokes U')
            sc = plt.scatter(stellar_object.xdata, stellar_object.ydata_real[153:153*2], c=abs(stellar_object.zdata), cmap='jet')
            plt.xlabel('Azimuth Angle (radians)')
            plt.ylabel('Differential Visibility')
            cbar = plt.colorbar(sc)
            cbar.set_label('Baseline Length (m)')

            plt.subplot(2,2,4)
            plt.title('Model Visibilities - Stokes U')
            sc = plt.scatter(stellar_object.xdata, stellar_object.y_model[153:153*2], c=abs(stellar_object.zdata), cmap='jet')
            plt.xlabel('Azimuth Angle (radians)')
            plt.ylabel('Differential Visibility')
            cbar = plt.colorbar(sc)
            cbar.set_label('Baseline Length (m)')
            plt.tight_layout()


            plt.savefig(save_dir + 'plot1plotjustthings {}_{}.pdf'.format('TESTING_2018', tag))
            plt.close()

        np.save(save_dir + 'model_temp_Q.npy', stellar_object.y_model[0:153])
        np.save(save_dir + 'model_temp_U.npy', stellar_object.y_model[153:153*2])



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
                model_params['ydata_real'] = stellar_object.make_pol_diff_vis()


                sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_model ,
                moves=[(emcee.moves.DEMove(), 0.8),
                        (emcee.moves.DESnookerMove(), 0.2),])

                state = sampler.run_mcmc(pos, num_steps, progress = True)
                not_flat_samples = sampler.get_chain(flat=False, discard = discard)#00)
                flat_samples = sampler.get_chain(flat = False, discard = discard)#00)

                fig = corner.corner(flat_samples, truth_color='b', labels=model_params['changing_by_theta'], show_titles=True, hist_bin_factor=2)
                plt.savefig(save_dir + 'corner_injection_{}.pdf'.format(k))

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
            plt.savefig(save_dir + 'injection_testing_{}_{}.pdf'.format(model, tag))



        if run == True:

            time_start = time.time()

            num_steps = num_steps_run
            original = os.getcwd() + '/emcee_fit_package.py'
            target = save_dir_fit +'emcee_fit_package_' + model_params['filename'] + '_' + tag + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_walkers.py'
            shutil.copyfile(original, target)

            sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob_model ,
            moves=[(emcee.moves.DEMove(), 0.8),
                    (emcee.moves.DESnookerMove(), 0.2),])

            state = sampler.run_mcmc(pos, num_steps, progress = True)

            time_end = time.time()
            print('Total time taken is {} seconds'.format(time_end - time_start))


        else:
            num_steps = num_steps_run
            with open(save_dir_fit + model_params['filename'] + '_' + tag + '_' + str(star_radius) + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_sampler.pkl', 'rb') as file:
                sampler = pickle.load(file)
            with open(save_dir_fit + model_params['filename'] + '_' + tag + '_' + str(star_radius) + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_state.pkl', 'rb') as file:
                state = pickle.load(file)
            with open(save_dir_fit + model_params['filename'] + '_' + tag + '_' + str(star_radius) + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_smodelparams.pkl', 'rb') as file:
                model_params = pickle.load(file)



        if save == True:
            filep = open(save_dir_fit + model_params['filename'] + '_' + tag + '_' + str(star_radius) + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_sampler.pkl', 'wb')
            pickle.dump(sampler, filep)
            filep.close()
            filep = open(save_dir_fit + model_params['filename'] + '_' + tag + '_' + str(star_radius) + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_state.pkl', 'wb')
            pickle.dump(state, filep)
            filep.close()
            filep = open(save_dir_fit  + model_params['filename'] + '_' + tag + '_' + str(star_radius) + '_samples_' + str(num_steps) + '_steps_' + str(nwalkers) + '_smodelparams.pkl', 'wb')
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




        logZ = log_evidence_laplace_mixed(flat_samples, log_likelihoods, lnprob_model,  prior) #log_evidence_laplace_mixed(samples, log_probs, log_prob_fn, prior_fn, adaptive_mode_detection=True):
        print('Log evidence is {}'.format(logZ))





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



        print(model_params['dust_shape']) #model_class[mod]
        stellar_object = geometric_star(model_params)
        stellar_object.make_dust()
        y_model_this = stellar_object.simulate_nrm()

        np.save(save_dir + 'modelMLE_obs_{}.npy'.format(tag), stellar_object.y_model)

        Q_vals_real = onp.concatenate((stellar_object.ydata_real[0:153], stellar_object.ydata_real[153*2:153*2 + 816]), axis = 0)
        U_vals_real = onp.concatenate((stellar_object.ydata_real[153:153*2], stellar_object.ydata_real[153*2 + 816:153*2 + 816*2]), axis = 0)

        Q_vals_model =  onp.concatenate((stellar_object.y_model[0:153], stellar_object.y_model[153*2:153*2 + 816]), axis = 0)
        U_vals_model = onp.concatenate((stellar_object.y_model[153:153*2], stellar_object.y_model[153*2 + 816:153*2 + 816*2]), axis = 0)

        Q_vals_error =  onp.concatenate((stellar_object.ydata_real_err[0:153], stellar_object.ydata_real_err[153*2:153*2 + 816]), axis = 0)
        U_vals_error = onp.concatenate((stellar_object.ydata_real_err[153:153*2], stellar_object.ydata_real_err[153*2 + 816:153*2 + 816*2]), axis = 0)

        chisqu_Q = uf.chi_squ_red(Q_vals_real, Q_vals_model, Q_vals_error, len(model_params['changing_by_theta']))
        chisqu_U = uf.chi_squ_red(U_vals_real, U_vals_model, U_vals_error, len(model_params['changing_by_theta']))


        final_chi_red = (chisqu_Q + chisqu_U)/2
        chisqu_reduced = uf.chi_squ_red(stellar_object.ydata_real, stellar_object.y_model, stellar_object.ydata_real_err, len(model_params['changing_by_theta']))
        print('The chi squared reduced value is  {}'.format(chisqu_reduced))

        chisqu = uf.chi_squ(stellar_object.ydata_real, stellar_object.y_model, stellar_object.ydata_real_err)
        print('The chi squared value is  {}'.format(chisqu))


        if final_chi_red > 1:

            nobs = len(stellar_object.ydata_real)
            nparams = len(model_params['changing_by_theta'])
            scaling_factor = np.sqrt(chisqu/(nobs - nparams))

        if plot == True:
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
            plt.savefig(save_dir + 'corner_plot_{}_{}.pdf'.format(model, tag))
            plt.close()

        if plot == True:
            fig, ax = plt.subplots(3, 2, gridspec_kw={'height_ratios': [1, 1, 1]})
            fig.set_size_inches(15, 13)

            im3 = ax[1,0].scatter(stellar_object.xdata, stellar_object.ydata_real[0:153], c=abs(stellar_object.zdata), cmap='jet', marker='x', label = 'Real')
            ax[1,0].errorbar(stellar_object.xdata, stellar_object.ydata_real[0:153], yerr = stellar_object.ydata_real_err[0:153], fmt = 'none', ecolor= 'lightgrey')
            ax[1,0].plot(stellar_object.xdata, np.mean(stellar_object.ydata_real[0:153]) * np.ones(np.shape(stellar_object.xdata)), label = 'Real $\mu$')
            im4 = ax[1,0].scatter(stellar_object.xdata, stellar_object.y_model[0:153], c=abs(stellar_object.zdata), cmap='jet', marker='o', alpha=0.1, label='Model', vmin=np.min(stellar_object.zdata), vmax=np.max(stellar_object.zdata))
            ax[1,0].plot(stellar_object.xdata, np.mean(stellar_object.y_model[0:153]) * np.ones(np.shape(stellar_object.y_model[0:153])), alpha=0.2, label='Model $\mu$')
            ax[1, 0].set_title(r'Modelled and Real Stokes Q $V^2$, $\chi_{{red}}^2$ {:.2f}'.format(chisqu_Q))
            cbar = fig.colorbar(im3, ax=ax[1,0], label = "Baseline Length (m)")
            cbar.set_label('Baseline Length (m)')
            ax[1,0].set_ylabel('Differential $V^2$')
            ax[1,0].set_xlabel('Azimuth Angle (rad)')


            im4 = ax[1,1].scatter(stellar_object.xdata, stellar_object.ydata_real[153:153 * 2], c=abs(stellar_object.zdata), cmap='jet', marker='x', label = 'Real')
            ax[1,1].errorbar(stellar_object.xdata, stellar_object.ydata_real[153:153 * 2], yerr = stellar_object.ydata_real_err[153:153 * 2], fmt = 'none', ecolor = 'lightgrey')
            ax[1,1].plot(stellar_object.xdata, np.mean(stellar_object.ydata_real[153:153 * 2]) * np.ones(np.shape(stellar_object.xdata)), label = 'Real $\mu$')
            im4 = ax[1,1].scatter(stellar_object.xdata, stellar_object.y_model[153:153 * 2], c=abs(stellar_object.zdata), cmap='jet', marker='o', alpha=0.1, label='Model', vmin=np.min(stellar_object.zdata), vmax=np.max(stellar_object.zdata))
            ax[1,1].plot(stellar_object.xdata, np.mean(stellar_object.y_model[153:153 * 2]) * np.ones(np.shape(stellar_object.y_model[153:153 * 2])), alpha=0.2, label='Model $\mu$')
            ax[1, 1].set_title(r'Modelled and Real Stokes U $V^2$, $\chi_{{red}}^2$ {:.2f}'.format(chisqu_U))
            cbar = fig.colorbar(im3, ax=ax[1,1], label = "Baseline Length (m)")
            cbar.set_label('Baseline Length (m)')
            ax[1,1].set_ylabel('Differential $V^2$')
            ax[1,1].set_xlabel('Azimuth Angle (rad)')
            ax[1,1].legend(loc='best')


            big_thing = np.max(np.array([np.abs(np.min(stellar_object.image_Q)), np.abs(np.max(stellar_object.image_Q))]))

            im7 = ax[0,0].imshow((stellar_object.image_Q), cmap = 'seismic', clim = [-big_thing, big_thing])#.sum(axis=2)))
            ax[0,0].set_title('Stokes Q Image')
            ax[0,0].set_xlabel('x (mas)')
            ax[0,0].set_ylabel('y (mas)')
            plt.tight_layout()
            cbar = fig.colorbar(im7, ax=ax[0,0], label = "Baseline Length (m)")
            cbar.set_label('Normalised Polarised Flux')


            np.save(save_dir + 'model_obs_{}_{}.npy'.format(model, tag),  stellar_object.y_model)
            np.save(save_dir + 'real_obs_{}_{}.npy'.format(model, tag),   stellar_object.ydata_real)
            np.save(save_dir + 'real_err_{}_{}.npy'.format(model, tag),   stellar_object.ydata_real_err)


            big_thing = np.max(np.array([np.abs(np.min(stellar_object.image_U)), np.abs(np.max(stellar_object.image_U))]))
            im77 = ax[0,1].imshow((stellar_object.image_U),  cmap = 'seismic', clim = [-big_thing, big_thing])
            ax[0,1].set_title('Stokes U Image')
            ax[0,1].set_xlabel('x (mas)')
            ax[0,1].set_ylabel('y (mas)')
            cbar = fig.colorbar(im7, ax=ax[0,1], label = "Baseline Length (m)")
            cbar.set_label('Normalised Polarised Flux')


            im5 = ax[2,0].scatter(stellar_object.ydata_real[0:153], stellar_object.y_model[0:153],
                                c=abs(stellar_object.zdata), cmap='jet',
                                marker='o',  label='M', vmin=np.min(stellar_object.zdata),
                                vmax=np.max(stellar_object.zdata))
            ax[2,0].set_title('Correlation between real and modelled $V^2$')
            length_thing = np.arange(np.min(stellar_object.ydata_real[0:153]), np.max(stellar_object.ydata_real[0:153]), 0.0001)
            ax[2,0].scatter(length_thing, length_thing, marker =  '.', c = 'k', vmin=np.min(stellar_object.zdata),
                                    vmax=np.max(stellar_object.zdata), label = 'x = y')
            cbar = fig.colorbar(im5, ax=ax[2,0], label = "Baseline Length (m)")
            ax[2,0].set_ylabel('Modelled Differential $V^2$')
            ax[2,0].set_xlabel('Real Differential $V^2$')


            im6 = ax[2,1].scatter(stellar_object.ydata_real[153:153 * 2], stellar_object.y_model[153:153 * 2],
                                c=abs(stellar_object.zdata), cmap='jet',
                                marker='o', label='M', vmin=np.min(stellar_object.zdata),
                                vmax=np.max(stellar_object.zdata))
            ax[2,1].set_title('Correlation between real and modelled $V^2$')
            length_thing = np.arange(np.min(stellar_object.ydata_real[153:153 * 2]), np.max(stellar_object.ydata_real[153:153 * 2]), 0.0001)
            ax[2,1].scatter(length_thing, length_thing, marker = '.', c = 'k', vmin=np.min(stellar_object.zdata),
                                    vmax=np.max(stellar_object.zdata), label = 'x = y')
            ax[2,1].set_ylabel('Modelled Differential $V^2$')
            ax[2,1].set_xlabel('Real Differential $V^2$')
            ax[2,1].legend(loc='best')
            cbar = fig.colorbar(im6, ax=ax[2,1], label = "Baseline Length (m)")

            plt.tight_layout()
            plt.savefig(save_dir + 'model_plots_{}_{}.pdf'.format(model, tag))
            plt.close()

        if plot == True:
            plt.figure()
            plt.scatter(ucoords, vcoords)
            plt.savefig(save_dir + 'uvcoords_{}_{}.pdf'.format(model, tag))
            plt.close()

        if plot == True:
            plt.figure(figsize = (10,8))
            plt.subplot(2,2,1)
            plt.title('real data Q')
            plt.scatter(stellar_object.xdata, stellar_object.ydata_real[0:153], c=abs(stellar_object.zdata), cmap='jet')
            plt.subplot(2,2,2)
            plt.title('Model data Q')
            plt.scatter(stellar_object.xdata, stellar_object.y_model[0:153], c=abs(stellar_object.zdata), cmap='jet')

            plt.subplot(2,2,3)
            plt.title('real data U')
            plt.scatter(stellar_object.xdata, stellar_object.ydata_real[153:153*2], c=abs(stellar_object.zdata), cmap='jet')
            plt.subplot(2,2,4)
            plt.title('model data U')
            plt.scatter(stellar_object.xdata, stellar_object.y_model[153:153*2], c=abs(stellar_object.zdata), cmap='jet')

            plt.savefig(save_dir + 'plot2plotjustthings {}_{}.pdf'.format(model, tag))
            plt.close()


        pol_P = np.sqrt(stellar_object.image_U ** 2 + stellar_object.image_Q ** 2)
        phi_P = np.arctan2(stellar_object.image_U, stellar_object.image_Q) / 2
        xE = np.arange(0, np.shape(phi_P)[0], 1)
        yE = np.arange(0, np.shape(phi_P)[0], 1)
        X, Y = np.meshgrid(xE, yE)
        if plot == True:
            plt.figure()
            plt.quiver(X, Y, pol_P / pol_P.max(), pol_P / pol_P.max(), angles=np.rad2deg(phi_P + np.pi / 2), pivot='mid',
                       headwidth=0, headlength=0, scale=25)
            plt.title('Polarization Vectors')
            plt.xlabel('mas')
            plt.ylabel('mas')
            plt.savefig(save_dir + 'polarisation_map_{}_{}.pdf'.format(model, tag))

        Q_vals_real = onp.concatenate((stellar_object.ydata_real[0:153], stellar_object.ydata_real[153 * 2:153 * 2 + 816]),  axis=0)
        U_vals_real = onp.concatenate( (stellar_object.ydata_real[153:153 * 2], stellar_object.ydata_real[153 * 2 + 816:153 * 2 + 816 * 2]), axis=0)

        Q_vals_model = onp.concatenate((stellar_object.y_model[0:153], stellar_object.y_model[153 * 2:153 * 2 + 816]), axis=0)
        U_vals_model = onp.concatenate(  (stellar_object.y_model[153:153 * 2], stellar_object.y_model[153 * 2 + 816:153 * 2 + 816 * 2]), axis=0)

        Q_vals_error = onp.concatenate(  (stellar_object.ydata_real_err[0:153], stellar_object.ydata_real_err[153 * 2:153 * 2 + 816]), axis=0)
        U_vals_error = onp.concatenate( (stellar_object.ydata_real_err[153:153 * 2], stellar_object.ydata_real_err[153 * 2 + 816:153 * 2 + 816 * 2]),  axis=0)



        bl_things = zdata[indx_of_cp]
        bl_cp = np.max(bl_things, axis = 0)


        chisqu_reducedQ = uf.chi_squ_red(stellar_object.ydata_real[153 * 2:153 * 2 + 816],
                                         stellar_object.y_model[153 * 2:153 * 2 + 816],
                                         stellar_object.ydata_real_err[153 * 2:153 * 2 + 816],
                                         len(model_params['changing_by_theta']))


        chisqu_reducedU = uf.chi_squ_red(stellar_object.ydata_real[153 * 2 + 816:153 * 2 + 816 * 2],
                                         stellar_object.y_model[153 * 2:153 * 2 + 816],
                                         stellar_object.ydata_real_err[153 * 2:153 * 2 + 816],
                                         len(model_params['changing_by_theta']))
        if plot == True:
            plt.figure(figsize = (10,4))

            plt.subplot(1,2,1)
            min_q = np.min(stellar_object.ydata_real[153 * 2:153 * 2 + 816])
            max_q = np.max(stellar_object.ydata_real[153 * 2:153 * 2 + 816])
            ones_to_plot = np.arange(min_q, max_q, min_q/100)
            plt.errorbar(stellar_object.ydata_real[153 * 2:153 * 2 + 816], stellar_object.y_model[153 * 2:153 * 2 + 816], xerr =  stellar_object.ydata_real_err[153 * 2:153 * 2 + 816], fmt='none', ecolor='lightgrey')
            plt.scatter(stellar_object.ydata_real[153 * 2:153 * 2 + 816], stellar_object.y_model[153 * 2:153 * 2 + 816], c = bl_cp , cmap = 'jet')
            plt.scatter(ones_to_plot, ones_to_plot, linestyle='--', color='k')
            plt.xlabel('Real Closure Phases (Degrees)')
            plt.ylabel('Model Clousre Phases (Degrees)')
            plt.title('Stokes Q Closure Phases, Chisq {:.2f}'.format(chisqu_reducedQ))
            plt.xlim([min_q- 0.05, max_q+ 0.05])
            plt.ylim([min_q- 0.05, max_q+ 0.05])

            plt.subplot(1,2,2)
            min_u = np.min(stellar_object.ydata_real[153 * 2 + 816:153 * 2 + 816 * 2])
            max_u = np.max(stellar_object.ydata_real[153 * 2 + 816:153 * 2 + 816 * 2])
            ones_to_plot = np.arange(min_u, max_u, min_u/100)



            plt.errorbar(stellar_object.ydata_real[153 * 2 + 816:153 * 2 + 816 * 2], stellar_object.y_model[153 * 2:153 * 2 + 816], xerr =  stellar_object.ydata_real_err[153 * 2:153 * 2 + 816], fmt='none', ecolor='lightgrey')
            plt.scatter(stellar_object.ydata_real[153 * 2 + 816:153 * 2 + 816 * 2], stellar_object.y_model[153 * 2:153 * 2 + 816], c = bl_cp , cmap = 'jet')
            plt.xlabel('Real Closure Phases (Degrees)')
            plt.ylabel('Model Clousre Phases (Degrees)')
            plt.scatter(ones_to_plot, ones_to_plot, linestyle='--', color='k')
            plt.xlim([min_u - 0.05, max_u+ 0.05])
            plt.ylim([min_u- 0.05, max_u+0.05])
            plt.title('Stokes U Closure Phases, Chisq {:.2f}'.format(chisqu_reducedU))

            plt.savefig(save_dir + 'closurephasefit_{}_{}.pdf'.format(model, tag))


        ########################################################################################################################
        ########################################################################################################################
        if plot == True:
            plt.figure(figsize = (24.5, 14))


            range_mins = [0, 2, 4, 6]
            range_maxs = [2, 4, 6, 8]

            Q_vals_real = onp.concatenate((stellar_object.ydata_real[0:153], stellar_object.ydata_real[153 * 2:153 * 2 + 816]),  axis=0)
            U_vals_real = onp.concatenate( (stellar_object.ydata_real[153:153 * 2], stellar_object.ydata_real[153 * 2 + 816:153 * 2 + 816 * 2]), axis=0)

            Q_vals_model = onp.concatenate((stellar_object.y_model[0:153], stellar_object.y_model[153 * 2:153 * 2 + 816]), axis=0)
            U_vals_model = onp.concatenate(  (stellar_object.y_model[153:153 * 2], stellar_object.y_model[153 * 2 + 816:153 * 2 + 816 * 2]), axis=0)

            Q_vals_error = onp.concatenate(  (stellar_object.ydata_real_err[0:153], stellar_object.ydata_real_err[153 * 2:153 * 2 + 816]), axis=0)
            U_vals_error = onp.concatenate( (stellar_object.ydata_real_err[153:153 * 2], stellar_object.ydata_real_err[153 * 2 + 816:153 * 2 + 816 * 2]),  axis=0)

            for i in range(len(range_mins)):

                rel_inds = np.argwhere((stellar_object.zdata > range_mins[i]) & (stellar_object.zdata <= range_maxs[i]))


                chisqu_Q = uf.chi_squ_red(Q_vals_real[rel_inds], Q_vals_model[rel_inds], Q_vals_error[rel_inds], len(model_params['changing_by_theta']))
                chisqu_U = uf.chi_squ_red(U_vals_real[rel_inds], U_vals_model[rel_inds], U_vals_error[rel_inds], len(model_params['changing_by_theta']))

                plt.subplot(4,4,i*4 + 1)
                plt.scatter(stellar_object.xdata[rel_inds], stellar_object.ydata_real[0:153][rel_inds], c=abs(stellar_object.zdata[rel_inds]), cmap='jet',
                                    marker='x', vmin=np.min(stellar_object.zdata), vmax=np.max(stellar_object.zdata), label = 'Data')
                plt.errorbar(stellar_object.xdata[rel_inds].flatten(), stellar_object.ydata_real[0:153][rel_inds].flatten(),
                             yerr=stellar_object.ydata_real_err[0:153][rel_inds].flatten(), fmt='none', ecolor='lightgrey')

                plt.plot(stellar_object.xdata[rel_inds], np.mean(stellar_object.ydata_real[0:153][rel_inds]) * np.ones(np.shape(stellar_object.xdata[rel_inds])), label = 'Data $\mu$')
                plt.scatter(stellar_object.xdata[rel_inds], stellar_object.y_model[0:153][rel_inds], c=abs(stellar_object.zdata[rel_inds]), cmap='jet', marker='o', alpha=1, label = 'Model', vmin=np.min(stellar_object.zdata), vmax=np.max(stellar_object.zdata))
                plt.plot(stellar_object.xdata[rel_inds],  np.mean(stellar_object.y_model[0:153][rel_inds]) * np.ones(np.shape(stellar_object.y_model[0:153][rel_inds])), alpha=0.5,  label='Model $\mu$')
                plt.title(f'Stokes Q, $\chi_{{red}}^2$ {chisqu_Q:.2f}')
                plt.ylabel('Differential Visibility, {}-{} m'.format(range_mins[i], range_maxs[i]))
                plt.xlabel('Azimuth Angle (rad)')
                # plt.legend(loc='lower left')

                plt.subplot(4,4,i*4 + 2)
                plt.scatter(stellar_object.xdata[rel_inds], stellar_object.ydata_real[153:153 * 2][rel_inds], c=abs(stellar_object.zdata[rel_inds]),
                                    cmap='jet', marker='x', vmin=np.min(stellar_object.zdata), vmax=np.max(stellar_object.zdata), label = 'Data')
                plt.errorbar(stellar_object.xdata[rel_inds].flatten(), stellar_object.ydata_real[153:153 * 2][rel_inds].flatten(),
                             yerr=stellar_object.ydata_real_err[153:153 * 2][rel_inds].flatten(), fmt='none', ecolor='lightgrey')
                # plt.colorbar()
                plt.plot(stellar_object.xdata[rel_inds],
                           np.mean(stellar_object.ydata_real[153:153 * 2][rel_inds]) * np.ones(np.shape(stellar_object.xdata[rel_inds])), label = 'Data $\mu$')
                plt.scatter(stellar_object.xdata[rel_inds], stellar_object.y_model[153:153 * 2][rel_inds], c=abs(stellar_object.zdata[rel_inds]),
                                    cmap='jet', marker='o', alpha=1, label = 'Model', vmin=np.min(stellar_object.zdata), vmax=np.max(stellar_object.zdata))
                plt.plot(stellar_object.xdata[rel_inds],
                           np.mean(stellar_object.y_model[153:153 * 2][rel_inds]) * np.ones(np.shape(stellar_object.y_model[153:153 * 2][rel_inds])),  alpha=0.5,  label='Model $\mu$')
                plt.title(f'Stokes U, $\chi_{{red}}^2$ {chisqu_U:.2f}')
                plt.ylabel('Differential Visibility, {}-{} m'.format(range_mins[i], range_maxs[i]))
                plt.xlabel('Azimuth Angle (rad)')
                plt.legend(loc='best')


                plt.subplot(4,4,i*4 + 3)
                length_thing = np.arange(np.min(stellar_object.ydata_real[0:153][rel_inds]), np.max(stellar_object.ydata_real[0:153][rel_inds]), 0.0001)
                plt.scatter(length_thing, length_thing, marker='.', c='k', vmin=np.min(stellar_object.zdata),
                                    vmax=np.max(stellar_object.zdata))

                plt.scatter(stellar_object.ydata_real[0:153][rel_inds], stellar_object.y_model[0:153][rel_inds],
                                    c=abs(stellar_object.zdata[rel_inds]), cmap='jet',
                                    marker='o', label='M', vmin=np.min(stellar_object.zdata),
                                    vmax=np.max(stellar_object.zdata))
                # plt.colorbar()#im5, ax=ax[2])
                plt.xlabel('Data')
                plt.ylabel('Model')
                plt.title('Stokes Q')


                length_thing = np.arange(np.min(stellar_object.ydata_real[153:153 * 2][rel_inds]), np.max(stellar_object.ydata_real[153:153 * 2][rel_inds]), 0.0001)
                plt.scatter(length_thing, length_thing, marker='.', c='k', vmin=np.min(stellar_object.zdata),
                                    vmax=np.max(stellar_object.zdata))


                plt.subplot(4,4,i*4 + 4)
                plt.scatter(stellar_object.ydata_real[153:153 * 2][rel_inds], stellar_object.y_model[153:153 * 2][rel_inds],
                                    c=abs(stellar_object.zdata[rel_inds]), cmap='jet',
                                    marker='o', label='M', vmin=np.min(stellar_object.zdata),
                                    vmax=np.max(stellar_object.zdata))
                plt.colorbar(label = 'Baseline Length (m)')
                plt.scatter(length_thing, length_thing, marker='.', c='k', vmin=np.min(stellar_object.zdata),
                                    vmax=np.max(stellar_object.zdata))

                plt.xlabel('Data')
                plt.ylabel('Model')
                plt.title('Stokes U')
            plt.tight_layout()

            plt.savefig(save_dir + 'model_plots_individual_{}_{}.pdf'.format(model, tag))


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


        QonI = stellar_object.image_Q/stellar_object.image_I
        PolP_ = np.sqrt(stellar_object.image_Q**2 + stellar_object.image_U**2)/(stellar_object.image_I + 1e-36)
        pol_P = np.sqrt(stellar_object.image_U ** 2 + stellar_object.image_Q ** 2)


        if plot == True:
            plt.figure()
            plt.imshow(PolP_)
            plt.title('Pol P {:.4f}, fracP {:.4f}, phi P {:.4f}'.format(np.sum(pol_P), np.sum(PolP_), np.mean(phi_P)))
            plt.colorbar()
            plt.savefig(save_dir + 'POLP_{}_{}.pdf'.format(model, tag))




        radial_profile = radial_average(np.abs(QonI * 100))
        if plot == True:
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

            plt.subplot(1,3,3) # thing
            plt.plot(radial_profile)
            plt.xlabel("Radius (pixels, mas)")
            plt.ylabel("Average abs(Q/I) %")
            plt.title("Radial Q/I Profile")
            plt.tight_layout()
            plt.savefig(save_dir + 'QonI_{}_{}.pdf'.format(model, tag))




        sto_I = scat_H + scat_V
        Q = scat_H - scat_V
        U = scat_H45 - scat_V45
        I = scat_H + scat_V

        P = np.sqrt(Q**2 + U**2)/(I + 1e-36)

        dust_shell_final = stellar_object.shell*P
        np.save(save_dir + 'dust_density_{}_{}.pdf'.format(model, tag), dust_shell_final)
        dust_shell_final = dust_shell_final/dust_shell_final.sum()

        np.save(save_dir + 'dust_only_{}_{}.pdf'.format(model, tag), stellar_object.shell)

        Q_image = Q*stellar_object.shell.sum(axis=2)
        U_image = U*stellar_object.shell.sum(axis=2)
        I_image = sto_I*stellar_object.shell.sum(axis=2)

        plt.figure(figsize = (10,3))
        plt.subplot(1,3,1)
        tep = np.max(np.array([np.max(np.abs(I_image.sum(axis=2))), np.min(np.abs(I_image.sum(axis=2)))]))
        plt.imshow(I_image.sum(axis=2), cmap = 'seismic', clim = [-tep, tep])
        plt.subplot(1,3,2)
        tep = np.max(np.array([np.max(np.abs(Q_image.sum(axis=2))), np.min(np.abs(Q_image.sum(axis=2)))]))
        plt.imshow(Q_image.sum(axis=2), cmap = 'seismic', clim = [-tep, tep])
        plt.subplot(1,3,3)
        tep = np.max(np.array([np.max(np.abs(U_image.sum(axis=2))), np.min(np.abs(U_image.sum(axis=2)))]))
        plt.imshow(U_image.sum(axis=2), cmap = 'seismic', clim = [-tep, tep])
        plt.savefig(save_dir + 'dust_images.pdf')

        if plot == True:
            plt.figure(figsize=(16, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(rotate(dust_shell_final.sum(axis=2), angle=PA, reshape=False))
            # plt.imshow(dust_shell_final.sum(axis=2))
            plt.title('Rotated, BIC {:.2f}'.format(BIC))
            plt.colorbar()

            plt.subplot(1, 3, 2)

            dust_shell_final = stellar_object.shell*P
            dust_shell_final = dust_shell_final/dust_shell_final.sum()
            aa = rotate(dust_shell_final.sum(axis=2), angle=PA, reshape=False)
            mid = np.shape(dust_shell_final.sum(axis=2))[0]
            plt.imshow(aa[120:170, 120:170])
            # plt.imshow(dust_shell_final.sum(axis=2))#, angle=PA, reshape=False))
            plt.title('Rotated P, {:.2f}'.format(chisqu_reduced))
            plt.colorbar()

            plt.subplot(1, 3, 3)

            Irot = np.expand_dims(rotate(stellar_object.image_I, angle=PA, reshape=False), axis=0)
            Qrot = np.expand_dims(rotate(stellar_object.image_Q, angle=PA, reshape=False), axis=0)
            Urot = np.expand_dims(rotate(stellar_object.image_U, angle=PA, reshape=False), axis=0)
            Vrot = np.zeros(np.shape(Urot))

            input_stokes = np.concatenate((Irot, Qrot, Urot, Vrot), axis=0)

            mmrotQ = uf.comp_higher_matrix_mult(rotator(-PA), input_stokes)[1,]
            tep = np.max(np.array([np.min(mmrotQ), np.max(mmrotQ)]))
            plt.imshow(mmrotQ, cmap='seismic', clim=[-tep*0.1, tep*0.1])
            plt.title('Log Prob is {:.10f}'.format( logZ ))
            plt.colorbar()




            plt.savefig(save_dir + 'rotatedmodel_plots_dustonly_{}_{}.pdf'.format(  model, tag))

            plt.close('all')


            np.save(save_dir + 'dust_shell_{}_{}.npy'.format(model, tag), aa)
            tt = uf.comp_higher_matrix_mult(rotator(-PA), input_stokes)
            np.save(save_dir + 'POLCUBE_{}_{}.npy'.format(model, tag),tt)


        try:
            autocorr_time = sampler.get_autocorr_time()[0]
            # print('************************************************************************************************************')
            print(autocorr_time)
        except Exception as e:
            print(f"Error calculating autocorrelation time: {e}")
            autocorr_time = None

 

 
