
import matplotlib.pyplot as plt
import os
import sys
plt.rcParams['image.origin'] = 'lower'
sys.path.append('../all_projects')
from useful_functions import *

from astropy.coordinates import EarthLocation, AltAz, SkyCoord

SUBARU_LOC = EarthLocation.of_site('SUBARU')
import numpy as np

import matplotlib.pyplot as plt
plt.ion()
plt.rcParams["image.origin"] = 'lower'
import pandas as pd
from scipy.ndimage import rotate
from astropy.coordinates import EarthLocation
SUBARU_LOC = EarthLocation.of_site('SUBARU')
from datetime import datetime
from astropy.io import fits
# from scipy.ndimage import rotate
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.time import Time
import astropy.units as u
plt.rcParams['image.origin'] = 'lower'
import glob
import jax
from jax import jit
import equinox
from jax import vmap
jax.config.update("jax_enable_x64", True)#True)
SUBARU_LOC = EarthLocation.of_site('SUBARU')

import jax.numpy as jnp

import jax
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


def rotate_image(image, angle, reshape=False):
    """
    Rotate a 2D image using JAX, similar to scipy.ndimage.rotate.

    Parameters:
        image: jnp.ndarray
            Input image array of shape (H, W).
        angle: float
            Rotation angle in degrees (counterclockwise).
        reshape: bool
            If False, output image will have the same shape as input.

    Returns:
        jnp.ndarray
            Rotated image.
    """
    # Convert angle to radians
    theta = -jnp.deg2rad(angle)  # Negative for correct direction

    # Get image dimensions
    H, W = image.shape

    # Define rotation matrix
    rotation_matrix = jnp.array([
        [jnp.cos(theta), -jnp.sin(theta)],
        [jnp.sin(theta), jnp.cos(theta)]
    ])

    # Create a grid of coordinates
    y, x = jnp.meshgrid(jnp.arange(H), jnp.arange(W), indexing='ij')
    coords = jnp.stack([y.ravel(), x.ravel()], axis=1)  # Shape (N, 2)

    # Compute the center of the image
    center = jnp.array([H / 2, W / 2])

    # Shift coordinates to origin, apply rotation, and shift back
    coords_shifted = coords - center
    rotated_coords = jnp.dot(coords_shifted, rotation_matrix.T) + center

    # Split into separate arrays for map_coordinates
    rotated_y, rotated_x = rotated_coords[:, 0], rotated_coords[:, 1]
    rotated_coords = jnp.stack([rotated_y, rotated_x], axis=0)

    # Interpolate pixel values
    rotated_image = map_coordinates(image, rotated_coords, order=1, mode='constant', cval=0)

    # Reshape to original image shape
    rotated_image = rotated_image.reshape((H, W))

    if reshape:
        raise NotImplementedError("reshape=True is not supported yet in this implementation")

    return rotated_image



def comp_higher_matrix_mult(matrixx, cube):
    """
    Apply matrix multiplication to each (4, 1) slice in the spatial dimensions of the cube.

    Parameters:
        matrixx: A (4, 4) matrix.
        cube: A (4, 309, 309) array.

    Returns:
        A (4, 309, 309) array after applying the matrix multiplication.
    """
    # Reshape cube to align the spatial dimensions for iteration
    original_shape = cube.shape  # (4, 309, 309)
    flat_cube = jnp.reshape(cube, (4, -1))  # Flatten spatial dimensions to (4, 309 * 309)

    # Define the function to apply
    def matmul_fn(slice):
        return jnp.matmul(matrixx, slice)  # Apply matrix multiplication to (4,)

    # Use vmap to apply matmul_fn along the flattened spatial axis
    mapped_fn = jax.vmap(matmul_fn, in_axes=1, out_axes=1)  # Map along the flattened dimension
    result_flat = mapped_fn(flat_cube)  # Result is (4, 309 * 309)

    # Reshape back to the original spatial dimensions
    final_mm = jnp.reshape(result_flat, original_shape)  # Back to (4, 309, 309)

    return final_mm



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
    cos2t = jnp.cos(2 * theta)
    sin2t = jnp.sin(2 * theta)
    M = jnp.array(((1, 0, 0, 0), (0, cos2t, sin2t, 0), (0, -sin2t, cos2t, 0), (0, 0, 0, 1)))
    return M


def wollaston(ordinary: bool = True, eta=1):
    """Return the Mueller matrix for a Wollaston prism or polarizing beamsplitter.

    Parameters
    ----------
    ordinary : bool, optional
        Return the ordinary beam's Mueller matrix, by default True
    eta : float, optional
        For imperfect beamsplitters, the diattenuation of the optic, by default 1

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix for the selected output beam

    Examples
    --------
    >>> wollaston()
    array([[0.5, 0.5, 0. , 0. ],
           [0.5, 0.5, 0. , 0. ],
           [0. , 0. , 0. , 0. ],
           [0. , 0. , 0. , 0. ]])

    >>> wollaston(False, eta=0.8)
    array([[ 0.5, -0.4,  0. ,  0. ],
           [-0.4,  0.5,  0. ,  0. ],
           [ 0. ,  0. ,  0.3,  0. ],
           [ 0. ,  0. ,  0. ,  0.3]])
    """
    eta = eta if ordinary else -eta

    radicand = (1 - eta) * (1 + eta)
    M = jnp.array(
        ((1, eta, 0, 0), (eta, 1, 0, 0), (0, 0, jnp.sqrt(radicand), 0), (0, 0, 0, jnp.sqrt(radicand)))
    )
    return 0.5 * M



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
    cos2t = jnp.cos(2 * theta)
    sin2t = jnp.sin(2 * theta)
    M = jnp.array(((1, 0, 0, 0), (0, cos2t, sin2t, 0), (0, -sin2t, cos2t, 0), (0, 0, 0, 1)))
    return M


def generic(theta=0, epsilon=0, delta=0):
    """Return a generic optic with diattenuation ``epsilon`` and phase retardance ``delta`` oriented at angle ``theta``.

    Parameters
    ----------
    theta : float, optional
        Rotation angle of the fast-axis in radians, by default 0
    epsilon : float, optional
        Diattenuation, by default 0
    delta : float, optional
        Retardance in radians, by default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix for the optic

    Examples
    --------
    >>> generic() # Identity
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -0.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  0., -0.,  1.]])

    >>> generic(epsilon=0.01, delta=np.pi) # mirror with diatt.
    array([[ 1.     ,  0.01   ,  0.     ,  0.     ],
           [ 0.01   ,  1.     ,  0.     , -0.     ],
           [ 0.     ,  0.     , -0.99995,  0.     ],
           [ 0.     ,  0.     , -0.     , -0.99995]])
    """
    cos2t = jnp.cos(2 * theta)
    sin2t = jnp.sin(2 * theta)
    cosd = jnp.cos(delta)
    sind = jnp.sin(delta)
    fac = jnp.sqrt((1 - epsilon) * (1 + epsilon))
    M = jnp.array(
        (
            (1, epsilon * cos2t, epsilon * sin2t, 0),
            (
                epsilon * cos2t,
                cos2t**2 + sin2t**2 * fac * cosd,
                cos2t * sin2t - fac * cosd * cos2t * sin2t,
                -fac * sind * sin2t,
            ),
            (
                epsilon * sin2t,
                cos2t * sin2t - fac * cosd * cos2t * sin2t,
                sin2t**2 + cos2t**2 * fac * cosd,
                fac * sind * cos2t,
            ),
            (0, fac * sind * sin2t, -fac * sind * cos2t, fac * cosd),
        )
    )

    return M



def waveplate(theta=0, delta=0):
    """Return the Mueller matrix for a waveplate with arbitrary phase retardance.

    Parameters
    ----------
    theta : float, optional
        Rotation
        t 0
    delta : float, optional
        Retardance in radians, by default 0

    Returns
    -------
    NDArray
        (4, 4) Mueller matrix representing the waveplate

    Examples
    --------
    >>> waveplate(0, np.pi) # HWP
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  1.,  0., -0.],
           [ 0.,  0., -1.,  0.],
           [ 0.,  0., -0., -1.]])

    >>> waveplate(np.deg2rad(45), np.pi/2) # QWP at 45°
    array([[ 1.,  0.,  0.,  0.],
           [ 0.,  0.,  0., -1.],
           [ 0.,  0.,  1.,  0.],
           [ 0.,  1., -0.,  0.]])
    """
    cos2t = jnp.cos(2 * theta)
    sin2t = jnp.sin(2 * theta)
    cosd = jnp.cos(delta)
    sind = jnp.sin(delta)
    a = (1 - cosd) * sin2t * cos2t
    M = jnp.array(
        (
            (1, 0, 0, 0),
            (0, cos2t**2 + cosd * sin2t**2, a, -sind * sin2t),
            (0, a, sin2t**2 + cosd * cos2t**2, sind * cos2t),
            (0, sind * sin2t, -sind * cos2t, cosd),
        )
    )
    return M


def hwp_adi_sync_offset(alt, az, lat=SUBARU_LOC.lat.rad):
    alpha = jnp.sin(az)
    beta = jnp.sin(alt) * jnp.cos(az) + jnp.cos(alt) * jnp.tan(lat)
    return 0.5 * jnp.arctan2(alpha, beta) + alt


def make_mm(pa=jnp.nan, alt=jnp.nan, az=jnp.nan, hwp=jnp.nan, imr=jnp.nan, flc_state=jnp.nan, camera=jnp.nan,
            hwp_adi_sync=False, dict_ideal={}):
    # PA rotation
    pa_theta = jnp.deg2rad(pa)
    m3_theta = jnp.deg2rad(dict_ideal['m3_offset'])
    m3 = generic(epsilon=dict_ideal['m3_diat'], theta=m3_theta, delta=jnp.pi)

    # Telescope rotation
    alt_theta = jnp.deg2rad(alt)
    tel_mm = rotator(-alt_theta) @ m3 @ rotator(pa_theta)

    # HWP
    if hwp_adi_sync:
        az_theta = jnp.deg2rad(az - 180)
        hwp_adi_offset = hwp_adi_sync_offset(alt=alt_theta, az=az_theta)
    else:
        hwp_adi_offset = 0

    hwp_theta = jnp.deg2rad(hwp + dict_ideal['hwp_offset']) + hwp_adi_offset
    hwp_mm = waveplate(hwp_theta, dict_ideal['hwp_phi'] * 2 * jnp.pi)

    # IMR
    imr_theta = jnp.deg2rad(imr + dict_ideal['imr_offset'])
    imr_mm = waveplate(imr_theta, dict_ideal['imr_phi'] * 2 * jnp.pi)

    # Optics
    optics_mm = generic(epsilon=dict_ideal['optics_diat'],
                        theta=jnp.deg2rad(dict_ideal['optics_theta']),
                        delta=dict_ideal['optics_phi'] * 2 * jnp.pi)

    cp_mm = optics_mm @ imr_mm @ hwp_mm @ tel_mm

    # FLC
    flc_theta = jnp.deg2rad(dict_ideal['flc_theta'][flc_state])
    flc_mm = waveplate(flc_theta, dict_ideal['flc_phi'] * 2 * jnp.pi)


    dichroic_mm = generic(epsilon=dict_ideal['dichroic_diat'],
                        theta=jnp.deg2rad(dict_ideal['dichroic_theta']),
                        delta=dict_ideal['dichroic_phi'] * 2 * jnp.pi)

    # Wollaston prism
    is_ordinary = camera == 1
    pbs_mm = wollaston(is_ordinary)

    # Final Mueller matrix
    M = pbs_mm @ dichroic_mm @ flc_mm @ cp_mm #cp_mm is optics_mm @ imr_mm @ hwp_mm @ tel_mm

    return M.astype(jnp.float32)


@equinox.filter_jit
def triple_diff_ideal(input_stokes, pa, alt, az, imr, hwp_adi_sync, derot, imrp, dict_ideal, dict_MMs):

    # if dict_MMs == []:
    #
    #     hwp0_flcA_cam1 = make_mm(pa=pa, alt=alt, az=az, hwp=0, imr=imr, flc_state='A', camera=1, hwp_adi_sync=hwp_adi_sync, dict_ideal = dict_ideal)
    #     hwp0_flcA_cam2 = make_mm(pa=pa, alt=alt, az=az, hwp=0, imr=imr, flc_state='A', camera=2, hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #     hwp0_flcB_cam1 = make_mm(pa=pa, alt=alt, az=az, hwp=0, imr=imr, flc_state='B', camera=1, hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #     hwp0_flcB_cam2 = make_mm(pa=pa, alt=alt, az=az, hwp=0, imr=imr, flc_state='B', camera=2, hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #
    #     hwp45_flcA_cam1 = make_mm(pa=pa, alt=alt, az=az, hwp=45, imr=imr, flc_state='A', camera=1,  hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #     hwp45_flcA_cam2 = make_mm(pa=pa, alt=alt, az=az, hwp=45, imr=imr, flc_state='A', camera=2, hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #     hwp45_flcB_cam1 = make_mm(pa=pa, alt=alt, az=az, hwp=45, imr=imr, flc_state='B', camera=1,  hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #     hwp45_flcB_cam2 = make_mm(pa=pa, alt=alt, az=az, hwp=45, imr=imr, flc_state='B', camera=2,  hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #
    #     hwp225_flcA_cam1 = make_mm(pa=pa, alt=alt, az=az, hwp=22.5, imr=imr, flc_state='A', camera=1, hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #     hwp225_flcA_cam2 = make_mm(pa=pa, alt=alt, az=az, hwp=22.5, imr=imr, flc_state='A', camera=2, hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #     hwp225_flcB_cam1 = make_mm(pa=pa, alt=alt, az=az, hwp=22.5, imr=imr, flc_state='B', camera=1, hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #     hwp225_flcB_cam2 = make_mm(pa=pa, alt=alt, az=az, hwp=22.5, imr=imr, flc_state='B', camera=2, hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #
    #     hwp675_flcA_cam1 = make_mm(pa=pa, alt=alt, az=az, hwp=67.5, imr=imr, flc_state='A', camera=1, hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #     hwp675_flcA_cam2 = make_mm(pa=pa, alt=alt, az=az, hwp=67.5, imr=imr, flc_state='A', camera=2, hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #     hwp675_flcB_cam1 = make_mm(pa=pa, alt=alt, az=az, hwp=67.5, imr=imr, flc_state='B', camera=1, hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #     hwp675_flcB_cam2 = make_mm(pa=pa, alt=alt, az=az, hwp=67.5, imr=imr, flc_state='B', camera=2, hwp_adi_sync=hwp_adi_sync, dict_ideal =dict_ideal)
    #
    # else:

    hwp0_flcA_cam1 = dict_MMs['hwp0_cam1_flcA_MM']
    hwp0_flcA_cam2 = dict_MMs['hwp0_cam2_flcA_MM']
    hwp0_flcB_cam1 = dict_MMs['hwp0_cam1_flcB_MM']
    hwp0_flcB_cam2 = dict_MMs['hwp0_cam2_flcB_MM']

    hwp45_flcA_cam1 = dict_MMs['hwp45_cam1_flcA_MM']
    hwp45_flcA_cam2 = dict_MMs['hwp45_cam2_flcA_MM']
    hwp45_flcB_cam1 = dict_MMs['hwp45_cam1_flcB_MM']
    hwp45_flcB_cam2 = dict_MMs['hwp45_cam2_flcB_MM']

    hwp225_flcA_cam1 = dict_MMs['hwp225_cam1_flcA_MM']
    hwp225_flcA_cam2 = dict_MMs['hwp225_cam2_flcA_MM']
    hwp225_flcB_cam1 = dict_MMs['hwp225_cam1_flcB_MM']
    hwp225_flcB_cam2 = dict_MMs['hwp225_cam2_flcB_MM']

    hwp675_flcA_cam1 = dict_MMs['hwp675_cam1_flcA_MM']
    hwp675_flcA_cam2 = dict_MMs['hwp675_cam2_flcA_MM']
    hwp675_flcB_cam1 = dict_MMs['hwp675_cam1_flcB_MM']
    hwp675_flcB_cam2 = dict_MMs['hwp675_cam2_flcB_MM']

    hwp0_flcA_cam1_ =  comp_higher_matrix_mult(hwp0_flcA_cam1, input_stokes)[0,]
    hwp0_flcA_cam2_ = comp_higher_matrix_mult(hwp0_flcA_cam2, input_stokes)[0,]
    hwp0_flcB_cam1_ =  comp_higher_matrix_mult(hwp0_flcB_cam1, input_stokes)[0,]
    hwp0_flcB_cam2_ = comp_higher_matrix_mult(hwp0_flcB_cam2, input_stokes)[0,]

    hwp45_flcA_cam1_ =  comp_higher_matrix_mult(hwp45_flcA_cam1, input_stokes)[0,]
    hwp45_flcA_cam2_ = comp_higher_matrix_mult(hwp45_flcA_cam2, input_stokes)[0,]
    hwp45_flcB_cam1_ =  comp_higher_matrix_mult(hwp45_flcB_cam1, input_stokes)[0,]
    hwp45_flcB_cam2_ = comp_higher_matrix_mult(hwp45_flcB_cam2, input_stokes)[0,]

    hwp225_flcA_cam1_ =  comp_higher_matrix_mult(hwp225_flcA_cam1, input_stokes)[0,]
    hwp225_flcA_cam2_ = comp_higher_matrix_mult(hwp225_flcA_cam2, input_stokes)[0,]
    hwp225_flcB_cam1_ =  comp_higher_matrix_mult(hwp225_flcB_cam1, input_stokes)[0,]
    hwp225_flcB_cam2_ = comp_higher_matrix_mult(hwp225_flcB_cam2, input_stokes)[0,]

    hwp675_flcA_cam1_ =  comp_higher_matrix_mult(hwp675_flcA_cam1, input_stokes)[0,]
    hwp675_flcA_cam2_ = comp_higher_matrix_mult(hwp675_flcA_cam2, input_stokes)[0,]
    hwp675_flcB_cam1_ =  comp_higher_matrix_mult(hwp675_flcB_cam1, input_stokes)[0,]
    hwp675_flcB_cam2_ = comp_higher_matrix_mult(hwp675_flcB_cam2, input_stokes)[0,]

    hwp0_flcA_cam1_ = rotate_image(hwp0_flcA_cam1_, (140.4 - 180 + imrp), reshape=False)
    hwp0_flcA_cam2_ = rotate_image(hwp0_flcA_cam2_, (140.4 - 180 + imrp), reshape=False)
    hwp0_flcB_cam1_ = rotate_image(hwp0_flcB_cam1_, (140.4 - 180 + imrp), reshape=False)
    hwp0_flcB_cam2_ = rotate_image(hwp0_flcB_cam2_, (140.4 - 180 + imrp), reshape=False)

    hwp45_flcA_cam1_ = rotate_image(hwp45_flcA_cam1_, (140.4 - 180 + imrp), reshape=False)
    hwp45_flcA_cam2_ = rotate_image(hwp45_flcA_cam2_, (140.4 - 180 + imrp), reshape=False)
    hwp45_flcB_cam1_ = rotate_image(hwp45_flcB_cam1_, (140.4 - 180 + imrp), reshape=False)
    hwp45_flcB_cam2_ = rotate_image(hwp45_flcB_cam2_, (140.4 - 180 + imrp), reshape=False)

    hwp225_flcA_cam1_ = rotate_image(hwp225_flcA_cam1_, (140.4 - 180 + imrp), reshape=False)
    hwp225_flcA_cam2_ = rotate_image(hwp225_flcA_cam2_, (140.4 - 180 + imrp), reshape=False)
    hwp225_flcB_cam1_ = rotate_image(hwp225_flcB_cam1_, (140.4 - 180 + imrp), reshape=False)
    hwp225_flcB_cam2_ = rotate_image(hwp225_flcB_cam2_, (140.4 - 180 + imrp), reshape=False)

    hwp675_flcA_cam1_ = rotate_image(hwp675_flcA_cam1_, (140.4 - 180 + imrp), reshape=False)
    hwp675_flcA_cam2_ = rotate_image(hwp675_flcA_cam2_, (140.4 - 180 + imrp), reshape=False)
    hwp675_flcB_cam1_ = rotate_image(hwp675_flcB_cam1_, (140.4 - 180 + imrp), reshape=False)
    hwp675_flcB_cam2_ = rotate_image(hwp675_flcB_cam2_, (140.4 - 180 + imrp), reshape=False)

    hwp0_1 = hwp0_flcA_cam1_ - hwp0_flcA_cam2_
    hwp0_2 = hwp0_flcB_cam1_ - hwp0_flcB_cam2_
    hwp0_3 = hwp0_1 - hwp0_2

    hwp45_1 = hwp45_flcA_cam1_ - hwp45_flcA_cam2_
    hwp45_2 = hwp45_flcB_cam1_ - hwp45_flcB_cam2_
    hwp45_3 = hwp45_1 - hwp45_2

    hwpQ = hwp0_3 - hwp45_3

    hwp225_1 = hwp225_flcA_cam1_ - hwp225_flcA_cam2_
    hwp225_2 = hwp225_flcB_cam1_ - hwp225_flcB_cam2_
    hwp225_3 = hwp225_1 - hwp225_2

    hwp675_1 = hwp675_flcA_cam1_ - hwp675_flcA_cam2_
    hwp675_2 = hwp675_flcB_cam1_ - hwp675_flcB_cam2_
    hwp675_3 = hwp675_1 - hwp675_2

    hwpU = hwp225_3 - hwp675_3

    hwpI = hwp0_flcA_cam1_ + hwp0_flcA_cam2_ + hwp0_flcB_cam1_ + hwp0_flcB_cam2_ + hwp45_flcA_cam1_ + hwp45_flcA_cam2_ + hwp45_flcB_cam1_ + hwp45_flcB_cam2_

    return hwpI, hwpQ, hwpU




def parallactic_angle_altaz(alt, az, lat=19.823806):
    """
    Calculate parallactic angle using the altitude/elevation and aziumth directly
    ```math
    ```
    Parameters
    ----------
    alt : float
        altitude or elevation, in degrees
    az : float
        azimuth, in degrees CCW from North
    lat : float, optional
        latitude of observation in degrees, by default 19.823806
    Returns
    -------
    float
        parallactic angle, in degrees East of North
    """
    ## Astronomical Algorithms, Jean Meeus
    # get angles, rotate az to S
    _az = np.deg2rad(az) - np.pi
    _alt = np.deg2rad(alt)
    _lat = np.deg2rad(lat)
    # calculate values ahead of time
    sin_az, cos_az = np.sin(_az), np.cos(_az)
    sin_alt, cos_alt = np.sin(_alt), np.cos(_alt)
    sin_lat, cos_lat = np.sin(_lat), np.cos(_lat)
    # get declination
    dec = np.arcsin(sin_alt * sin_lat - cos_alt * cos_lat * cos_az)
    # get hour angle
    ha = np.arctan2(sin_az, cos_az * sin_lat + np.tan(_alt) * cos_lat)
    # get parallactic angle
    pa = np.arctan2(np.sin(ha), np.tan(_lat) * np.cos(dec) - np.sin(dec) * np.cos(ha))
    return np.rad2deg(pa)


image_size = 308
pixel_ratio = 1


chems = ['mg95_fe05', 'CorundumCrystal'] #'Silica'] # 'Enstatite', 'Forsterite', 'Al2O3',  'EnstatiteCrystal', 'ForsteriteCrystal',  'Spinel'] # 'CorundumCrystal',  'mg95_fe05', 'Silica']#,
#'Silica','CorundumCrystal',
#] 'mg95_fe05', 'CorundumCrystal', #mmg95_fe05

         #, 'ForsteriteCrystal', 'Spinel',)
       #  'Silica', 'CorundumCrystal', 'mg60_fe40', 'mg70_fe30', 'mg80_fe20', 'mg95_fe05', 'mg0.95_fe0.05_olivine',
       #  'mg0.80_fe0.20_olivine', 'mg0.70_fe0.30_olivine', 'mg0.60_fe0.40_olivine']

# chems = ['Enstatite', 'Forsterite', 'Al2O3', 'Olivine', 'pyroxene', 'EnstatiteCrystal', 'ForsteriteCrystal', 'Spinel',
#          'Silica', 'CorundumCrystal', 'mg60_fe40', 'mg70_fe30', 'mg80_fe20', 'mg95_fe05', 'mg0.95_fe0.05_olivine',
#          'mg0.80_fe0.20_olivine', 'mg0.70_fe0.30_olivine', 'mg0.60_fe0.40_olivine']


# radius_specs = [(1,500,-4),
#                 (1,500,-3),
#                 (1,500,-2),
#                 (1,700,-4),
#                 (1,700,-3),
#                 (1,700,-2),
#                 (1,900,-4),
#                 (1,900,-3),
#                 (1,900,-2),
#                 (1, 1000, -4),
#                 (1, 1000, -3),
#                 (1, 1000, -2) ]

# radius_specs = [(1100, 1100.1, -0.1),
#                 (1200, 1200.1, -0.1),
#                 (1300, 1300.1, -0.1),
#                 (1400, 1400.1, -0.1),
#                 (1500, 1500.1, -0.1),
#                 (1600, 1600.1, -0.1),
#                 (1700, 1700.1, -0.1),
#                 (1800, 1800.1, -0.1),
#                 (1900, 1900.1, -0.1),
#                 (2000, 2000.1, -0.1),
#
#                 (1, 1100, -4),
#                 (1, 1100, -3),
#                 (1, 1100, -2),
#
#                 (1, 1200, -4),
#                 (1, 1200, -3),
#                 (1, 1200, -2),
#
#                 (1, 1300, -4),
#                 (1, 1300, -3),
#                 (1, 1300, -2),
#
#                 (1, 1400, -4),
#                 (1, 1400, -3),
#                 (1, 1400, -2),
#
#                 (1, 1500, -4),
#                 (1, 1500, -3),
#                 (1, 1500, -2),
#
#                 (1, 1600, -4),
#                 (1, 1600, -3),
#                 (1, 1600, -2),
#
#                 (1, 1700, -4),
#                 (1, 1700, -3),
#                 (1, 1700, -2),
#
#                 (1, 1800, -4),
#                 (1, 1800, -3),
#                 (1, 1800, -2),
#
#                 (1, 1900, -4),
#                 (1, 1900, -3),
#                 (1, 1900, -2),
#
#                 (1, 2000, -4),
#                 (1, 2000, -3),
#                 (1, 2000, -2) ]




radius_specs = [(100, 1000, -4),
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
                (500, 2000, -2)]


save_rad_keys = [f"{a}-{b}-{abs(p)}" for a,b,p in radius_specs]
wvs = [610, 670, 720, 760]
bandpass = 50

for wv in wvs:
    print(wv)
    for chem in chems:
        print(chem)
        for grain in save_rad_keys:
            print(grain)

            tag = f"m{chem}_r{grain}_w{wv}_band{int(bandpass)}"


            stH45 = np.load( '/import/*1/*/mie_scat_grids_raw/H45_scat_{}.npy'.format(tag))[0:308, 0:308, 0:308]
            stV45 = np.load( '/import/*1/*/mie_scat_grids_raw/V45_scat_{}.npy'.format(tag))[0:308, 0:308, 0:308]
            stV = np.load('/import/*1/*/mie_scat_grids_raw/V_scat_{}.npy'.format(tag))[0:308, 0:308, 0:308]
            stH = np.load('/import/*1/*/mie_scat_grids_raw/H_scat_{}.npy'.format(tag))[0:308, 0:308,  0:308]

            obs = 'muCep_2023'

            if obs == 'muCep_2023':
                imr = 89.05153818827708
                alt = 49.08596894316163
                imrpad = -63.798
                az = 346
                pa = 154.0516583509961
                DRA =  140.4 - 180 + imrpad
                pp_ = '/import/*1/*/mie_scat_grids/muCep_2023/'


                if wv == 610:
                    print('DOING 610 with the changed thingies')

                    dict_MMs ={'hwp0_cam1_flcA_MM': np.array([[0.50942406, 0.21149027, 0.44578929, 0.12671484],
                       [0.50942406, 0.21149027, 0.44578929, 0.12671484],
                       [0.        , 0.        , 0.        , 0.        ],
                       [0.        , 0.        , 0.        , 0.        ]]), 'hwp0_cam1_flcB_MM': np.array([[ 0.4905833 , -0.20123622, -0.42830336, -0.12935294],
                       [ 0.4905833 , -0.20123622, -0.42830336, -0.12935294],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp0_cam2_flcA_MM': np.array([[ 0.49057594, -0.20266206, -0.42791933, -0.12836626],
                       [-0.49057594,  0.20266206,  0.42791933,  0.12836626],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp0_cam2_flcB_MM': np.array([[ 0.5094167 ,  0.21006443,  0.44617332,  0.12770153],
                       [-0.5094167 , -0.21006443, -0.44617332, -0.12770153],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp45_cam1_flcA_MM': np.array([[ 0.50942406, -0.20923431, -0.36557023,  0.28651751],
                       [ 0.50942406, -0.20923431, -0.36557023,  0.28651751],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp45_cam1_flcB_MM': np.array([[ 0.4905833 ,  0.19979225,  0.34793637, -0.28230357],
                       [ 0.4905833 ,  0.19979225,  0.34793637, -0.28230357],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp45_cam2_flcA_MM': np.array([[ 0.49057594,  0.20112989,  0.34809271, -0.28114584],
                       [-0.49057594, -0.20112989, -0.34809271,  0.28114584],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp45_cam2_flcB_MM': np.array([[ 0.5094167 , -0.20789667, -0.36541389,  0.28767524],
                       [-0.5094167 ,  0.20789667,  0.36541389, -0.28767524],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp225_cam1_flcA_MM': np.array([[ 0.50942406,  0.41037754, -0.18815814,  0.23600775],
                       [ 0.50942406,  0.41037754, -0.18815814,  0.23600775],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp225_cam1_flcB_MM': np.array([[ 0.4905833 , -0.39246533,  0.1785053 , -0.2340487 ],
                       [ 0.4905833 , -0.39246533,  0.1785053 , -0.2340487 ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp225_cam2_flcA_MM': np.array([[ 0.49057594, -0.39236128,  0.17998517, -0.23307236],
                       [-0.49057594,  0.39236128, -0.17998517,  0.23307236],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp225_cam2_flcB_MM': np.array([[ 0.5094167 ,  0.4104816 , -0.18667828,  0.23698409],
                       [-0.5094167 , -0.4104816 ,  0.18667828, -0.23698409],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp675_cam1_flcA_MM': np.array([[ 0.50942406, -0.36132436,  0.2590901 ,  0.24865619],
                       [ 0.50942406, -0.36132436,  0.2590901 ,  0.24865619],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp675_cam1_flcB_MM': np.array([[ 0.4905833 ,  0.3435206 , -0.24944557, -0.24585052],
                       [ 0.4905833 ,  0.3435206 , -0.24944557, -0.24585052],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp675_cam2_flcA_MM': np.array([[ 0.49057594,  0.34377805, -0.2504743 , -0.24442592],
                       [-0.49057594, -0.34377805,  0.2504743 ,  0.24442592],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp675_cam2_flcB_MM': np.array([[ 0.5094167 , -0.3610669 ,  0.25806137,  0.25008079],
                       [-0.5094167 ,  0.3610669 , -0.25806137, -0.25008079],
                       [ 0.        ,  0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.        ,  0.        ]])}

                    dict_ideal = {'m3_diat': 0,
                                  'm3_offset': 0,  # deg
                                  'hwp_offset': 0,  # was 0# deg
                                  'hwp_phi': 0.5,  # wave
                                  'imr_offset': 0,  # deg
                                  'imr_phi': 0.5,  # wave
                                  'optics_diat': 0,
                                  'optics_theta': 0,  # deg #
                                  'optics_phi': 0,  # wave
                                  'flc_theta': {"A": 0, "B": 45},  # 0, 45 before
                                  'flc_phi': 0.5,
                                  'dichroic_diat': 0,
                                  'dichroic_theta': 0,
                                  'dichroic_phi': 0}

                    dict_ideal['hwp_offset'] = 3

                elif wv == 670:

                    dict_MMs = {'hwp0_cam1_flcA_MM': np.array([[0.5       , 0.35495547, 0.35138163, 0.02318542],
                               [0.5       , 0.35495547, 0.35138163, 0.02318542],
                               [0.        , 0.        , 0.        , 0.        ],
                               [0.        , 0.        , 0.        , 0.        ]]), 'hwp0_cam1_flcB_MM': np.array([[ 0.5       , -0.32687163, -0.37707721, -0.03110815],
                               [ 0.5       , -0.32687163, -0.37707721, -0.03110815],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp0_cam2_flcA_MM': np.array([[ 0.5       , -0.35495547, -0.35138163, -0.02318542],
                               [-0.5       ,  0.35495547,  0.35138163,  0.02318542],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp0_cam2_flcB_MM': np.array([[ 0.5       ,  0.32687163,  0.37707721,  0.03110815],
                               [-0.5       , -0.32687163, -0.37707721, -0.03110815],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp45_cam1_flcA_MM': np.array([[ 0.5       , -0.34518851, -0.33334312,  0.14045374],
                               [ 0.5       , -0.34518851, -0.33334312,  0.14045374],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp45_cam1_flcB_MM': np.array([[ 0.5       ,  0.31654234,  0.35376323, -0.15701122],
                               [ 0.5       ,  0.31654234,  0.35376323, -0.15701122],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp45_cam2_flcA_MM': np.array([[ 0.5       ,  0.34518851,  0.33334312, -0.14045374],
                               [-0.5       , -0.34518851, -0.33334312,  0.14045374],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp45_cam2_flcB_MM': np.array([[ 0.5       , -0.31654234, -0.35376323,  0.15701122],
                               [-0.5       ,  0.31654234,  0.35376323, -0.15701122],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp225_cam1_flcA_MM': np.array([[ 0.5       ,  0.34731496, -0.34279203,  0.10893092],
                               [ 0.5       ,  0.34731496, -0.34279203,  0.10893092],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp225_cam1_flcB_MM': np.array([[ 0.5       , -0.37072407,  0.31355351, -0.1193644 ],
                               [ 0.5       , -0.37072407,  0.31355351, -0.1193644 ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp225_cam2_flcA_MM': np.array([[ 0.5       , -0.34731496,  0.34279203, -0.10893092],
                               [-0.5       ,  0.34731496, -0.34279203,  0.10893092],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp225_cam2_flcB_MM': np.array([[ 0.5       ,  0.37072407, -0.31355351,  0.1193644 ],
                               [-0.5       , -0.37072407,  0.31355351, -0.1193644 ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp675_cam1_flcA_MM': np.array([[ 0.5       , -0.33327988,  0.35925808,  0.09928824],
                               [ 0.5       , -0.33327988,  0.35925808,  0.09928824],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp675_cam1_flcB_MM': np.array([[ 0.5       ,  0.35179733, -0.33370003, -0.12199561],
                               [ 0.5       ,  0.35179733, -0.33370003, -0.12199561],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp675_cam2_flcA_MM': np.array([[ 0.5       ,  0.33327988, -0.35925808, -0.09928824],
                               [-0.5       , -0.33327988,  0.35925808,  0.09928824],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp675_cam2_flcB_MM': np.array([[ 0.5       , -0.35179733,  0.33370003,  0.12199561],
                               [-0.5       ,  0.35179733, -0.33370003, -0.12199561],
                               [ 0.        ,  0.        ,  0.        ,  0.        ],
                               [ 0.        ,  0.        ,  0.        ,  0.        ]])}

                    dict_ideal = {'m3_diat': 0,
                                  'm3_offset': 0,  # deg
                                  'hwp_offset': 0,  # was 0# deg
                                  'hwp_phi': 0.5,  # wave
                                  'imr_offset': 0,  # deg
                                  'imr_phi': 0.5,  # wave
                                  'optics_diat': 0,
                                  'optics_theta': 0,  # deg #
                                  'optics_phi': 0,  # wave
                                  'flc_theta': {"A": 0, "B": 45},  # 0, 45 before
                                  'flc_phi': 0.5,
                                  'dichroic_diat': 0,
                                  'dichroic_theta': 0,
                                  'dichroic_phi': 0}


                elif wv == 720:

                    dict_MMs = {'hwp0_cam1_flcA_MM': np.array([[ 0.5       ,  0.4101816 ,  0.26685693, -0.10265689],
                                [ 0.5       ,  0.4101816 ,  0.26685693, -0.10265689],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp0_cam1_flcB_MM': np.array([[ 0.5       , -0.43517118, -0.2444088 ,  0.02983923],
                                [ 0.5       , -0.43517118, -0.2444088 ,  0.02983923],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp0_cam2_flcA_MM': np.array([[ 0.5       , -0.22361737, -0.44424731, -0.051377  ],
                                [-0.5       ,  0.22361737,  0.44424731,  0.051377  ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp0_cam2_flcB_MM': np.array([[ 0.5       ,  0.23497984,  0.43162266,  0.09212142],
                                [-0.5       , -0.23497984, -0.43162266, -0.09212142],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp45_cam1_flcA_MM': np.array([[ 0.5       , -0.4064524 , -0.28461804, -0.06155502],
                                [ 0.5       , -0.4064524 , -0.28461804, -0.06155502],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp45_cam1_flcB_MM': np.array([[ 0.5       ,  0.43298721,  0.24997856, -0.00572662],
                                [ 0.5       ,  0.43298721,  0.24997856, -0.00572662],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp45_cam2_flcA_MM': np.array([[ 0.5       ,  0.22181195,  0.43044343, -0.12457089],
                                [-0.5       , -0.22181195, -0.43044343,  0.12457089],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp45_cam2_flcB_MM': np.array([[ 0.5       , -0.23403998, -0.41097331,  0.16225358],
                                [-0.5       ,  0.23403998,  0.41097331, -0.16225358],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp225_cam1_flcA_MM': np.array([[ 0.5       ,  0.27693821, -0.41096001, -0.06646127],
                                [ 0.5       ,  0.27693821, -0.41096001, -0.06646127],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp225_cam1_flcB_MM': np.array([[ 0.5       , -0.24799562,  0.43413893, -0.0046434 ],
                                [ 0.5       , -0.24799562,  0.43413893, -0.0046434 ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp225_cam2_flcA_MM': np.array([[ 0.5       , -0.43860055,  0.21912439, -0.09805129],
                                [-0.5       ,  0.43860055, -0.21912439,  0.09805129],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp225_cam2_flcB_MM': np.array([[ 0.5       ,  0.42233093, -0.22947517,  0.13775968],
                                [-0.5       , -0.42233093,  0.22947517, -0.13775968],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp675_cam1_flcA_MM': np.array([[ 0.5       , -0.28893164,  0.39783371, -0.09081217],
                                [ 0.5       , -0.28893164,  0.39783371, -0.09081217],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp675_cam1_flcB_MM': np.array([[ 0.5       ,  0.25268078, -0.43059409,  0.02722411],
                                [ 0.5       ,  0.25268078, -0.43059409,  0.02722411],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp675_cam2_flcA_MM': np.array([[ 0.5       ,  0.42844758, -0.23046754, -0.11540099],
                                [-0.5       , -0.42844758,  0.23046754,  0.11540099],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]]),
                         'hwp675_cam2_flcB_MM': np.array([[ 0.5       , -0.40805704,  0.24619384,  0.15125492],
                                [-0.5       ,  0.40805704, -0.24619384, -0.15125492],
                                [ 0.        ,  0.        ,  0.        ,  0.        ],
                                [ 0.        ,  0.        ,  0.        ,  0.        ]])}

                    dict_ideal = {'m3_diat': 0,
                                  'm3_offset': 0,  # deg
                                  'hwp_offset': 0,  # was 0# deg
                                  'hwp_phi': 0.5,  # wave
                                  'imr_offset': 0,  # deg
                                  'imr_phi': 0.5,  # wave
                                  'optics_diat': 0,
                                  'optics_theta': 0,  # deg #
                                  'optics_phi': 0,  # wave
                                  'flc_theta': {"A": 0, "B": 45},  # 0, 45 before
                                  'flc_phi': 0.5,
                                  'dichroic_diat': 0,
                                  'dichroic_theta': 0,
                                  'dichroic_phi': 0}


                elif wv == 760:

                    dict_MMs = {'hwp0_cam1_flcA_MM': np.array([[0.57629301, 0.4436881 , 0.36660778, 0.02921032],
                           [0.57629301, 0.4436881 , 0.36660778, 0.02921032],
                           [0.        , 0.        , 0.        , 0.        ],
                           [0.        , 0.        , 0.        , 0.        ]]),
                               'hwp0_cam1_flcB_MM': np.array([[ 0.42370699, -0.36277495, -0.21710266, -0.02807809],
                           [ 0.42370699, -0.36277495, -0.21710266, -0.02807809],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp0_cam2_flcA_MM': np.array([[ 0.41830921, -0.06941308, -0.41229076,  0.01344442],
                           [-0.41830921,  0.06941308,  0.41229076, -0.01344442],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp0_cam2_flcB_MM': np.array([[ 0.58169079,  0.15032623,  0.56179587, -0.01231219],
                           [-0.58169079, -0.15032623, -0.56179587,  0.01231219],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp45_cam1_flcA_MM': np.array([[ 0.57629301, -0.4436881 , -0.36660778,  0.02921032],
                           [ 0.57629301, -0.4436881 , -0.36660778,  0.02921032],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp45_cam1_flcB_MM': np.array([[ 0.42370699,  0.36277495,  0.21710266, -0.02807809],
                           [ 0.42370699,  0.36277495,  0.21710266, -0.02807809],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp45_cam2_flcA_MM': np.array([[ 0.41830921,  0.06941308,  0.41229076,  0.01344442],
                           [-0.41830921, -0.06941308, -0.41229076, -0.01344442],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp45_cam2_flcB_MM': np.array([[ 0.58169079, -0.15032623, -0.56179587, -0.01231219],
                           [-0.58169079,  0.15032623,  0.56179587,  0.01231219],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp225_cam1_flcA_MM': np.array([[ 0.57629301,  0.36660778, -0.4436881 ,  0.02921032],
                           [ 0.57629301,  0.36660778, -0.4436881 ,  0.02921032],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp225_cam1_flcB_MM': np.array([[ 0.42370699, -0.21710266,  0.36277495, -0.02807809],
                           [ 0.42370699, -0.21710266,  0.36277495, -0.02807809],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp225_cam2_flcA_MM': np.array([[ 0.41830921, -0.41229076,  0.06941308,  0.01344442],
                           [-0.41830921,  0.41229076, -0.06941308, -0.01344442],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp225_cam2_flcB_MM': np.array([[ 0.58169079,  0.56179587, -0.15032623, -0.01231219],
                           [-0.58169079, -0.56179587,  0.15032623,  0.01231219],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp675_cam1_flcA_MM': np.array([[ 0.57629301, -0.36660778,  0.4436881 ,  0.02921032],
                           [ 0.57629301, -0.36660778,  0.4436881 ,  0.02921032],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp675_cam1_flcB_MM': np.array([[ 0.42370699,  0.21710266, -0.36277495, -0.02807809],
                           [ 0.42370699,  0.21710266, -0.36277495, -0.02807809],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp675_cam2_flcA_MM': np.array([[ 0.41830921,  0.41229076, -0.06941308,  0.01344442],
                           [-0.41830921, -0.41229076,  0.06941308, -0.01344442],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]]), 'hwp675_cam2_flcB_MM': np.array([[ 0.58169079, -0.56179587,  0.15032623, -0.01231219],
                           [-0.58169079,  0.56179587, -0.15032623,  0.01231219],
                           [ 0.        ,  0.        ,  0.        ,  0.        ],
                           [ 0.        ,  0.        ,  0.        ,  0.        ]])}



                    dict_ideal = {'m3_diat': 0,
                                  'm3_offset': 0,  # deg
                                  'hwp_offset': 0,  # was 0# deg
                                  'hwp_phi': 0.5,  # wave
                                  'imr_offset': 0,  # deg
                                  'imr_phi': 0.5,  # wave
                                  'optics_diat': 0,
                                  'optics_theta': 0,  # deg #
                                  'optics_phi': 0,  # wave
                                  'flc_theta': {"A": 0, "B": 45},  # 0, 45 before
                                  'flc_phi': 0.5,
                                  'dichroic_diat': 0,
                                  'dichroic_theta': 0,
                                  'dichroic_phi': 0}




            Is = []
            Qs = []

            Us = []
            phis = []
            beforeHWPI = []
            beforeHWPQ = []
            beforeHWPU = []
            beforeHWPphi = []

            stokes_cubeI = np.expand_dims(stH + stV, axis = 0)
            stokes_cubeQ = np.expand_dims(stH - stV, axis = 0)
            stokes_cubeU = np.expand_dims(stH45 - stV45, axis = 0)
            stokes_cubeV = np.zeros(np.shape(stokes_cubeU))
            stokes_image = np.concatenate((stokes_cubeI, stokes_cubeQ, stokes_cubeU, stokes_cubeV), axis = 0)







            vectorised_triple_diff_ideal = vmap(triple_diff_ideal, in_axes = (3, None, None,None, None, None, None, None, None, None), out_axes = (2,2,2))
            hwpI, hwpQ, hwpU = vectorised_triple_diff_ideal(stokes_image, pa, alt, az, imr, False, DRA, imrpad, dict_ideal, dict_MMs)


            hwpI = jnp.expand_dims(hwpI, axis = 0)
            hwpQ = jnp.expand_dims(hwpQ, axis = 0)
            hwpU = jnp.expand_dims(hwpU, axis = 0)
            hwpV = jnp.zeros(np.shape(hwpU))

            stokes_image = np.concatenate((hwpI, hwpQ, hwpU, hwpV), axis = 0)



            plt.figure(figsize = (10,4))
            plt.subplot(1,3,1)
            plt.title('IMR {:.2f}'.format(imr))
            plt.imshow(stokes_image[0,].sum(axis=2))
            plt.subplot(1,3,2)
            plt.title('PA {:.2f}'.format(pa))
            plt.imshow(stokes_image[1,].sum(axis=2))
            plt.subplot(1,3,3)
            plt.title('ALT {:.2f}, AZ {:.2f}'.format(alt, az))
            plt.imshow(stokes_image[2,].sum(axis=2))
            plt.savefig(pp_ + 'after_triple_diff_image_model.pdf')


            Hs = np.zeros(np.shape(stokes_image)[1:])
            Vs = np.zeros(np.shape(stokes_image)[1:])
            H45s = np.zeros(np.shape(stokes_image)[1:])
            V45s = np.zeros(np.shape(stokes_image)[1:])

            camera = 1
            is_ordinary = camera == 1

            def apply_final_stuff(current_frame):
                hwp0mm = wollaston(is_ordinary) @ waveplate(np.deg2rad(0), np.pi)
                hwp45mm = wollaston(is_ordinary) @ waveplate(np.deg2rad(45), np.pi)
                hwp225mm = wollaston(is_ordinary) @ waveplate(np.deg2rad(22.5), np.pi)
                hwp675mm = wollaston(is_ordinary) @ waveplate(np.deg2rad(67.5), np.pi)

                Hs  =   comp_higher_matrix_mult(hwp0mm, current_frame)[0,]
                Vs =   comp_higher_matrix_mult(hwp45mm, current_frame)[0,]
                H45s  = comp_higher_matrix_mult(hwp225mm, current_frame)[0,]
                V45s  = comp_higher_matrix_mult(hwp675mm, current_frame)[0,]

                return Hs, Vs, H45s, V45s




            vectorised_rotate_final = vmap(apply_final_stuff, in_axes = (3,), out_axes = (2,2,2,2))
            Hs, Vs, H45s, V45s = vectorised_rotate_final(stokes_image)

            Hs_new = Hs
            Vs_new = Vs
            H45s_new = H45s
            V45s_new = V45s

            if chem in ['mg0.80_fe0.20_olivine', 'mg0.70_fe0.30_olivine', 'mg0.60_fe0.40_olivine']:

                if chem == 'mg0.80_fe0.20_olivine':
                    chemt = 'mg0.8_fe0.2_olivine'
                elif chem ==  'mg0.70_fe0.30_olivine':
                    chemt =  'mg0.7_fe0.3_olivine'
                elif chem == 'mg0.60_fe0.40_olivine':
                    chemt = 'mg0.6_fe0.4_olivine'

                np.save(pp_+  'H_scat_m{}_r{}_w{}.npy'.format(chemt, grain, wv),   Hs_new)
                np.save(pp_+  'V_scat_m{}_r{}_w{}.npy'.format(chemt, grain, wv),   Vs_new)
                np.save(pp_ + 'H45_scat_m{}_r{}_w{}.npy'.format(chemt, grain, wv), H45s_new)
                np.save(pp_ + 'V45_scat_m{}_r{}_w{}.npy'.format(chemt, grain, wv), V45s_new)

            else:

                np.save(pp_+  'H_scat_m{}_r{}_w{}.npy'.format(chem, grain, wv),   Hs_new)
                np.save(pp_+  'V_scat_m{}_r{}_w{}.npy'.format(chem, grain, wv),   Vs_new)
                np.save(pp_ + 'H45_scat_m{}_r{}_w{}.npy'.format(chem, grain, wv), H45s_new)
                np.save(pp_ + 'V45_scat_m{}_r{}_w{}.npy'.format(chem, grain, wv), V45s_new)


            plt.figure()
            plt.subplot(1,2,1)
            plt.title('Stokes Q')
            plt.imshow(Hs_new.sum(axis=2) - Vs_new.sum(axis=2)) # says Q is H - V
            plt.subplot(1,2,2)
            plt.title('Stokes U')
            plt.imshow(H45s_new.sum(axis=2) - V45s_new.sum(axis=2)) # says U is H45 - V45
            plt.savefig(pp_ + 'after_mms_new_scat_fit.pdf')
