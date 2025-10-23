import jax
# jax.config.update("jax_debug_nans", True)
# jax.config.update("jax_enable_x64", True)
import matplotlib.pyplot as plt
from jax import numpy as np
import sys
from jax.tree_util import tree_map
from jax.scipy.ndimage import map_coordinates

plt.rcParams['image.origin'] = 'lower'
sys.path.append('../all_projects')

def nd_coords(
    npixels,
    pixel_scales = 1.0,
    offsets = 0.0,
    indexing = "xy", ):

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

def rotate_volume(volume, angle):
    """ Rotate a 3D volume around the z-axis using JAX."""
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))

    # Define rotation matrix around z-axis
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])

    # Compute new coordinates
    shape = volume.shape
    coords = np.indices(shape).reshape(3, -1)  # Flattened grid of coordinates
    new_coords = np.matmul(rotation_matrix, coords - np.array(shape)[:, None] / 2) + np.array(shape)[:, None] / 2

    # Interpolate the rotated image
    rotated_volume = map_coordinates(volume, new_coords, order=1, mode='nearest').reshape(shape)
    return rotated_volume





@jax.jit
def rotate_volume_jax(volume, angle):
    """Rotate a 3D volume around the z-axis using JAX."""
    cos_a = np.cos(np.radians(angle))
    sin_a = np.sin(np.radians(angle))

    # Rotation matrix for z-axis
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])

    # Get the volume shape
    shape = np.array(volume.shape)
    center = shape / 2

    # Generate grid coordinates
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    coords = np.stack([x, y, z], axis=0).reshape(3, -1)  # Flatten the grid

    # Apply rotation
    new_coords = rotation_matrix @ (coords - center[:, None]) + center[:, None]

    # Interpolation
    rotated_volume = map_coordinates(volume, new_coords, order=1, mode='nearest').reshape(shape)

    return rotated_volume





def rotateC(coords, phi, interpolator):

    rot_xs = coords[1,] * np.cos(np.deg2rad(phi)) - coords[0,] * np.sin(np.deg2rad(phi))
    rot_ys = coords[1,] * np.sin(np.deg2rad(phi)) + coords[0,] * np.cos(np.deg2rad(phi))

    rot_xs = np.expand_dims(rot_xs, axis=0)
    rot_ys = np.expand_dims(rot_ys, axis=0)

    rotated_coords = np.concatenate((rot_xs, rot_ys), axis=0)
    ypts, xpts = rotated_coords.reshape(2, -1)

    return interpolator(xpts, ypts).reshape((101, 101))





class geometric_star():
    def __init__(self, model_params):

        self.pc_to_AU = 206265
        self.model_params = model_params
        self.mas_to_rad = 4.8481368 * 10 ** (-9)
        self.image_size = model_params['image_size']
        self.pixel_ratio =  model_params['pixel_ratio']
        self.size_biggest_baseline_m = model_params['size_biggest_baseline_m']
        self.wavelength =  model_params['wavelength']
        self.dftm_grid = model_params['dftm_grid']

        self.u_coords = model_params['u_coords']
        self.v_coords = model_params['v_coords']

        self.zdata = model_params['zdata']
        self.xdata = model_params['xdata']
        self.indx_of_cp = model_params['indx_of_cp']

        self.H = model_params['H_scat']
        self.V = model_params['V_scat']
        self.H45 = model_params['H45_scat']
        self.V45 = model_params['V45_scat']

        self.ydata_real = model_params['ydata_real']
        self.ydata_real_err = model_params['ydata_real_err']
        self.star_radius = model_params['star_radius']
        self.dust_star_contrast = model_params['dust_star_contrast']


    def generalized_gaussian(self,x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, beta):
        exponent = -0.5 * ((x - mu_x) ** 2 / sigma_x ** 2 + (y - mu_y) ** 2 / sigma_y ** 2 + (
                z - mu_z) ** 2 / sigma_z ** 2) ** (1 / beta)
        return np.exp(exponent)

    def gaussian_shelll(self, x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, thickness, beta):
        """
        Compute a Gaussian shell with an inner radius based on the innermost dust measurement
        and thickness extending outwards only.

        Parameters:
            x, y, z (numpy.ndarray): Grids of coordinates.
            mu_x, mu_y, mu_z (float): Mean position of the shell (innermost point).
            sigma_x, sigma_y, sigma_z (float): Standard deviations in x, y, z.
            thickness (float): Thickness of the shell, extending outward only.
            beta (float): Shape parameter for the Gaussian.

        Returns:
            numpy.ndarray: 3D Gaussian shell.
        """
        # Generalized Gaussian core (innermost shell, centered around the mean)
        core = np.exp(-0.5 * (((x - mu_x) / sigma_x) ** 2 +
                              ((y - mu_y) / sigma_y) ** 2 +
                              ((z - mu_z) / sigma_z) ** 2) ** beta)

        # Outer Gaussian shell (only extending outward, beyond the inner radius)
        outer = np.exp(-0.5 * (((x - mu_x) / (sigma_x + thickness)) ** 2 +
                               ((y - mu_y) / (sigma_y + thickness)) ** 2 +
                               ((z - mu_z) / (sigma_z + thickness)) ** 2) ** beta)

        # The shell is the difference between the outer and core Gaussians
        shell = outer - core

        # Normalize the shell to ensure it sums to 1
        return shell / shell.sum()


    def gaussian_shell(self, x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, thickness, beta):
        """
        Compute a Gaussian shell where sigma_x, sigma_y, sigma_z define the middle of the Gaussian profile,
        and the thickness is a standard deviation on each side.

        Parameters:
            x, y, z (numpy.ndarray): Grids of coordinates.
            mu_x, mu_y, mu_z (float): Mean position of the shell (center).
            sigma_x, sigma_y, sigma_z (float): Position of the peak of the Gaussian.
            thickness (float): Standard deviation on each side of the peak.
            beta (float): Shape parameter for the Gaussian.

        Returns:
            numpy.ndarray: 3D Gaussian shell.
        """
        # Inner Gaussian boundary (lower edge of the shell)
        inner = np.exp(-0.5 * ((((x - mu_x) / (sigma_x - thickness)) ** 2 +
                                ((y - mu_y) / (sigma_y - thickness)) ** 2 +
                                ((z - mu_z) / (sigma_z - thickness)) ** 2) ** beta))

        # Outer Gaussian boundary (upper edge of the shell)
        outer = np.exp(-0.5 * ((((x - mu_x) / (sigma_x + thickness)) ** 2 +
                                ((y - mu_y) / (sigma_y + thickness)) ** 2 +
                                ((z - mu_z) / (sigma_z + thickness)) ** 2) ** beta))

        # The shell is the difference between the outer and inner boundaries
        shell = outer - inner

        # Normalize the shell to ensure it sums to 1
        return shell / shell.sum()


    def rotate_point(self,x, y, z, angle_degrees):
        angle_radians = np.deg2rad(angle_degrees)
        x_rotated = x * np.cos(angle_radians) - y * np.sin(angle_radians)
        y_rotated = x * np.sin(angle_radians) + y * np.cos(angle_radians)
        return x_rotated, y_rotated, z


    def make_3D_model(self):

        star_radius_pixel = (self.star_radius) / self.pixel_ratio   # [pixel]
        x, y, z = np.ogrid[-154:154, -154:154, -154:154]
        rad_distance = np.sqrt(x ** 2 + y ** 2 + z ** 2)

        half_size = int(np.shape(rad_distance)[0] / 2)
        rad_distance = rad_distance.at[half_size, half_size, half_size].set(rad_distance[half_size + 1, half_size + 1, half_size + 1])
        rad_profile = rad_distance.copy()
        rad_profile = (rad_profile) / (self.model_params['a'])
        powerlaw_part = self.model_params['dust_pl_const'] * (rad_profile ** (-self.model_params['dust_pl_exp']))
        self.rad_profile = powerlaw_part

        sigma = star_radius_pixel
        star = np.zeros((self.image_size, self.image_size, self.image_size))
        star = 1 * (1 - (rad_profile / sigma) ** 2)
        star = np.where(rad_distance <= sigma, star, 0)
        star = star.sum(axis=2)
        star = star / star.sum()

        self.theta = np.arctan2(y,x)
        self.phi = np.arccos(z / np.sqrt(x ** 2 + y ** 2 + z ** 2))

        self.star = star
        shell = np.zeros((self.image_size, self.image_size, self.image_size))

        def _nn(a):
            # Replace NaN/Inf with finite values
            return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        def _safe_norm_sum(a, axis=None):
            a2 = _nn(a)
            s = np.sum(a2, axis=axis)
            s = np.where(s == 0.0, 1.0, s)
            return a2 / s

        def _safe_div_max(a):
            a2 = _nn(a)
            m = np.max(a2)
            m = np.where(m == 0.0, 1.0, m)
            return a2 / m

        if self.model_params['dust_shape'] == 'ellipse_enhance_and_blob':
            # Scale parameters to pixel units
            a_pixel = self.model_params['a'] / self.model_params['pixel_ratio']
            b_pixel = self.model_params['b'] / self.model_params['pixel_ratio']
            c_pixel = self.model_params['c'] / self.model_params['pixel_ratio']
            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x, mu_y, mu_z = h_pixel, k_pixel, n_pixel
            sigma_x, sigma_y, sigma_z = a_pixel, b_pixel, c_pixel
            beta = 0.01

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            # Base ellipse as generalized Gaussian "shell"
            zt = 1.0 - self.generalized_gaussian(
                rotated_x, rotated_y, rotated_z,
                mu_x, mu_y, mu_z,
                sigma_x, sigma_y, sigma_z,
                beta
            )
            zt = _nn(zt)

            # ------------- Spherical enhancement (xy-plane) -------------
            amp = self.model_params['enhancement_amp']
            sigma = np.deg2rad(self.model_params['enhancement_size'])
            psi = np.deg2rad(self.model_params['enhancement_loc'])  # azimuth in xy plane (like alpha)

            # Unit direction for the enhancement center in xy-plane
            n0x, n0y, n0z = np.cos(psi), np.sin(psi), 0.0

            # Unit direction vectors from ellipse center
            dx, dy, dz = rotated_x - mu_x, rotated_y - mu_y, rotated_z - mu_z
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            eps = np.finfo(dx.dtype).eps
            nx, ny, nz = dx / (r + eps), dy / (r + eps), dz / (r + eps)

            cos_gamma = np.clip(nx * n0x + ny * n0y + nz * n0z, -1.0, 1.0)
            gamma = np.arccos(cos_gamma)

            # Guard sigma == 0 (degenerate width) → treat as no enhancement
            use_flat = np.isclose(sigma, 0.0)
            # Avoid /0 inside exp argument when use_flat=True
            denom = np.where(use_flat, 1.0, sigma)
            gauss_arg = -0.5 * (gamma / denom) ** 2
            spherical_enhancement = 1.0 + amp * np.where(use_flat, 0.0, np.exp(gauss_arg))
            spherical_enhancement = _nn(spherical_enhancement)

            # Apply enhancement multiplicatively to the ellipse
            zt_enh = _nn(zt * spherical_enhancement)

            # Multiply by radial profile and normalise safely
            rad_prof = _nn(self.rad_profile)
            zt_enh = _nn(zt_enh * rad_prof)
            shell = shell + _safe_norm_sum(zt_enh)

            # ----------------- Blob component (unchanged math, but safe) -----------------
            ro = self.model_params['blob_radial_distance']
            thetao = self.model_params['theta_blob']
            phio = self.model_params['phi_blob']

            xo = ro * np.sin(thetao) * np.cos(phio)
            yo = ro * np.sin(thetao) * np.sin(phio)
            zo = ro * np.cos(thetao)

            basis_sd = self.model_params['r_blob']
            sigmax = sigmay = sigmaz = basis_sd
            beta = 1.0

            def_gauss = np.exp(-(
                    (x - xo) ** 2 / (2 * sigmax ** 2) +
                    (y - yo) ** 2 / (2 * sigmay ** 2) +
                    (z - zo) ** 2 / (2 * sigmaz ** 2)
            )) ** (1.0 / beta)
            def_gauss = _nn(def_gauss)

            # If sum is zero, return zeros; else, normalise
            pred = np.isclose(np.sum(_nn(def_gauss)), 0.0)
            true_fn = lambda arr: np.zeros_like(arr)
            false_fn = lambda arr: _safe_norm_sum(arr)
            cube_gauss = jax.lax.cond(pred, true_fn, false_fn, def_gauss)

            store_blob = cube_gauss
            blob = self.model_params['blob_contrast'] * cube_gauss

            shell = _nn(shell + blob)
            shell = _safe_norm_sum(shell)

            # ----------------- Diagnostics / plots (safe) -----------------
            self.shell_plot = _safe_div_max(_nn(z)) + _safe_div_max(_nn(store_blob))
            self.dust_shell = shell

        if self.model_params['dust_shape'] == 'ellipse_enhance':
            # Scale parameters to pixel units
            a_pixel = self.model_params['a'] / self.model_params['pixel_ratio']
            b_pixel = self.model_params['b'] / self.model_params['pixel_ratio']
            c_pixel = self.model_params['c'] / self.model_params['pixel_ratio']
            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x, mu_y, mu_z = h_pixel, k_pixel, n_pixel
            sigma_x, sigma_y, sigma_z = a_pixel, b_pixel, c_pixel
            beta = 0.01

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            # Base ellipse as generalized Gaussian "shell"
            zt = 1.0 - self.generalized_gaussian(
                rotated_x, rotated_y, rotated_z,
                mu_x, mu_y, mu_z,
                sigma_x, sigma_y, sigma_z,
                beta
            )
            zt = _nn(zt)

            # ------------- Spherical enhancement (xy-plane) -------------
            amp = self.model_params['enhancement_amp']
            sigma = np.deg2rad(self.model_params['enhancement_size'])
            psi = np.deg2rad(self.model_params['enhancement_loc'])  # azimuth in xy plane (like alpha)

            # Unit direction for the enhancement center in xy-plane
            n0x, n0y, n0z = np.cos(psi), np.sin(psi), 0.0

            # Unit direction vectors from ellipse center
            dx, dy, dz = rotated_x - mu_x, rotated_y - mu_y, rotated_z - mu_z
            r = np.sqrt(dx * dx + dy * dy + dz * dz)
            eps = np.finfo(dx.dtype).eps
            nx, ny, nz = dx / (r + eps), dy / (r + eps), dz / (r + eps)

            cos_gamma = np.clip(nx * n0x + ny * n0y + nz * n0z, -1.0, 1.0)
            gamma = np.arccos(cos_gamma)

            # Guard sigma == 0 (degenerate width) → treat as no enhancement
            use_flat = np.isclose(sigma, 0.0)
            # Avoid /0 inside exp argument when use_flat=True
            denom = np.where(use_flat, 1.0, sigma)
            gauss_arg = -0.5 * (gamma / denom) ** 2
            spherical_enhancement = 1.0 + amp * np.where(use_flat, 0.0, np.exp(gauss_arg))
            spherical_enhancement = _nn(spherical_enhancement)

            # Apply enhancement multiplicatively to the ellipse
            zt_enh = _nn(zt * spherical_enhancement)

            # Multiply by radial profile and normalise safely
            rad_prof = _nn(self.rad_profile)
            zt_enh = _nn(zt_enh * rad_prof)
            shell = shell + _safe_norm_sum(zt_enh)

            shell = _safe_norm_sum(shell)

            # ----------------- Diagnostics / plots (safe) -----------------
            self.shell_plot = _safe_div_max(_nn(z))
            self.dust_shell = shell

        if self.model_params['dust_shape'] == 'ellipse':

            a_pixel = self.model_params['a'] / self.model_params['pixel_ratio']
            b_pixel = self.model_params['b'] / self.model_params['pixel_ratio']
            c_pixel = self.model_params['c'] / self.model_params['pixel_ratio']

            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']


            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel
            sigma_x = a_pixel
            sigma_y = b_pixel
            sigma_z = c_pixel

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)


            beta = 0.01
            z = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, beta)
            z = z * self.rad_profile


            self.shell_plot = z.sum(axis=2)
            shell = z/z.sum()

        # if self.model_params['dust_shape'] == 'ellipse':
        #     a_pixel = self.model_params['a'] / self.model_params['pixel_ratio']
        #     b_pixel = self.model_params['b'] / self.model_params['pixel_ratio']
        #     c_pixel = self.model_params['c'] / self.model_params['pixel_ratio']
        #
        #     h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
        #     k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
        #     n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']
        #
        #     # Shifted and rotated grid
        #     rotation_angle = self.model_params['alpha']
        #     rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)
        #
        #     # Translate
        #     x_rel = rotated_x - h_pixel
        #     y_rel = rotated_y - k_pixel
        #     z_rel = rotated_z - n_pixel
        #
        #     # Ellipsoidal radius
        #     ellipsoid_r = (x_rel / a_pixel) ** 2 + (y_rel / b_pixel) ** 2 + (z_rel / c_pixel) ** 2
        #
        #     # Solid shell: 1 outside inner ellipsoid (r >= 1), 0 inside (r < 1)
        #     shell_mask = np.where(ellipsoid_r >= 1.0, 1.0, 0.0)
        #
        #     # Multiply by power-law radial profile
        #     z = shell_mask * self.rad_profile
        #
        #     self.shell_plot = np.sum(z, axis=2)
        #     shell = z / np.sum(z)




        if self.model_params['dust_shape'] == 'two_pl_exp':


            a_pixel = self.model_params['a'] / self.model_params['pixel_ratio']
            b_pixel = self.model_params['b'] / self.model_params['pixel_ratio']
            c_pixel = self.model_params['c'] / self.model_params['pixel_ratio']

            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']


            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel
            sigma_x = a_pixel
            sigma_y = b_pixel
            sigma_z = c_pixel

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            unit_x = rotated_x / rad_distance
            unit_y = rotated_y / rad_distance
            unit_z = rotated_z / rad_distance

            ellipsoid_radius = np.sqrt( 1 / (unit_x ** 2 / sigma_x ** 2 + unit_y ** 2 / sigma_y ** 2 + unit_z ** 2 / sigma_z ** 2))
            ellipsoid_radius = ellipsoid_radius.at[half_size, half_size, half_size].set(
                ellipsoid_radius[half_size + 1, half_size + 1, half_size + 1])
            rad_p = rad_distance - ellipsoid_radius
            rad_p = np.where(rad_p <= 0, 0, rad_p)  # Ensure no negatives

            rad_pp = rad_distance.copy()
            rad_profile_copy = ((rad_pp) ** (- self.model_params['dust_pl_exp']))

            z1 = rad_p * rad_profile_copy

            z1 = z1/z1.sum()

            a_pixel = self.model_params['a2'] / self.model_params['pixel_ratio']
            b_pixel = self.model_params['b2'] / self.model_params['pixel_ratio']
            c_pixel = self.model_params['c2'] / self.model_params['pixel_ratio']

            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']


            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel
            sigma_x = a_pixel
            sigma_y = b_pixel
            sigma_z = c_pixel

            rotation_angle = self.model_params['alpha2']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            unit_x = rotated_x / rad_distance
            unit_y = rotated_y / rad_distance
            unit_z = rotated_z / rad_distance

            ellipsoid_radius = np.sqrt( 1 / (unit_x ** 2 / sigma_x ** 2 + unit_y ** 2 / sigma_y ** 2 + unit_z ** 2 / sigma_z ** 2))
            ellipsoid_radius = ellipsoid_radius.at[half_size, half_size, half_size].set(
                ellipsoid_radius[half_size + 1, half_size + 1, half_size + 1])
            rad_p = rad_distance - ellipsoid_radius
            rad_p = np.where(rad_p <= 0, 0, rad_p)  # Ensure no negatives

            rad_pp = rad_distance.copy()
            rad_profile_copy = ((rad_pp) ** (- self.model_params['dust_pl_exp2']))

            z2 = rad_p * rad_profile_copy
            z2 = z2/z2.sum()

            z = z1 + z2*self.model_params['dust_c']
            shell = z/z.sum()
            self.shell_plot = shell.sum(axis=2)



        if self.model_params['dust_shape'] == 'thick_ellipse':


            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel

            self.model_params['thickb'] = self.model_params['thicka']
            self.model_params['thickc'] = self.model_params['thicka']

            sigma_x = self.model_params['a'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c'] / self.model_params['pixel_ratio']

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            beta = 0.01 # 0.1
            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x+ self.model_params['thicka'], sigma_y+ self.model_params['thickb'], sigma_z+ self.model_params['thickb'], beta)
            # z = z * self.rad_profile
            z_1 = z1 * z2
            z = z_1/z_1.sum()

            self.shell_plot = z.sum(axis=2)
            shell = shell + (z/z.sum())

            shell = np.where(np.isnan(shell), 0, shell)


        if self.model_params['dust_shape'] == 'two_thick_circles':

            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel
            #
            # self.model_params['thick1'] = 5
            # self.model_params['thick2'] = 5

            sigma_x = self.model_params['a'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c'] / self.model_params['pixel_ratio']

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            beta = 0.01
            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x+ self.model_params['thick1'], sigma_y+ self.model_params['thick1'], sigma_z+ self.model_params['thick1'], beta)
            # z = z * self.rad_profile
            z_1 = z1 * z2
            z_1 = z_1/z_1.sum()


            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel

            sigma_x = self.model_params['a2'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b2'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c2'] / self.model_params['pixel_ratio']

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)


            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x+ self.model_params['thick2'], sigma_y+ self.model_params['thick2'], sigma_z+ self.model_params['thick2'], beta)
            # z = z * self.rad_profile
            z_2 = z1 * z2
            z_2 = z_2/z_2.sum()

            z = z_1 + z_2 * self.model_params['ellipse_contrast_ratio']
            z = z/z.sum()

            self.shell_plot = z.sum(axis=2)
            shell = shell + (z/z.sum())
            shell = np.where(np.isnan(shell), 0, shell)


        if self.model_params['dust_shape'] == 'pl_ellipse':

            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel

            sigma_x = self.model_params['a'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c'] / self.model_params['pixel_ratio']

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            beta = 0.01
            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y,
                                               sigma_z, beta)

            z = z1 * self.rad_profile
            z = z / z.sum()

            self.shell_plot = z.sum(axis=2)
            shell = shell + (z / z.sum())

            shell = np.where(np.isnan(shell), 0, shell)



        if self.model_params['dust_shape'] == 'ellipse_blob_bright':


            a_pixel = self.model_params['a'] / self.model_params['pixel_ratio']
            b_pixel = self.model_params['b'] / self.model_params['pixel_ratio']
            c_pixel = self.model_params['c'] / self.model_params['pixel_ratio']
            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']
            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel
            sigma_x = a_pixel
            sigma_y = b_pixel
            sigma_z = c_pixel
            beta = 0.01  # Shape parameter
            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)


            zt = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y,
                                          sigma_z, beta)



            zt2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x + self.model_params['thicka'], sigma_y + self.model_params['thicka'], sigma_z + self.model_params['thicka'], beta)

            zt = zt*zt2

            shell = shell + (zt / zt.sum())

            ro = self.model_params['blob_radial_distance']
            thetao = self.model_params['theta_blob']
            phio = self.model_params['phi_blob']

            xo = ro * np.sin(thetao) * np.cos(phio)
            yo = ro * np.sin(thetao) * np.sin(phio)
            zo = ro * np.cos(thetao)
            basis_sd = self.model_params['r_blob']

            sigmax = basis_sd
            sigmay = basis_sd
            sigmaz = basis_sd

            beta = 1#0

            def_gauss = np.exp(-(
                        (x - xo) ** 2 / (2 * sigmax ** 2) + (y - yo) ** 2 / (2 * sigmay ** 2) + (z - zo) ** 2 / (
                            2 * sigmaz ** 2))) ** (1 / beta)

            cond = lambda x: np.sum(x) == 0
            cond = lambda x: np.allclose(np.sum(x), 0)

            true_fn = lambda x: np.zeros(x.shape)
            false_fn = lambda x: x / x.sum()

            cube_gauss = jax.lax.cond(
                cond(def_gauss),
                true_fn,
                false_fn,
                def_gauss,
            )

            store_blob = cube_gauss

            blob = self.model_params['blob_contrast'] * cube_gauss
            shell = shell + blob
            shell = shell / shell.sum()

            self.shell_plot = z / z.max() + store_blob / store_blob.max()
            self.dust_shell = shell



        if self.model_params['dust_shape'] == 'thick_circle_blob':


            a_pixel = self.model_params['a'] / self.model_params['pixel_ratio']
            b_pixel = self.model_params['b'] / self.model_params['pixel_ratio']
            c_pixel = self.model_params['c'] / self.model_params['pixel_ratio']
            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']
            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel
            sigma_x = a_pixel
            sigma_y = b_pixel
            sigma_z = c_pixel
            beta = 0.01  # Shape parameter
            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)


            zt = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y,
                                          sigma_z, beta)

            zt2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x + self.model_params['thicka'], sigma_y + self.model_params['thicka'], sigma_z + self.model_params['thicka'], beta)

            zt = zt*zt2
            shell = shell + (zt / zt.sum())

            ro = self.model_params['blob_radial_distance']
            thetao = self.model_params['theta_blob']
            phio = self.model_params['phi_blob']

            xo = ro * np.sin(thetao) * np.cos(phio)
            yo = ro * np.sin(thetao) * np.sin(phio)
            zo = ro * np.cos(thetao)
            basis_sd = self.model_params['r_blob']

            sigmax = basis_sd
            sigmay = basis_sd
            sigmaz = basis_sd

            beta = 1#0

            def_gauss = np.exp(-(
                        (x - xo) ** 2 / (2 * sigmax ** 2) + (y - yo) ** 2 / (2 * sigmay ** 2) + (z - zo) ** 2 / (
                            2 * sigmaz ** 2))) ** (1 / beta)

            cond = lambda x: np.sum(x) == 0
            cond = lambda x: np.allclose(np.sum(x), 0)

            true_fn = lambda x: np.zeros(x.shape)
            false_fn = lambda x: x / x.sum()

            cube_gauss = jax.lax.cond(
                cond(def_gauss),
                true_fn,
                false_fn,
                def_gauss,
            )

            store_blob = cube_gauss

            blob = self.model_params['blob_contrast'] * cube_gauss
            shell = shell + blob
            shell = shell / shell.sum()

            self.shell_plot = z / z.max() + store_blob / store_blob.max()
            self.dust_shell = shell




        if self.model_params['dust_shape'] == 'ellipse_and_blob':


            a_pixel = self.model_params['a'] / self.model_params['pixel_ratio']
            b_pixel = self.model_params['b'] / self.model_params['pixel_ratio']
            c_pixel = self.model_params['c'] / self.model_params['pixel_ratio']
            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']
            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel
            sigma_x = a_pixel
            sigma_y = b_pixel
            sigma_z = c_pixel
            beta = 0.01  # Shape parameter
            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)
            zt = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y,
                                          sigma_z, beta)

            zt = zt * self.rad_profile
            shell = shell + (zt / zt.sum())

            ro = self.model_params['blob_radial_distance']
            thetao = self.model_params['theta_blob']
            phio = self.model_params['phi_blob']

            xo = ro * np.sin(thetao) * np.cos(phio)
            yo = ro * np.sin(thetao) * np.sin(phio)
            zo = ro * np.cos(thetao)
            basis_sd = self.model_params['r_blob']

            sigmax = basis_sd
            sigmay = basis_sd
            sigmaz = basis_sd

            beta = 1#0

            def_gauss = np.exp(-(
                        (x - xo) ** 2 / (2 * sigmax ** 2) + (y - yo) ** 2 / (2 * sigmay ** 2) + (z - zo) ** 2 / (
                            2 * sigmaz ** 2))) ** (1 / beta)

            cond = lambda x: np.sum(x) == 0
            cond = lambda x: np.allclose(np.sum(x), 0)

            true_fn = lambda x: np.zeros(x.shape)
            false_fn = lambda x: x / x.sum()

            cube_gauss = jax.lax.cond(
                cond(def_gauss),
                true_fn,
                false_fn,
                def_gauss,
            )

            store_blob = cube_gauss

            blob = self.model_params['blob_contrast'] * cube_gauss
            shell = shell + blob
            shell = shell / shell.sum()

            self.shell_plot = z / z.max() + store_blob / store_blob.max()
            self.dust_shell = shell

        if self.model_params['dust_shape'] == 'ellipse_and_twoblob':

            a_pixel = self.model_params['a'] / self.model_params['pixel_ratio']
            b_pixel = self.model_params['b'] / self.model_params['pixel_ratio']
            c_pixel = self.model_params['c'] / self.model_params['pixel_ratio']
            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']
            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel
            sigma_x = a_pixel
            sigma_y = b_pixel
            sigma_z = c_pixel
            beta = 0.01  # Shape parameter
            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)
            zt = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y,
                                               sigma_z, beta)

            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x + 5, sigma_y  + 5, sigma_z  + 5, beta)
            zt = zt * z2
            shell = shell + (zt / zt.sum())

            ro = self.model_params['blob_radial_distance']
            thetao = self.model_params['theta_blob']
            phio = self.model_params['phi_blob']

            xo = ro * np.sin(thetao) * np.cos(phio)
            yo = ro * np.sin(thetao) * np.sin(phio)
            zo = ro * np.cos(thetao)
            basis_sd = self.model_params['r_blob']

            sigmax = basis_sd
            sigmay = basis_sd
            sigmaz = basis_sd

            beta = 1

            def_gauss = np.exp(-(
                    (x - xo) ** 2 / (2 * sigmax ** 2) + (y - yo) ** 2 / (2 * sigmay ** 2) + (z - zo) ** 2 / (
                    2 * sigmaz ** 2))) ** (1 / beta)

            cond = lambda x: np.sum(x) == 0
            cond = lambda x: np.allclose(np.sum(x), 0)

            true_fn = lambda x: np.zeros(x.shape)
            false_fn = lambda x: x / x.sum()

            cube_gauss = jax.lax.cond(
                cond(def_gauss),
                true_fn,
                false_fn,
                def_gauss,
            )

            store_blob = cube_gauss

            blob = self.model_params['blob_contrast'] * cube_gauss
            shell = shell + blob

            ro = self.model_params['blob_radial_distance2']
            thetao = self.model_params['theta_blob2']
            phio = self.model_params['phi_blob2']

            xo = ro * np.sin(thetao) * np.cos(phio)
            yo = ro * np.sin(thetao) * np.sin(phio)
            zo = ro * np.cos(thetao)

            basis_sd = self.model_params['r_blob2']

            sigmax = basis_sd
            sigmay = basis_sd
            sigmaz = basis_sd

            beta = 1

            def_gauss = np.exp(-(
                    (x - xo) ** 2 / (2 * sigmax ** 2) + (y - yo) ** 2 / (2 * sigmay ** 2) + (z - zo) ** 2 / (
                    2 * sigmaz ** 2))) ** (1 / beta)

            cond = lambda x: np.sum(x) == 0
            cond = lambda x: np.allclose(np.sum(x), 0)

            true_fn = lambda x: np.zeros(x.shape)
            false_fn = lambda x: x / x.sum()

            cube_gauss = jax.lax.cond(
                cond(def_gauss),
                true_fn,
                false_fn,
                def_gauss,
            )

            store_blob = cube_gauss

            blob = self.model_params['blob_contrast2'] * cube_gauss
            shell = shell + blob

            shell = shell / shell.sum()

            self.shell = shell
            self.shell_plot = z / z.max() + store_blob / store_blob.max()
            self.dust_shell = shell


        if self.model_params['dust_shape'] == 'Ntwo_thick_circles':


            mu_x = self.model_params['h'] / self.model_params['pixel_ratio']
            mu_y = self.model_params['k'] / self.model_params['pixel_ratio']
            mu_z = self.model_params['n'] / self.model_params['pixel_ratio']

            sigma_x = self.model_params['a'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c'] / self.model_params['pixel_ratio']



            thickness1 = self.model_params['thick1']
            beta = 0.01

            # First shell computation
            shell1 = self.gaussian_shell(x, y, z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, thickness1, beta)

            # Parameters for the second shell
            sigma_x2 = self.model_params['a2'] / self.model_params['pixel_ratio']
            sigma_y2 = self.model_params['a2'] / self.model_params['pixel_ratio']
            sigma_z2 = self.model_params['a2'] / self.model_params['pixel_ratio']

            thickness2 = self.model_params['thick2']

            # Second shell computation
            shell2 = self.gaussian_shell(x, y, z, mu_x, mu_y, mu_z, sigma_x2, sigma_y2, sigma_z2, thickness2, beta)

            # Combine shells with contrast adjustment
            z = shell1 + shell2 * self.model_params['ellipse_contrast_ratio']

            # Normalize and finalize
            self.shell_plot = z.sum(axis=2)
            shell = shell + (z / z.sum())
            shell = np.where(np.isnan(shell), 0, shell)



        if self.model_params['dust_shape'] == 'two_thin_circles':

            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel

            self.model_params['thicka'] = 2
            self.model_params['thickb'] = 2
            self.model_params['thickc'] = 2


            sigma_x = self.model_params['a'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c'] / self.model_params['pixel_ratio']

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            beta = 0.01 # 0.1
            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x+ self.model_params['thicka'], sigma_y+ self.model_params['thickb'], sigma_z+ self.model_params['thickb'], beta)

            z_1 = z1 * z2
            z_1 = z_1/z_1.sum()

            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel


            sigma_x = self.model_params['a2'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b2'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c2'] / self.model_params['pixel_ratio']

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            beta = 0.01 # 0.1
            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x+ self.model_params['thicka'], sigma_y+ self.model_params['thickb'], sigma_z+ self.model_params['thickb'], beta)
            z_2 = z1 * z2
            z_2 = z_2/z_2.sum()

            z = z_1 + z_2*self.model_params['ellipse_contrast_ratio']
            z = z / z.sum()


            self.shell_plot = z.sum(axis=2)
            shell = shell + (z/z.sum())
            shell = np.where(np.isnan(shell), 0, shell)


        if self.model_params['dust_shape'] == 'two_thin_circles_and_blob':

            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel

            self.model_params['thicka'] = 3
            self.model_params['thickb'] = 3
            self.model_params['thickc'] = 3

            sigma_x = self.model_params['a'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c'] / self.model_params['pixel_ratio']

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            beta = 0.01  # 0.1
            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y,
                                               sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z,
                                           sigma_x + self.model_params['thicka'], sigma_y + self.model_params['thickb'],
                                           sigma_z + self.model_params['thickb'], beta)

            z_1 = z1 * z2
            z_1 = z_1 / z_1.sum()

            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel

            sigma_x = self.model_params['a2'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b2'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c2'] / self.model_params['pixel_ratio']

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            beta = 0.01  # 0.1
            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y,
                                               sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z,
                                           sigma_x + self.model_params['thicka'], sigma_y + self.model_params['thickb'],
                                           sigma_z + self.model_params['thickb'], beta)
            z_2 = z1 * z2
            z_2 = z_2 / z_2.sum()

            z = z_1 + z_2 * self.model_params['ellipse_contrast_ratio']

            z_ellipses = z / z.sum()


            #$$$$$

            ro = self.model_params['blob_radial_distance']
            thetao = self.model_params['theta_blob']
            phio = self.model_params['phi_blob']

            xo = ro * np.sin(thetao) * np.cos(phio)
            yo = ro * np.sin(thetao) * np.sin(phio)
            zo = ro * np.cos(thetao)

            basis_sd = self.model_params['r_blob']

            sigmax = basis_sd
            sigmay = basis_sd
            sigmaz = basis_sd

            beta = 1

            def_gauss = np.exp(-(
                    (x - xo) ** 2 / (2 * sigmax ** 2) + (y - yo) ** 2 / (2 * sigmay ** 2) + (z - zo) ** 2 / (
                    2 * sigmaz ** 2))) ** (1 / beta)

            cond = lambda x: np.sum(x) == 0
            cond = lambda x: np.allclose(np.sum(x), 0)

            true_fn = lambda x: np.zeros(x.shape)
            false_fn = lambda x: x / x.sum()

            cube_gauss = jax.lax.cond(
                cond(def_gauss),
                true_fn,
                false_fn,
                def_gauss,
            )


            cube_gauss = cube_gauss/cube_gauss.sum()
            blob = self.model_params['blob_contrast'] * cube_gauss

            shell = z_ellipses + blob

            z = shell/shell.sum()
            shell = z



            self.shell_plot = z.sum(axis=2)

            shell = np.where(np.isnan(shell), 0, shell)





        if self.model_params['dust_shape'] == 'three_thin_circles':            #'three_thin_circles'

            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel

            self.model_params['thicka'] = 2
            self.model_params['thickb'] = 2
            self.model_params['thickc'] = 2

            sigma_x = self.model_params['a'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c'] / self.model_params['pixel_ratio']

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            beta = 0.01  # 0.1
            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y,
                                               sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z,
                                           sigma_x + self.model_params['thicka'],
                                           sigma_y + self.model_params['thickb'],
                                           sigma_z + self.model_params['thickb'], beta)

            z_1 = z1 * z2
            z_1 = z_1 / z_1.sum()



            sigma_x = self.model_params['a2'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b2'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c2'] / self.model_params['pixel_ratio']



            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y,
                                               sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z,
                                           sigma_x + self.model_params['thicka'],
                                           sigma_y + self.model_params['thickb'],
                                           sigma_z + self.model_params['thickb'], beta)
            z_2 = z1 * z2
            z_2 = z_2 / z_2.sum()




            sigma_x = self.model_params['a3'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b3'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c3'] / self.model_params['pixel_ratio']



            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y,
                                               sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z,
                                           sigma_x + self.model_params['thicka'],
                                           sigma_y + self.model_params['thickb'],
                                           sigma_z + self.model_params['thickb'], beta)
            z_3 = z1 * z2
            z_3 = z_3 / z_3.sum()

            z = z_1 + z_2 * self.model_params['ellipse_contrast_ratio'] + z_3 * self.model_params['ellipse_contrast_ratio2']
            z = z / z.sum()

            self.shell_plot = z.sum(axis=2)
            shell = shell + (z / z.sum())
            shell = np.where(np.isnan(shell), 0, shell)


        if self.model_params['dust_shape'] == 'one_thin_one_thick_circle':


            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel

            self.model_params['thicka'] = 2
            self.model_params['thickb'] = 2
            self.model_params['thickc'] = 2


            sigma_x = self.model_params['a'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c'] / self.model_params['pixel_ratio']

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            beta = 0.01 # 0.1
            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x+ self.model_params['thicka'], sigma_y+ self.model_params['thickb'], sigma_z+ self.model_params['thickb'], beta)

            z_1 = z1 * z2
            z_1 = z_1/z_1.sum()

            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel


            sigma_x = self.model_params['a2'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b2'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c2'] / self.model_params['pixel_ratio']

            self.model_params['thickb2'] = self.model_params['thicka2']
            self.model_params['thickc2'] = self.model_params['thicka2']


            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            beta = 0.01 # 0.1
            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x+ self.model_params['thicka2'], sigma_y+ self.model_params['thickb2'], sigma_z+ self.model_params['thickb2'], beta)
            z_2 = z1 * z2
            z_2 = z_2/z_2.sum()

            z = z_1 + z_2*self.model_params['ellipse_contrast_ratio']
            z = z / z.sum()



            self.shell_plot = z.sum(axis=2)
            shell = shell + (z/z.sum())
            shell = np.where(np.isnan(shell), 0, shell)


        if self.model_params['dust_shape'] == 'ellipse_thin':

            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel

            self.model_params['thicka'] = 2
            self.model_params['thickb'] = 2
            self.model_params['thickc'] = 2

            sigma_x = self.model_params['a'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c'] / self.model_params['pixel_ratio']

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            beta = 0.01 # 0.1
            z1 = 1 - self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x, sigma_y, sigma_z, beta)
            z2 = self.generalized_gaussian(rotated_x, rotated_y, rotated_z, mu_x, mu_y, mu_z, sigma_x+ self.model_params['thicka'], sigma_y+ self.model_params['thickb'], sigma_z+ self.model_params['thickb'], beta)
            z_1 = z1 * z2
            z = z_1/z_1.sum()

            self.shell_plot = z.sum(axis=2)
            shell = shell + (z/z.sum())

            shell = np.where(np.isnan(shell), 0, shell)


        if self.model_params['dust_shape'] == 'bright_spot_ellipse':
            h_pixel = self.model_params['h'] / self.model_params['pixel_ratio']
            k_pixel = self.model_params['k'] / self.model_params['pixel_ratio']
            n_pixel = self.model_params['n'] / self.model_params['pixel_ratio']

            mu_x = h_pixel
            mu_y = k_pixel
            mu_z = n_pixel

            self.model_params['thicka'] = 2
            self.model_params['thickb'] = 2
            self.model_params['thickc'] = 2

            sigma_x = self.model_params['a'] / self.model_params['pixel_ratio']
            sigma_y = self.model_params['b'] / self.model_params['pixel_ratio']
            sigma_z = self.model_params['c'] / self.model_params['pixel_ratio']

            rotation_angle = self.model_params['alpha']
            rotated_x, rotated_y, rotated_z = self.rotate_point(x, y, z, rotation_angle)

            beta = 0.01
            z1 = 1 - self.generalized_gaussian(
                rotated_x, rotated_y, rotated_z,
                mu_x, mu_y, mu_z,
                sigma_x, sigma_y, sigma_z,
                beta
            )
            z_1 = z1 * self.rad_profile

            def phase_wrap(theta):
                return np.mod(theta + np.pi, 2 * np.pi) - np.pi

            def mask_cone(theta, phi, theta_bright, phi_bright, phi_width, mag):
                v_x = np.sin(phi) * np.cos(theta)
                v_y = np.sin(phi) * np.sin(theta)
                v_z = np.cos(phi)

                v_bright_x = np.sin(phi_bright) * np.cos(theta_bright)
                v_bright_y = np.sin(phi_bright) * np.sin(theta_bright)
                v_bright_z = np.cos(phi_bright)

                cos_d = v_x * v_bright_x + v_y * v_bright_y + v_z * v_bright_z
                cos_d = np.clip(cos_d, -1.0, 1.0)
                angular_distance = np.arccos(cos_d)

                mask = angular_distance <= phi_width
                return np.where(mask, mag, 1.0)

            theta_bright = phase_wrap(self.model_params['bright_location_theta'])
            phi_bright = self.model_params['bright_location_phi']
            phi_width = self.model_params['bright_size']
            mag = self.model_params['bright_mag']

            masked_brightness = mask_cone(self.theta, self.phi, theta_bright, phi_bright, phi_width, mag)

            z = z_1 / np.sum(z_1)
            z = z * masked_brightness
            z = z / np.sum(z)
            shell = np.where(np.isnan(z), 0.0, z)

        if self.model_params['dust_shape'] == 1:

            shell = self.model_params['dust_shell']
            self.shell = shell


        self.dust_shell = (shell/shell.sum()) * self.dust_star_contrast
        self.shell = (shell/shell.sum()) * self.dust_star_contrast

        self.H = self.H/self.H.sum()
        self.V = self.V/self.V.sum()
        self.H45 = self.H45/self.H45.sum()
        self.V45 = self.V45 / self.V45.sum()


        self.H_diff =   self.shell * self.H
        self.V_diff =   self.shell * self.V
        self.H45_diff = self.shell * self.H45
        self.V45_diff = self.shell * self.V45

        self.H_diff = self.H_diff.sum(axis=2)
        self.V_diff = self.V_diff.sum(axis=2)
        self.H45_diff = self.H45_diff.sum(axis=2)
        self.V45_diff = self.V45_diff.sum(axis=2)

        total_flux = np.sum(self.H_diff + self.V_diff)
        self.star = (star / star.sum()) * total_flux * self.dust_star_contrast



        self.H_diff =   self.H_diff   + self.star
        self.V_diff =   self.V_diff   + self.star
        self.H45_diff = self.H45_diff + self.star
        self.V45_diff = self.V45_diff + self.star  # physically accurate images


        self.image_I = self.H_diff + self.V_diff
        self.image_Q = self.H_diff - self.V_diff
        self.image_U = self.H45_diff - self.V45_diff

        return



    def simulate_nrm(self):#make_pol_diff_vis(self):

        self.size_biggest_baseline_m = self.wavelength / (self.pixel_ratio * 2 * self.mas_to_rad)
        self.image_size = len(self.V_diff)
        power_m_per_pix = self.size_biggest_baseline_m / (self.image_size / 2)
        self.power_m_per_pix = power_m_per_pix

        ft_Hdiff = self.apply_DFTM1(self.H_diff, self.dftm_grid)
        ft_Vdiff = self.apply_DFTM1(self.V_diff, self.dftm_grid)
        ft_H45diff = self.apply_DFTM1(self.H45_diff, self.dftm_grid)
        ft_V45diff = self.apply_DFTM1(self.V45_diff, self.dftm_grid)

        self.hwp0_vis = self.calc_vis(ft_Hdiff)
        self.hwp225_vis =  self.calc_vis(ft_H45diff)
        self.hwp45_vis =  self.calc_vis(ft_Vdiff)
        self.hwp675_vis  =  self.calc_vis(ft_V45diff)


        self.hwp0_cp = self.calc_cp(ft_Hdiff, self.indx_of_cp)
        self.hwp225_cp = self.calc_cp(ft_H45diff, self.indx_of_cp)
        self.hwp45_cp = self.calc_cp(ft_Vdiff, self.indx_of_cp)
        self.hwp675_cp = self.calc_cp(ft_V45diff, self.indx_of_cp)

        self.diff_Q =  self.hwp0_vis/self.hwp45_vis
        self.diff_U = self.hwp225_vis/self.hwp675_vis
        self.Q_cp = self.hwp0_cp - self.hwp45_cp
        self.U_cp = self.hwp225_cp - self.hwp675_cp

        self.y_model = np.concatenate((self.diff_Q, self.diff_U, self.Q_cp, self.U_cp))

        return self.y_model


    def calc_vis(self, fft):
        vis = np.abs(fft)**2
        return vis


    def apply_DFTM1(self,image, dftm):
        '''Apply a direct Fourier transform matrix to an image.'''

        image /= image.sum()
        return np.dot(dftm, image.ravel())

    def calc_cp(self, fft_arr, indx_of_cp):

        cvis_cals = fft_arr[indx_of_cp]
        bispectrum = cvis_cals[0, :] * cvis_cals[1, :] * np.conj(cvis_cals[2, :])
        closure_phases = np.angle(bispectrum, deg=True)
        return closure_phases



    def calc_bispectrum_real(self, fft_arr, indx_of_cp):
        indx_ = np.array(indx_of_cp)
        cvis_cals = fft_arr[indx_]
        bispectrum = cvis_cals[0, :] * cvis_cals[1, :] * np.conj(cvis_cals[2, :])
        real_bi = np.real(bispectrum)
        return real_bi


    def calc_bispectrum_imag(self, fft_arr, indx_of_cp):
        indx_ = np.array(indx_of_cp)
        cvis_cals = fft_arr[indx_]
        bispectrum = cvis_cals[0, :] * cvis_cals[1, :] * np.conj(cvis_cals[2, :])
        imag_bi = np.imag(bispectrum)
        return imag_bi
