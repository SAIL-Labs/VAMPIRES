from JIT2make_3D_geom_stars import geometric_star
import os
import pylab as plt
plt.ion()
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
import time
from helper_make_3D_geom_stars import comp_higher_matrix_mult, comp_higher_element_mult, cut_out_background
import useful_functions as uf
from jax.tree_util import tree_map
import interpax as ipx
plt.rcParams['image.origin'] = 'lower'


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

def calc_vis( fft):
    vis = np.abs(fft) ** 2
    return vis

def comp_higher_matrix_mult(matrixx, cube):
    final_mm = np.nan*np.ones(np.shape(cube))
    for y in range(np.shape(cube)[1]):
        for z in range(np.shape(cube)[2]):
                final_mm[:, y,z] = np.matmul(matrixx, cube[:, y, z])

    return final_mm

def gkern(l=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def study_through_lp(final_stokes):
    lp_H = 0.5 * np.array([[1, 1, 0, 0],
                           [1, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
    lp_V = 0.5 * np.array([[1, -1, 0, 0],
                           [-1, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
    lp_H45 = 0.5 * np.array([[1, 0, 1, 0],
                             [0, 0, 0, 0],
                             [1, 0, 1, 0],
                             [0, 0, 0, 0]])
    lp_V45 = 0.5 * np.array([[1, 0, -1, 0],
                             [0, 0, 0, 0],
                             [-1, 0, 1, 0],
                             [0, 0, 0, 0]])


    H = comp_higher_matrix_mult(lp_H ,final_stokes)
    V =  comp_higher_matrix_mult(lp_V ,final_stokes)
    H45 = comp_higher_matrix_mult( lp_H45 , final_stokes)
    V45 =  comp_higher_matrix_mult(lp_V45 , final_stokes)

    H = H[0,:, : ]
    V = V[0,:, : ]
    H45 = H45[0,:, : ]
    V45 = V45[0,:, : ]

    return H, V, H45, V45

def chi_squ(experimental, model, exp_err, ndim):
    chi_squ_new = np.sum(((experimental - model) ** 2) / exp_err**2)
    chi_squ_norm = chi_squ_new / (153*2-ndim)
    return chi_squ_norm



def apply_DFTM1( image, dftm):
    '''Apply a direct Fourier transform matrix to an image.'''
    image /= image.sum()
    return np.dot(dftm, image.ravel())


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


pol_mode =  'rayleigh_resolved'
test = os.getcwd()

# image_size =  101
image_size =  308
star_radius = 7
pixel_ratio = 1
wavelength = 750 * 10 ** (-9)
field_of_view = pixel_ratio  * image_size

# stH, stV, stH45, stV45  =  cut_out_background(image_size, 1, pixel_ratio)

tag = 'muCep_2018_02'
observing_run =  'muCep_2018_02'
size_biggest_baseline_m = ( wavelength) / (pixel_ratio * 2 *  (4.8481368 * 10 ** (-9)))

meta_data =  '../geometric_models_data/{}/'.format(observing_run)

scat_H =  np.load(meta_data  + 'stokesH_scattering.npy')#[0:-1, 0:-1, 0:-1]#[29:-28, 29:-28, 29:-28]
scat_V =  np.load(meta_data  + 'stokesV_scattering.npy')#[0:-1, 0:-1, 0:-1]##[29:-28, 29:-28, 29:-28]
scat_H45 =  np.load(meta_data  + 'stokesH45_scattering.npy')#[0:-1, 0:-1, 0:-1]##[29:-28, 29:-28, 29:-28]
scat_V45 =  np.load(meta_data  + 'stokesV45_scattering.npy')#[0:-1, 0:-1, 0:-1]##[29:-28, 29:-28, 29:-28]

ucoords = np.load(meta_data + 'u_coords.npy')
vcoords = np.load(meta_data + 'v_coords.npy')
xdatar = np.arctan(vcoords/ucoords)
zdatar = np.sqrt(ucoords**2 + vcoords**2)
uv_concat = np.concatenate((np.expand_dims(ucoords, axis = 1), np.expand_dims(vcoords, axis = 1)), axis = 1)

# x, y, z = np.ogrid[-182: 183, -182: 183, -182: 183]
# x, y, z = np.ogrid[-50: 51, -50: 51, -50: 51]
x, y, z = np.ogrid[-154:154, -154:154, -154:154]

xx, yy = np.meshgrid(x.flatten(), y.flatten())
dftm_grid = compute_DFTM1(xx.flatten(), yy.flatten(), uv_concat, wavelength)

indx_of_cp = np.load(meta_data + 'indx_of_cp.npy')

qpolr =   np.load(meta_data + 'q_vis.npy')
upolr =   np.load(meta_data + 'u_vis.npy')
qpolerr = np.load(meta_data + 'q_vis_err.npy')
upolerr = np.load(meta_data + 'u_vis_err.npy')

ydatar = np.concatenate((qpolr, upolr))
ydataerr = np.concatenate((qpolerr, upolerr))



model_params = {'image_size': image_size,
                'pixel_ratio': pixel_ratio,
                'fpad': image_size * 3,
                'distance_to_star': 100,
                'size_biggest_baseline_m': size_biggest_baseline_m,
                'plot': False,
                'wavelength': wavelength,
                'u_coords': ucoords,
                'v_coords': vcoords,
                'indx_of_cp': 0,
                'xdata': xdatar,
                'zdata': zdatar,
                'indx_of_cp': indx_of_cp,

                'H_scat': scat_H,
                'V_scat': scat_V,
                'H45_scat': scat_H45,
                'V45_scat': scat_V45,
                'bright_location_theta': np.pi / 8,
                'bright_location_phi': np.pi / 8,
                'bright_size': np.pi / 8,
                'bright_mag': 4,

                'dftm_grid': dftm_grid,
                'ydata_real': ydatar,
                'ydata_real_err': ydataerr,
                'star_radius': star_radius,
                'onsky_rotation': 0,
                'a_scat': 300 * 10 ** (-9),
                'n_scat': 1.636,
                'I0': 1,
                'dust_shape': 'ellipse', #ellipse',#_and_blob',
                'dust_pl_const': 1,
                'dust_pl_exp': 20, #0.34,
                'shell_thickness':0.12,
                'a': 185.29,
                'thick1': 4.98,
                'thick2': 149.49,
                'a2': 21.01,
                'ellipse_contrast_ratio':0.036,

                'b': 13,
                'c': 19,
                'h': 0,
                'k': 0,
                'n': 0,
                'alpha': 41.94,

                'Q_rot': 0,
                'U_rot': 0,
                'vec_rot': 0.3,

                'a3':  26,
                'b3': 26.20,  # 16
                'c3': 26.20,

                'b2': 26.20, # 16
                'c2': 26.20,
                'h2': 0,
                'k2': 0,
                'n2': 0,
                'alpha2': 45,

                'dust_star_contrast': 4.29, #0.0001,


                'blob_radial_distance': 337,
                'theta_blob': 1.5,
                'phi_blob': 4.52,
                'r_blob': 35,
                'blob_contrast': 133,
                'altitude': 45,

                'blob_radial_distance1': 0,
                'theta_blob1': 0,
                'phi_blob1': 0,
                'r_blob1': 10,
                'blob_contrast1':1,

                'blob_radial_distance2': 30,
                'theta_blob2': 1.5,
                'phi_blob2': 2.9,
                'r_blob2': 10,
                'blob_contrast2': 5,

                'ellipse_scale': 2,

                'blob_radial_distance3': 40,
                'theta_blob3': 1.5,
                'phi_blob3': 2.9,
                'r_blob3': 5,
                'blob_contrast3': 0.01,

                'steepness':1,
                'proto_rotate':0,
                'proto_slant': 45,
                'disk_height': 500,
                'disk_start': 0,

                'gauss_start': 14,
                'theta_gauss': 1.5,
                'phi_gauss': np.pi / 3,
                'sd_gauss': 13,
                'gauss_contrast': 2,

                'rot_ang': 0,
                'obs_run': 'mucep_test/'}



model_args =  { 'a': model_params['a'],
                'b': model_params['b'],
                'alpha': model_params['alpha'],
            'dust_star_contrast': model_params['dust_star_contrast']}


model_param_ranges =   { 'a': [12, 200],
                        'b': [12, 200],
                         'alpha':[0, 90],
            'dust_star_contrast': [0.01, 50]}



stellar_object = geometric_star(model_params)
stellar_object.make_3D_model()
stellar_object.make_pol_diff_vis()

plt.figure()
tep = np.max(np.array([np.abs(np.min(stellar_object.image_Q)), np.abs(np.max(stellar_object.image_Q))]))
plt.imshow(stellar_object.image_Q, cmap = 'seismic', clim = [-tep, tep])
plt.savefig('test.pdf')

# def f(model_args):
#
#     list_keys = list(model_args.keys())
#
#     for i in range(len(list_keys)):
#         model_params[list_keys[i]] = model_args[list_keys[i]]
#
#     time_start = time.time()
#     stellar_object = geometric_star(model_params)
#     stellar_object.make_3D_model()
#     stellar_object.make_pol_diff_vis()
#     time_stop = time.time()
#
#     print('Time to create model... {:.2f} seconds'.format(time_stop-time_start))
#     return stellar_object, stellar_object.y_model
#
#
# fig, (ax) = plt.subplots(2, 4, gridspec_kw={'width_ratios': [3, 1, 1, 1], 'height_ratios': [1.1,1]})
# fig.set_size_inches(15,6)
#
#
# #zdatar, indx_of_cp
#
# bl_things = zdatar[indx_of_cp]
# maxbl = np.max(bl_things, axis = 0)
#
# tep = f(model_args)
#
# print(np.shape(tep[1][0:153]))
# print(np.shape(x))
# print(np.shape(z))
#
#
#
# # im3 = ax[0,0].errorbar(x,qpolr, yerr = qpolerr, fmt = 'none', ecolor = 'lightgrey')
# im3 = ax[0,0].scatter(x, tep[1][0:153], c=abs( z), cmap='jet', marker='x', alpha = 1, label= 'Real {:.4f}'.format(np.mean(qpolr)))
# cbar3 = fig.colorbar(im3, ax=ax[0,0])
# ax[0,0].set_title('Stokes Q')
# cbar3.set_label('Baseline Length (m)')
# mean_Q = np.mean(tep[1][0:153])
# min_Q = np.min(tep[1][0:153])
# max_Q = np.max(tep[1][0:153])
# ax[0,0].plot(x,  np.mean( tep[1][0:153])*np.ones((153, 1)),    linestyle='dashed', color='black' )
# ax[0,0].set_ylabel('Differential Visibility V^2')
# # ax[0,0].legend()
#
#
#
# # im4 = ax[1,0].errorbar(x, upolr, yerr = upolerr, fmt = 'none', ecolor = 'lightgrey')
# im4 = ax[1,0].scatter(x, tep[1][153:153*2], c=abs(z), cmap='jet', marker='x', alpha = 1, label= 'Real {:.4f}'.format(np.mean(upolr)))
# cbar4 = fig.colorbar(im4, ax=ax[1,0])
# ax[1,0].set_title('Stokes U')
#
# cbar4.set_label('Baseline Length (m)')
# ax[1,0].set_ylabel('Differential Visibility V^2')
# ax[1,0].set_xlabel('Azimuth Angle (rad)')
#
# mean_U = np.mean(tep[1][153:153*2])
# min_U = np.min(tep[1][153:153*2])
# max_U = np.max(tep[1][153:153*2])
# ax[1,0].plot(x, np.mean(tep[1][153:153*2])*np.ones((153, 1)),  linestyle='dashed', color='black' )
# # ax[1,0].legend()
#
#
# # instead would like to plot I
#
#
# pol_phi5 = np.arctan2(tep[0].image_U, tep[0].image_Q) /2
# pol_P5 = np.sqrt(tep[0].image_U ** 2 + tep[0].image_Q ** 2)
#
# sto_I = scat_H + scat_V
# Q = scat_H - scat_V
# U = scat_H45 - scat_V45
# I = scat_H + scat_V
#
# P = np.sqrt(Q ** 2 + U ** 2) / (I + 1e-36)
# dust_shell_final = tep[0].dust_shell * P
# dust_shell_final = dust_shell_final / dust_shell_final.sum()
#
# im5 = ax[0, 1].imshow(dust_shell_final.sum(axis=2))#, clim = []) #[25:75, 25:75])
# cbar4 = fig.colorbar(im5, ax=ax[0, 1])
# ax[0,1].set_title('Stokes I - Dust')#, PA {:.2f}'.format(tep[0].pa_theta))
# ax[0,1].set_xlabel('mas')
# ax[0,1].set_ylabel('mas')
# # ax[0,1].set_ylim([100, 200])
# # ax[0,1].set_xlim([100, 200])
#
# tot = np.max(np.array([np.min(tep[0].image_Q), np.max(tep[0].image_Q)]))
# im6 = ax[0, 2].imshow( tep[0].image_Q, cmap = 'seismic', clim = [-tot, tot])  #, clim = [1,1.05])
# # ax[0,2].scatter(tep[0].u_coords_pix_long, tep[0].v_coords_pix_long, color = 'red', s = 0.5)
# ax[0,2].set_title('Stokes Q - Dust')
# ax[0,2].set_xlabel('mas')
# ax[0,2].set_ylabel('mas')
# cbar4 = fig.colorbar(im6, ax=ax[0,2])
# # ax[0,2].set_ylim([100, 200])
# # ax[0,2].set_xlim([100, 200])
#
# tot = np.max(np.array([np.min(tep[0].image_U), np.max(tep[0].image_U)]))
# im6 = ax[0, 3].imshow( tep[0].image_U, cmap = 'seismic', clim = [-tot, tot])  #, clim = [1,1.05])
# # ax[0,3].scatter(tep[0].u_coords_pix_long, tep[0].v_coords_pix_long, color = 'red', s = 0.5)
# ax[0, 3].set_title('Stokes U - Dust')
# ax[0,3].set_xlabel('mas')
# ax[0,3].set_ylabel('mas')
# cbar4 = fig.colorbar(im6, ax=ax[0, 3])
# # ax[0,3].set_ylim([100, 200])
# # ax[0,3].set_xlim([100, 200])
#
#
# xE = np.arange(0, np.shape(pol_phi5)[0], 1)
# yE = np.arange(0, np.shape(pol_phi5)[0], 1)
# X, Y = np.meshgrid(xE, yE) # had a + pi /2
# ax[1, 1].quiver(X, Y, pol_P5 / pol_P5.max(), pol_P5 / pol_P5.max(), angles=np.rad2deg(pol_phi5  + np.pi/2),
#            pivot='mid', headwidth=0, headlength=0, scale=50)
#
# ax[1, 1].set_title('Polarization')
# ax[1, 1].set_xlabel('mas')
# ax[1, 1].set_ylabel('mas')
# # ax[1, 1].set_ylim([130, 175])
# # ax[1, 1].set_xlim([130, 175])
#
#
#
# im11 = ax[1,2].scatter(maxbl, tep[0].Q_cp)
# ax[1, 2].set_title('Stokes Q CP')
# ax[1, 2].set_xlabel('Max Bl')
# ax[1, 2].set_ylabel('Q CP')
#
#
# im11 = ax[1,3].scatter(maxbl, tep[0].U_cp)
# ax[1, 3].set_title('Stokes U CP')
# ax[1, 3].set_xlabel('Max Bl')
# ax[1, 3].set_ylabel('U CP')
#
# plt.tight_layout()
# plt.show()
#
#
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
#
#
#
#
#
#
#
# axes_config = [[0.25, 0.1, 0.65, 0.03]]
# axes_direction = [['horizontal']]
#
# for i in range(1,len(model_args)):
#     axes_config.append([0.25, 0.1+ 0.04*i, 0.65, 0.03])
#     axes_direction.append(['horizontal'])
#
#
# x = stellar_object.xdata
# z =  stellar_object.zdata
#
# chisqu_Q = chi_squ(stellar_object.ydata_real[0:153], stellar_object.y_model[0:153], stellar_object.ydata_real_err[0:153], len(model_args))
# chisqu_U = chi_squ(stellar_object.ydata_real[153:153 * 2], stellar_object.y_model[153:153 * 2], stellar_object.ydata_real_err[153:153 * 2], len(model_args))
#
# fig, (ax) = plt.subplots(2, 4, gridspec_kw={'width_ratios': [3, 1, 1, 1], 'height_ratios': [1.1,1]})
# fig.set_size_inches(15,6)
#
#
# #zdatar, indx_of_cp
#
# bl_things = zdatar[indx_of_cp]
# maxbl = np.max(bl_things, axis = 0)
#
# tep = f(model_args)
#
# # im3 = ax[0,0].errorbar(x,qpolr, yerr = qpolerr, fmt = 'none', ecolor = 'lightgrey')
# im3 = ax[0,0].scatter(x, tep[1][0:153], c=abs(z), cmap='jet', marker='x', alpha = 1, label= 'Real {:.4f}'.format(np.mean(qpolr)))
# cbar3 = fig.colorbar(im3, ax=ax[0,0])
# ax[0,0].set_title('Stokes Q')
# cbar3.set_label('Baseline Length (m)')
# mean_Q = np.mean(tep[1][0:153])
# min_Q = np.min(tep[1][0:153])
# max_Q = np.max(tep[1][0:153])
# ax[0,0].plot(x,  np.mean( tep[1][0:153])*np.ones((153, 1)),    linestyle='dashed', color='black' )
# ax[0,0].set_ylabel('Differential Visibility V^2')
# # ax[0,0].legend()
#
#
#
# # im4 = ax[1,0].errorbar(x, upolr, yerr = upolerr, fmt = 'none', ecolor = 'lightgrey')
# im4 = ax[1,0].scatter(x, tep[1][153:153*2], c=abs(z), cmap='jet', marker='x', alpha = 1, label= 'Real {:.4f}'.format(np.mean(upolr)))
# cbar4 = fig.colorbar(im4, ax=ax[1,0])
# ax[1,0].set_title('Stokes U')
#
# cbar4.set_label('Baseline Length (m)')
# ax[1,0].set_ylabel('Differential Visibility V^2')
# ax[1,0].set_xlabel('Azimuth Angle (rad)')
#
# mean_U = np.mean(tep[1][153:153*2])
# min_U = np.min(tep[1][153:153*2])
# max_U = np.max(tep[1][153:153*2])
# ax[1,0].plot(x, np.mean(tep[1][153:153*2])*np.ones((153, 1)),  linestyle='dashed', color='black' )
# # ax[1,0].legend()
#
#
# # instead would like to plot I
#
#
# pol_phi5 = np.arctan2(tep[0].image_U, tep[0].image_Q) /2
# pol_P5 = np.sqrt(tep[0].image_U ** 2 + tep[0].image_Q ** 2)
#
# sto_I = scat_H + scat_V
# Q = scat_H - scat_V
# U = scat_H45 - scat_V45
# I = scat_H + scat_V
#
# P = np.sqrt(Q ** 2 + U ** 2) / (I + 1e-36)
# dust_shell_final = tep[0].dust_shell * P
# dust_shell_final = dust_shell_final / dust_shell_final.sum()
#
# im5 = ax[0, 1].imshow(dust_shell_final.sum(axis=2))#, clim = []) #[25:75, 25:75])
# cbar4 = fig.colorbar(im5, ax=ax[0, 1])
# ax[0,1].set_title('Stokes I - Dust')#, PA {:.2f}'.format(tep[0].pa_theta))
# ax[0,1].set_xlabel('mas')
# ax[0,1].set_ylabel('mas')
# # ax[0,1].set_ylim([100, 200])
# # ax[0,1].set_xlim([100, 200])
#
# tot = np.max(np.array([np.min(tep[0].image_Q), np.max(tep[0].image_Q)]))
# im6 = ax[0, 2].imshow( tep[0].image_Q, cmap = 'seismic', clim = [-tot, tot])  #, clim = [1,1.05])
# # ax[0,2].scatter(tep[0].u_coords_pix_long, tep[0].v_coords_pix_long, color = 'red', s = 0.5)
# ax[0,2].set_title('Stokes Q - Dust')
# ax[0,2].set_xlabel('mas')
# ax[0,2].set_ylabel('mas')
# cbar4 = fig.colorbar(im6, ax=ax[0,2])
# # ax[0,2].set_ylim([100, 200])
# # ax[0,2].set_xlim([100, 200])
#
# tot = np.max(np.array([np.min(tep[0].image_U), np.max(tep[0].image_U)]))
# im6 = ax[0, 3].imshow( tep[0].image_U, cmap = 'seismic', clim = [-tot, tot])  #, clim = [1,1.05])
# # ax[0,3].scatter(tep[0].u_coords_pix_long, tep[0].v_coords_pix_long, color = 'red', s = 0.5)
# ax[0, 3].set_title('Stokes U - Dust')
# ax[0,3].set_xlabel('mas')
# ax[0,3].set_ylabel('mas')
# cbar4 = fig.colorbar(im6, ax=ax[0, 3])
# # ax[0,3].set_ylim([100, 200])
# # ax[0,3].set_xlim([100, 200])
#
#
# xE = np.arange(0, np.shape(pol_phi5)[0], 1)
# yE = np.arange(0, np.shape(pol_phi5)[0], 1)
# X, Y = np.meshgrid(xE, yE) # had a + pi /2
# ax[1, 1].quiver(X, Y, pol_P5 / pol_P5.max(), pol_P5 / pol_P5.max(), angles=np.rad2deg(pol_phi5  + np.pi/2),
#            pivot='mid', headwidth=0, headlength=0, scale=50)
#
# ax[1, 1].set_title('Polarization')
# ax[1, 1].set_xlabel('mas')
# ax[1, 1].set_ylabel('mas')
# # ax[1, 1].set_ylim([130, 175])
# # ax[1, 1].set_xlim([130, 175])
#
#
#
# im11 = ax[1,2].scatter(maxbl, tep[0].Q_cp)
# ax[1, 2].set_title('Stokes Q CP')
# ax[1, 2].set_xlabel('Max Bl')
# ax[1, 2].set_ylabel('Q CP')
#
#
# im11 = ax[1,3].scatter(maxbl, tep[0].U_cp)
# ax[1, 3].set_title('Stokes U CP')
# ax[1, 3].set_xlabel('Max Bl')
# ax[1, 3].set_ylabel('U CP')
#
# plt.tight_layout()
# plt.show()
#
#
# fig2, ax2 = plt.subplots(figsize = (7,5))
# ax2.set_visible(False)
#
# #
#
# slider_dictionary = {}
# # fig2.subplots_adjust(left=0.25, bottom=0.25)
#
#
# for i in range(len(model_args)):
#     ax_current = fig2.add_axes(axes_config[i])
#     current_slider = Slider(
#                     ax=ax_current,
#                     label=list(model_args.keys())[i],
#                     valmin= model_param_ranges[list(model_args.keys())[i]][0],
#                     valmax= model_param_ranges[list(model_args.keys())[i]][1],
#                     valinit=model_args[list(model_args.keys())[i]],
#                     orientation=axes_direction[i][0]
#                     )
#     slider_dictionary[list(model_args.keys())[i]] = current_slider
#
#
#
# def update(val):
#
#     for i in range(len(model_args)):
#         model_args[list(model_args.keys())[i]] = slider_dictionary[list(model_args.keys())[i]].val
#
#
#     tep = f(model_args)
#
#
#     ax[1,0].cla()
#     x = stellar_object.xdata
#     ax[1,1].cla()
#     ax[1,2].cla()
#     ax[1,3].cla()
#
#     ax[0,0].cla()
#     ax[0,1].cla()
#     ax[0,2].cla()
#     ax[0,3].cla()
#
#     im3 = ax[0, 0].scatter(x, tep[1][0:153], c=z, cmap='jet', marker='x', label='Model {:.4f}'.format(np.mean(tep[1][0:153])))
#     # im33 = ax[0, 0].scatter(x, qpolr, c=abs(z), cmap='jet', marker='o', alpha=0.1, label='Real {:.4f}'.format(np.mean(qpolr)))
#     #cbar3 = fig.colorbar(im3, ax=ax[0, 0])
#     ax[0, 0].set_title('Stokes Q')
#    # cbar3.set_label('Baseline Length (m)')
#     mean_Q = np.mean(tep[1][0:153])
#     min_Q = np.min(tep[1][0:153])
#     max_Q = np.max(tep[1][0:153])
#     ax[0, 0].plot(x, np.mean(tep[1][0:153])*np.ones((153, 1)), linestyle='dashed', color='black')
#     # ax[0, 0].legend()
#     ax[0, 0].set_ylabel('Differential Visibility V^2')
#     # ax[0,0].set_xlabel('Azimuth Angle (rad)')
#     # ax[0,0].legend()
#     # mini = np.min(tep[1][0:153])
#     # maxi = np.max(tep[1][0:153])
#     # scat_y = np.arange(mini, maxi, (maxi - mini) / 153)
#     # ax[0,0].scatter(-0.3*np.ones(np.shape(x)), scat_y, marker = '.', s = 1)
#
#     im4 = ax[1, 0].scatter(x, tep[1][153:153*2], c=z, cmap='jet', marker='x', label='Model {:.4f}'.format(np.mean(tep[1][153:153*2])))
#     # im44 = ax[1, 0].scatter(x, upolr, c=abs(z), cmap='jet', marker='o', alpha=0.1, label='Real {:.4f}'.format(np.mean(upolr)))
#     #cbar4 = fig.colorbar(im4, ax=ax[1, 0])
#     ax[1, 0].set_title('Stokes U')
#     #cbar4.set_label('Baseline Length (m)')
#     ax[1, 0].set_ylabel('Differential Visibility V^2')
#     ax[1, 0].set_xlabel('Azimuth Angle (rad)')
#     mini = np.min(tep[1][153:153*2])
#     maxi = np.max(tep[1][153:153*2])
#     scat_y = np.arange(mini, maxi, (maxi - mini) / 153)
#     # ax[1,0].scatter(-0.3*np.ones(np.shape(x)), scat_y, marker = '.', s = 1)
#     # ax[1, 0].legend()
#     mean_U = np.mean(tep[1][153:153*2])
#     min_U = np.min(tep[1][153:153*2])
#     max_U = np.max(tep[1][153:153*2])
#     ax[1, 0].plot(x,  np.mean(tep[1][153:153*2])*np.ones((153, 1)), linestyle='dashed', color='black')
#     # ax[1, 0].legend()
#
#     # instead would like to plot I
#
#     sto_I = scat_H + scat_V
#     Q = scat_H - scat_V
#     U = scat_H45 - scat_V45
#     I = scat_H + scat_V
#
#     P = np.sqrt(Q ** 2 + U ** 2) / (I + 1e-36)
#     dust_shell_final = tep[0].dust_shell * P
#     dust_shell_final = dust_shell_final/dust_shell_final.sum()
#
#
#     im5 = ax[0, 1].imshow(dust_shell_final.sum(axis=2))  # [25:75, 25:75])
#     # cbar4 = fig.colorbar(im5, ax=ax[0, 1])
#     ax[0, 1].set_title('Stokes I - Dust')#, PA {:.2f}'.format(tep[0].pa_theta))
#     ax[0, 1].set_xlabel('mas')
#     ax[0, 1].set_ylabel('mas')
#     # ax[0, 1].set_ylim([100, 200])
#     # ax[0, 1].set_xlim([100, 200])
#
#     tot = np.max(np.array([np.min(tep[0].image_U), np.max(tep[0].image_U)]))
#     im6 = ax[0, 2].imshow(tep[0].image_Q, cmap = 'seismic', clim = [-tot, tot]) # , clim = [1,1.05])
#     # ax[0, 2].scatter(tep[0].u_coords_pix_long, tep[0].v_coords_pix_long, color='red', s=0.5)
#     ax[0, 2].set_title('Stokes Q - Dust')
#     ax[0, 2].set_xlabel('mas')
#     ax[0, 2].set_ylabel('mas')
#     # cbar4 = fig.colorbar(im6, ax=ax[0,2])
#     # ax[0, 2].set_ylim([100, 200])
#     # ax[0, 2].set_xlim([100, 200])
#
#     tot = np.max(np.array([np.min(tep[0].image_U), np.max(tep[0].image_U)]))
#     im6 = ax[0, 3].imshow(tep[0].image_U, cmap = 'seismic', clim = [-tot, tot])  # , clim = [1,1.05])
#     # ax[0, 3].scatter(tep[0].u_coords_pix_long, tep[0].v_coords_pix_long, color='red', s=0.5)
#     ax[0, 3].set_xlabel('mas')
#     ax[0, 3].set_ylabel('mas')
#     ax[0, 3].set_title('Stokes U - Dust')
#     # cbar4 = fig.colorbar(im6, ax=ax[0, 3])
#     # ax[0, 3].set_ylim([100, 200])
#     # ax[0, 3].set_xlim([100, 200])
#
#     pol_phi5 = np.arctan2(tep[0].image_U, tep[0].image_Q) / 2
#     pol_P5 = np.sqrt(tep[0].image_U ** 2 + tep[0].image_Q ** 2)
#
#     xE = np.arange(0, np.shape(pol_phi5)[0], 1)
#     yE = np.arange(0, np.shape(pol_phi5)[0], 1)
#     X, Y = np.meshgrid(xE, yE)
#     ax[1, 1].quiver(X, Y, pol_P5 / pol_P5.max(),pol_P5 / pol_P5.max(), angles=np.rad2deg(pol_phi5  + np.pi/2),
#                pivot='mid', headwidth=0, headlength=0, scale=50)
#
#     ax[1, 1].set_title('Polarization')
#     ax[1, 1].set_xlabel('mas')
#     ax[1, 1].set_ylabel('mas')
#     # ax[1, 1].set_ylim([130, 175])
#     # ax[1, 1].set_xlim([130, 175])
#
#
#     im11 = ax[1, 2].scatter(maxbl, tep[0].Q_cp)
#     ax[1, 2].set_title('Stokes Q CP')
#     ax[1, 2].set_xlabel('Max Bl')
#     ax[1, 2].set_ylabel('Q CP')
#
#     im11 = ax[1, 3].scatter(maxbl, tep[0].U_cp)
#     ax[1, 3].set_title('Stokes U CP')
#     ax[1, 3].set_xlabel('Max Bl')
#     ax[1, 3].set_ylabel('U CP')
#
#     fig.canvas.draw_idle()
#     plt.tight_layout()
#
#
# for i in range(len(slider_dictionary)):
#     slider_dictionary[list(slider_dictionary.keys())[i]].on_changed(update)
#
# # Call update once to populate the plots on startup
# update(None)
#
# plt.ion()
