#!/usr/bin/env python3
# ================================================================
#  Fast (Numba-only) scattering-grid builder for Enstatite grains
#  • miepython ≥ 3.0 (compiled kernel _S1_S2_nb, CPU dispatcher)
#  • grain-size distributions compressed → centres + weights
#  • full band-pass averaging with λ-dependent refractive index
#  • wall-clock timing for every λ-band and every radius set
# ================================================================

# ───────── 0 · env: force JIT & where to cache Numba code ───────────
import os, tempfile, time
os.environ["MIEPYTHON_USE_JIT"] = "1"
os.environ["NUMBA_CACHE_DIR"]  = tempfile.gettempdir()

# ───────── imports ──────────────────────────────────────────────────
import numpy as np
import miepython as mie
from numba import njit, prange
from numba.core.registry import CPUDispatcher
from inspect import signature

# ───────── compiled dispatcher (abort if absent) ────────────────────
from miepython.mie_jit import _S1_S2_nb as S1_S2_fast
if not isinstance(S1_S2_fast, CPUDispatcher):
    raise RuntimeError("_S1_S2_nb is not a Numba dispatcher")
_SIG_LEN = len(signature(S1_S2_fast.py_func).parameters)
if _SIG_LEN not in (3, 4):
    raise RuntimeError("Unexpected _S1_S2_nb signature")
print("[info] Using compiled miepython.mie_jit._S1_S2_nb()")

# ───────── 1 · immutable geometry terms ─────────────────────────────
x = np.linspace(-154, 154, 309)
y = np.linspace(-154, 154, 309)
z = np.linspace(-154, 154, 309)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
theta = np.arccos(np.clip(Z / np.sqrt(X**2 + Y**2 + Z**2), -1., 1.))

def _init_scatter_geometry(X, Y, Z, theta):
    mu  = np.cos(theta.ravel())
    norm = np.linalg.norm(np.stack([X, Y, Z]), axis=0)
    k_out = np.stack([(X/norm).ravel(),
                      (Y/norm).ravel(),
                      (Z/norm).ravel()], axis=1)
    kin  = np.array([0., 0., 1.])
    n_plane = np.cross(kin, k_out)
    n_norm  = np.linalg.norm(n_plane, axis=1, keepdims=True)
    n_norm[n_norm == 0] = 1.
    V_dir = n_plane / n_norm
    H_dir = np.cross(V_dir, k_out)
    H_dir /= np.linalg.norm(H_dir, axis=1, keepdims=True)

    d45 = np.array([[1,  1, 0],
                    [1, -1, 0]], dtype=float) / np.sqrt(2)
    coeff = np.empty((mu.size, 8))
    coeff[:,0] = H_dir[:,0]**2;  coeff[:,1] = V_dir[:,0]**2
    coeff[:,2] = H_dir[:,1]**2;  coeff[:,3] = V_dir[:,1]**2
    coeff[:,4] = (H_dir @ d45[0])**2; coeff[:,5] = (V_dir @ d45[0])**2
    coeff[:,6] = (H_dir @ d45[1])**2; coeff[:,7] = (V_dir @ d45[1])**2
    return mu.astype(float), coeff.astype(float)

MU, COEFF = _init_scatter_geometry(X, Y, Z, theta)

# ───────── 2 · Numba kernel (λ-dependent m) ──────────────────────────
if _SIG_LEN == 4:                                 # ≥ 3.0.2
    @njit(parallel=True, cache=True, fastmath=True)
    def _accumulate(mr, mi, radii, w, wavs, mu, coeff):
        # mr, mi: (nw,) arrays
        out = np.zeros((mu.size, 4))
        for i in prange(radii.size):
            r, wgt = radii[i], w[i]
            for j in range(wavs.size):
                m = complex(mr[j], mi[j])
                x = 2.0 * np.pi * r / wavs[j]
                S1, S2 = S1_S2_fast(m, x, mu, 0)
                dH, dV = np.abs(S2)**2 * wgt, np.abs(S1)**2 * wgt
                out[:,0] += dH*coeff[:,0] + dV*coeff[:,1]
                out[:,1] += dH*coeff[:,2] + dV*coeff[:,3]
                out[:,2] += dH*coeff[:,4] + dV*coeff[:,5]
                out[:,3] += dH*coeff[:,6] + dV*coeff[:,7]
        return out / wavs.size
else:                                             # dev build (3 args)
    @njit(parallel=True, cache=True, fastmath=True)
    def _accumulate(mr, mi, radii, w, wavs, mu, coeff):
        out = np.zeros((mu.size, 4))
        for i in prange(radii.size):
            r, wgt = radii[i], w[i]
            for j in range(wavs.size):
                m = complex(mr[j], mi[j])
                x = 2.0 * np.pi * r / wavs[j]
                S1, S2 = S1_S2_fast(m, x, mu)
                dH, dV = np.abs(S2)**2 * wgt, np.abs(S1)**2 * wgt
                out[:,0] += dH*coeff[:,0] + dV*coeff[:,1]
                out[:,1] += dH*coeff[:,2] + dV*coeff[:,3]
                out[:,2] += dH*coeff[:,4] + dV*coeff[:,5]
                out[:,3] += dH*coeff[:,6] + dV*coeff[:,7]
        return out / wavs.size

def make_scattering_grids(m_band, wavelengths, radii_m, weights):
    """m_band: complex array same length as wavelengths."""
    mr = m_band.real.astype(np.float64)
    mi = m_band.imag.astype(np.float64)
    acc = _accumulate(mr, mi,
                      np.asarray(radii_m), np.asarray(weights),
                      np.asarray(wavelengths), MU, COEFF)
    return (acc[:,0].reshape(X.shape),
            acc[:,1].reshape(X.shape),
            acc[:,2].reshape(X.shape),
            acc[:,3].reshape(X.shape))

# ───────── 3 · grain distributions (vector nbins) ────────────────────
def sample_powerlaw_grains(amin, amax, q, N):
    r = np.random.rand(N)
    if q == -1:
        return amin*(amax/amin)**r
    exp = q + 1
    return (r*(amax**exp-amin**exp) + amin**exp)**(1/exp)

def histogramise(radii_nm, nbins):
    counts, edges = np.histogram(radii_nm, bins=nbins, density=False)
    weights = counts / counts.sum()
    centres = 0.5*(edges[:-1] + edges[1:])
    return centres, weights.astype(np.float64)

def build_compressed(specs, N, nbins):
    nbins = nbins if np.ndim(nbins) else [int(nbins)]*len(specs)
    if len(nbins) != len(specs):
        raise ValueError("nbins length must match specs")
    return [histogramise(sample_powerlaw_grains(*s, N), nb)
            for s, nb in zip(specs, nbins)]

def m_at_lambda_arr(lam_nm_arr):
    """Vectorised complex refractive index."""
    n = np.interp(lam_nm_arr, λ_tab_nm, n_tab)
    k = np.interp(lam_nm_arr, λ_tab_nm, k_tab)
    return n + 1j*k


import os
import time
import numpy as np

import os
import time
import numpy as np

# Define function to interpolate complex refractive index from tabulated data
def m_at_lambda_arr(lam_nm_arr, λ_tab_nm, n_tab, k_tab):
    n = np.interp(lam_nm_arr, λ_tab_nm, n_tab)
    k = np.interp(lam_nm_arr, λ_tab_nm, k_tab)
    return n + 1j * k

# Main function to generate scattering grids from radius specs
def generate_scattering_grids_from_radius_specs(
    E_file, material, radius_specs, nbins_vec, n_samples,
    λ_centres, bandpass_nm, λ_step_nm,
    outdir_base='/import/*1/*/mie_scat_grids_raw/',
):
    """
    Main routine to compute scattering grids from a radius specification setup.
    """

    # Load optical constants
    λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=1).T
    λ_tab_nm *= 1000  # Convert from µm to nm

    # Output directory
    outdir = os.path.join(outdir_base)
    os.makedirs(outdir, exist_ok=True)

    # Generate radius sets
    compressed_sets = build_compressed(radius_specs, n_samples, nbins_vec)
    save_rad_keys = [f"{a}-{b}-{abs(p)}" for a, b, p in radius_specs]

    print(f'Making Scattering Grids for {material}')
    overall_t0 = time.perf_counter()

    for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
        set_t0 = time.perf_counter()
        centres_m = centres_nm * 1e-9
        print(f'Processing radius set {key}')

        for λ0 in λ_centres:
            λ_band_nm = np.arange(
                λ0 - bandpass_nm / 2,
                λ0 + bandpass_nm / 2 + λ_step_nm * 0.5,
                λ_step_nm
            )
            λ_band_m = λ_band_nm * 1e-9
            m_band = m_at_lambda_arr(λ_band_nm, λ_tab_nm, n_tab, k_tab)

            t0 = time.perf_counter()
            H, V, H45, V45 = make_scattering_grids(m_band, λ_band_m, centres_m, weights)
            t1 = time.perf_counter()

            tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
            np.save(os.path.join(outdir, f"H_scat_{tag}"),   H)
            np.save(os.path.join(outdir, f"V_scat_{tag}"),   V)
            np.save(os.path.join(outdir, f"H45_scat_{tag}"), H45)
            np.save(os.path.join(outdir, f"V45_scat_{tag}"), V45)
            print(f"  λ = {λ0}±{bandpass_nm/2} nm "
                  f"({λ_band_nm.size} λ) in {t1 - t0:5.2f} s")

        set_t1 = time.perf_counter()
        print(f'Radius set {key} finished in {set_t1 - set_t0:5.2f} s\n')

    overall_t1 = time.perf_counter()
    print(f"All scattering grids finished in {(overall_t1 - overall_t0) / 60:.1f} min.")


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



radius_specs = [ (100, 100.1, -0.1)]


n_samples = 5#0000
nbins_vec =  [1] * len(radius_specs)
λ_centres = [750]#, 720, 670, 610]
bandpass_nm = 1
λ_step_nm = 1

E_file='/*/*/mcfost/utils/Dust/corundum_crystal_usual.dat'
material='corundum_crystal_usual'

generate_scattering_grids_from_radius_specs(
    E_file, material, radius_specs, nbins_vec, n_samples,
    λ_centres, bandpass_nm, λ_step_nm
)


E_file='/*/*/mcfost/utils/Dust/corundum_crystal_more.dat'
material='corundum_crystal_more'

generate_scattering_grids_from_radius_specs(
    E_file, material, radius_specs, nbins_vec, n_samples,
    λ_centres, bandpass_nm, λ_step_nm
)

E_file='/*/*/mcfost/utils/Dust/corundum_crystal_less.dat'
material='corundum_crystal_less'

generate_scattering_grids_from_radius_specs(
    E_file, material, radius_specs, nbins_vec, n_samples,
    λ_centres, bandpass_nm, λ_step_nm
)

cat = dog

# E_file='/*/*/mcfost/utils/Dust/enstatite_Jaegar_*.dat'
# material='Enstatite'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )
#
#
# #
# E_file='/*/*/mcfost/utils/Dust/Forsterite_wv.dat'
# material='Forsterite'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )


# E_file='/*/*/mcfost/utils/Dust/Al2O3-Jena_up.dat'
# material='Al2O3'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )

# E_file='/*/*/mcfost/utils/Dust/Olivine.dat'
# material='Olivine'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )

# E_file='/*/*/mcfost/utils/Dust/pyroxene_*.dat'
# material='pyroxene'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )

# E_file='/*/*/mcfost/utils/Dust/enstatite_crystal_*.dat'
# material='EnstatiteCrystal'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )
#
# E_file='/*/*/mcfost/utils/Dust/forsterite_crystal_*.dat'
# material='ForsteriteCrystal'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )
#
# E_file='/*/*/mcfost/utils/Dust/spinel_*.dat'
# material='Spinel'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )
#
# E_file='/*/*/mcfost/utils/Dust/silica_*.dat'
# material='Silica'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )
#
# E_file='/*/*/mcfost/utils/Dust/corundum_crystal_*.dat'
# material='CorundumCrystal'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )


# E_file='/*/*/mcfost/utils/Dust/mg60_fe40_pyroxene.dat'
# material='mg60_fe40'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )
#
# E_file='/*/*/mcfost/utils/Dust/mg70_fe30_pyroxene.dat'
# material='mg70_fe30'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )
#
# E_file='/*/*/mcfost/utils/Dust/mg80_fe20_pyroxene.dat'
# material='mg80_fe20'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )

E_file='/*/*/mcfost/utils/Dust/mg95_fe05_pyroxene.dat'
material='mg95_fe05'

generate_scattering_grids_from_radius_specs(
    E_file, material, radius_specs, nbins_vec, n_samples,
    λ_centres, bandpass_nm, λ_step_nm
)

# E_file='/*/*/mcfost/utils/Dust/mg95_fe5_olivine.dat'
# material='mg0.95_fe0.05_olivine'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )
#
# E_file='/*/*/mcfost/utils/Dust/mg0.8_fe0.2_olivine.dat'
# material='mg0.80_fe0.20_olivine'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )
#
# E_file='/*/*/mcfost/utils/Dust/mg0.7_fe0.3_olivine.dat'
# material='mg0.70_fe0.30_olivine'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )

# E_file='/*/*/mcfost/utils/Dust/mg0.6_fe0.4_olivine.dat'
# material='mg0.60_fe0.40_olivine'
#
# generate_scattering_grids_from_radius_specs(
#     E_file, material, radius_specs, nbins_vec, n_samples,
#     λ_centres, bandpass_nm, λ_step_nm
# )
#



#, 'Draine', 'Forsterite', 'Al2O3', 'Olivine', 'pyroxene',
# #'EnstatiteCrystal', 'ForsteriteCrystal',
 #               'Spinel', 'Silica', 'CorundumCrystal',
#               'mg60_fe40', 'mg70_fe30', 'mg80_fe20', 'mg95_fe05',
  #               'mg0.95_fe0.05_olivine', 'mg0.8_fe0.2_olivine', 'mg0.7_fe0.3_olivine', 'mg0.6_fe0.4_olivine',
   #               'FE50', 'FC50', 'EC50', 'FEC33']




















cat = dog








radius_specs = [(1,300,-3), (1,300,-2), (1,300,-1),
                (100,300,-3), (100,300,-2), (100,300,-1),
                (200,300,-3), (200,300,-2), (200,300,-1),
                (1, 1.1, -0.1),
                (5, 5.1, -0.1),
                (10, 10.1, -0.1),
                (25, 25.1, -0.1),
                (50, 50.1, -0.1),
                (100, 100.1, -0.1),
                (150, 150.1, -0.1),
                (200, 200.1, -0.1),
                (250, 250.1, -0.1),
                (300, 300.1, -0.1)]




n_samples = 50000
nbins_vec = [30, 30, 30, 20, 20, 20, 10, 10, 10,
             1,1,1,1,1, 1, 1, 1, 1, 1]
compressed_sets = build_compressed(radius_specs, n_samples, nbins_vec)
save_rad_keys = [f"{a}-{b}-{abs(p)}" for a,b,p in radius_specs]

λ_centres    = [760, 720, 670, 610]        # nm
bandpass_nm  = 50                           # ± half → 25 nm each side
λ_step_nm    = 5                            # sampling


#───────── 4 · optical constants table ───────────────────────────────
E_file = '/*/*/mcfost/utils/Dust/mg0.95_fe0.05_olivine.dat'
λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=1).T;  λ_tab_nm *= 1000
material     = 'mg0.95_fe0.05_olivine'
outdir       = '/import/*1/snert/*/mie_scattering_grids/'
os.makedirs(outdir, exist_ok=True)

# ───────── 5 · main loop with timings ────────────────────────────────
overall_t0 = time.perf_counter()

print('Making Scattering Grids for {}'.format(material))

for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
    set_t0 = time.perf_counter()
    centres_m = centres_nm * 1e-9
    print(f'Processing radius set {key}')

    for λ0 in λ_centres:
        λ_band_nm = np.arange(λ0-bandpass_nm/2,
                              λ0+bandpass_nm/2 + λ_step_nm*0.5,
                              λ_step_nm)
        λ_band_m  = λ_band_nm * 1e-9
        m_band    = m_at_lambda_arr(λ_band_nm)

        t0 = time.perf_counter()
        H,V,H45,V45 = make_scattering_grids(
            m_band, λ_band_m, centres_m, weights)
        t1 = time.perf_counter()

        tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
        np.save(outdir+f"H_scat_{tag}",   H)
        np.save(outdir+f"V_scat_{tag}",   V)
        np.save(outdir+f"H45_scat_{tag}", H45)
        np.save(outdir+f"V45_scat_{tag}", V45)
        print(f"  λ = {λ0}±{bandpass_nm/2} nm "
              f"({λ_band_nm.size} λ) in {t1-t0:5.2f} s")

    set_t1 = time.perf_counter()
    print(f'Radius set {key} finished in {set_t1-set_t0:5.2f} s\n')

overall_t1 = time.perf_counter()
print(f"All scattering grids finished in {(overall_t1-overall_t0)/60:.1f} min.")




#───────── 4 · optical constants table ───────────────────────────────
E_file = '/*/*/mcfost/utils/Dust/mg0.8_fe0.2_olivine.dat'
λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=1).T;  λ_tab_nm *= 1000
material     = 'mg0.8_fe0.2_olivine'
outdir       = '/import/*1/snert/*/mie_scattering_grids/'
os.makedirs(outdir, exist_ok=True)

# ───────── 5 · main loop with timings ────────────────────────────────
overall_t0 = time.perf_counter()

print('Making Scattering Grids for {}'.format(material))

for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
    set_t0 = time.perf_counter()
    centres_m = centres_nm * 1e-9
    print(f'Processing radius set {key}')

    for λ0 in λ_centres:
        λ_band_nm = np.arange(λ0-bandpass_nm/2,
                              λ0+bandpass_nm/2 + λ_step_nm*0.5,
                              λ_step_nm)
        λ_band_m  = λ_band_nm * 1e-9
        m_band    = m_at_lambda_arr(λ_band_nm)

        t0 = time.perf_counter()
        H,V,H45,V45 = make_scattering_grids(
            m_band, λ_band_m, centres_m, weights)
        t1 = time.perf_counter()

        tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
        np.save(outdir+f"H_scat_{tag}",   H)
        np.save(outdir+f"V_scat_{tag}",   V)
        np.save(outdir+f"H45_scat_{tag}", H45)
        np.save(outdir+f"V45_scat_{tag}", V45)
        print(f"  λ = {λ0}±{bandpass_nm/2} nm "
              f"({λ_band_nm.size} λ) in {t1-t0:5.2f} s")

    set_t1 = time.perf_counter()
    print(f'Radius set {key} finished in {set_t1-set_t0:5.2f} s\n')

overall_t1 = time.perf_counter()
print(f"All scattering grids finished in {(overall_t1-overall_t0)/60:.1f} min.")




#───────── 4 · optical constants table ───────────────────────────────
E_file = '/*/*/mcfost/utils/Dust/mg0.7_fe0.3_olivine.dat'
λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=1).T;  λ_tab_nm *= 1000
material     = 'mg0.7_fe0.3_olivine'
outdir       = '/import/*1/snert/*/mie_scattering_grids/'
os.makedirs(outdir, exist_ok=True)

# ───────── 5 · main loop with timings ────────────────────────────────
overall_t0 = time.perf_counter()

print('Making Scattering Grids for {}'.format(material))

for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
    set_t0 = time.perf_counter()
    centres_m = centres_nm * 1e-9
    print(f'Processing radius set {key}')

    for λ0 in λ_centres:
        λ_band_nm = np.arange(λ0-bandpass_nm/2,
                              λ0+bandpass_nm/2 + λ_step_nm*0.5,
                              λ_step_nm)
        λ_band_m  = λ_band_nm * 1e-9
        m_band    = m_at_lambda_arr(λ_band_nm)

        t0 = time.perf_counter()
        H,V,H45,V45 = make_scattering_grids(
            m_band, λ_band_m, centres_m, weights)
        t1 = time.perf_counter()

        tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
        np.save(outdir+f"H_scat_{tag}",   H)
        np.save(outdir+f"V_scat_{tag}",   V)
        np.save(outdir+f"H45_scat_{tag}", H45)
        np.save(outdir+f"V45_scat_{tag}", V45)
        print(f"  λ = {λ0}±{bandpass_nm/2} nm "
              f"({λ_band_nm.size} λ) in {t1-t0:5.2f} s")

    set_t1 = time.perf_counter()
    print(f'Radius set {key} finished in {set_t1-set_t0:5.2f} s\n')

overall_t1 = time.perf_counter()
print(f"All scattering grids finished in {(overall_t1-overall_t0)/60:.1f} min.")





#───────── 4 · optical constants table ───────────────────────────────
E_file = '/*/*/mcfost/utils/Dust/mg0.6_fe0.4_olivine.dat'
λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=1).T;  λ_tab_nm *= 1000
material     = 'mg0.6_fe0.4_olivine'
outdir       = '/import/*1/snert/*/mie_scattering_grids/'
os.makedirs(outdir, exist_ok=True)

# ───────── 5 · main loop with timings ────────────────────────────────
overall_t0 = time.perf_counter()

print('Making Scattering Grids for {}'.format(material))

for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
    set_t0 = time.perf_counter()
    centres_m = centres_nm * 1e-9
    print(f'Processing radius set {key}')

    for λ0 in λ_centres:
        λ_band_nm = np.arange(λ0-bandpass_nm/2,
                              λ0+bandpass_nm/2 + λ_step_nm*0.5,
                              λ_step_nm)
        λ_band_m  = λ_band_nm * 1e-9
        m_band    = m_at_lambda_arr(λ_band_nm)

        t0 = time.perf_counter()
        H,V,H45,V45 = make_scattering_grids(
            m_band, λ_band_m, centres_m, weights)
        t1 = time.perf_counter()

        tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
        np.save(outdir+f"H_scat_{tag}",   H)
        np.save(outdir+f"V_scat_{tag}",   V)
        np.save(outdir+f"H45_scat_{tag}", H45)
        np.save(outdir+f"V45_scat_{tag}", V45)
        print(f"  λ = {λ0}±{bandpass_nm/2} nm "
              f"({λ_band_nm.size} λ) in {t1-t0:5.2f} s")

    set_t1 = time.perf_counter()
    print(f'Radius set {key} finished in {set_t1-set_t0:5.2f} s\n')

overall_t1 = time.perf_counter()
print(f"All scattering grids finished in {(overall_t1-overall_t0)/60:.1f} min.")











#
#
# #───────── 4 · optical constants table ───────────────────────────────
# E_file = '/*/*/mcfost/utils/Dust/enstatite_crystal_*.dat'
# λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=1).T;  λ_tab_nm *= 1000
# material     = 'EnstatiteCrystal'
# outdir       = '/import/*1/snert/*/mie_scattering_grids/'
# os.makedirs(outdir, exist_ok=True)
#
# # ───────── 5 · main loop with timings ────────────────────────────────
# overall_t0 = time.perf_counter()
#
# print('Making Scattering Grids for {}'.format(material))
#
# for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
#     set_t0 = time.perf_counter()
#     centres_m = centres_nm * 1e-9
#     print(f'Processing radius set {key}')
#
#     for λ0 in λ_centres:
#         λ_band_nm = np.arange(λ0-bandpass_nm/2,
#                               λ0+bandpass_nm/2 + λ_step_nm*0.5,
#                               λ_step_nm)
#         λ_band_m  = λ_band_nm * 1e-9
#         m_band    = m_at_lambda_arr(λ_band_nm)
#
#         t0 = time.perf_counter()
#         H,V,H45,V45 = make_scattering_grids(
#             m_band, λ_band_m, centres_m, weights)
#         t1 = time.perf_counter()
#
#         tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
#         np.save(outdir+f"H_scat_{tag}",   H)
#         np.save(outdir+f"V_scat_{tag}",   V)
#         np.save(outdir+f"H45_scat_{tag}", H45)
#         np.save(outdir+f"V45_scat_{tag}", V45)
#         print(f"  λ = {λ0}±{bandpass_nm/2} nm "
#               f"({λ_band_nm.size} λ) in {t1-t0:5.2f} s")
#
#     set_t1 = time.perf_counter()
#     print(f'Radius set {key} finished in {set_t1-set_t0:5.2f} s\n')
#
# overall_t1 = time.perf_counter()
# print(f"All scattering grids finished in {(overall_t1-overall_t0)/60:.1f} min.")
#
#
#
#
# #───────── 4 · optical constants table ───────────────────────────────
# E_file = '/*/*/mcfost/utils/Dust/forsterite_crystal_*.dat'
# λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=1).T;  λ_tab_nm *= 1000
# material     = 'ForsteriteCrystal'
# outdir       = '/import/*1/snert/*/mie_scattering_grids/'
# os.makedirs(outdir, exist_ok=True)
#
# # ───────── 5 · main loop with timings ────────────────────────────────
# overall_t0 = time.perf_counter()
#
# print('Making Scattering Grids for {}'.format(material))
#
# for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
#     set_t0 = time.perf_counter()
#     centres_m = centres_nm * 1e-9
#     print(f'Processing radius set {key}')
#
#     for λ0 in λ_centres:
#         λ_band_nm = np.arange(λ0-bandpass_nm/2,
#                               λ0+bandpass_nm/2 + λ_step_nm*0.5,
#                               λ_step_nm)
#         λ_band_m  = λ_band_nm * 1e-9
#         m_band    = m_at_lambda_arr(λ_band_nm)
#
#         t0 = time.perf_counter()
#         H,V,H45,V45 = make_scattering_grids(
#             m_band, λ_band_m, centres_m, weights)
#         t1 = time.perf_counter()
#
#         tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
#         np.save(outdir+f"H_scat_{tag}",   H)
#         np.save(outdir+f"V_scat_{tag}",   V)
#         np.save(outdir+f"H45_scat_{tag}", H45)
#         np.save(outdir+f"V45_scat_{tag}", V45)
#         print(f"  λ = {λ0}±{bandpass_nm/2} nm "
#               f"({λ_band_nm.size} λ) in {t1-t0:5.2f} s")
#
#     set_t1 = time.perf_counter()
#     print(f'Radius set {key} finished in {set_t1-set_t0:5.2f} s\n')
#
# overall_t1 = time.perf_counter()
# print(f"All scattering grids finished in {(overall_t1-overall_t0)/60:.1f} min.")
#
#
#
#






# ───────── 4 · optical constants table ───────────────────────────────
# E_file = '/*/*/mcfost/utils/Dust/enstatite_Jaegar_*.dat'
# λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=1).T;  λ_tab_nm *= 1000
# material     = 'Enstatite'
# outdir       = '/import/*1/snert/*/mie_scattering_grids/'
# os.makedirs(outdir, exist_ok=True)
#
# # ───────── 5 · main loop with timings ────────────────────────────────
# overall_t0 = time.perf_counter()
#
# print('Making Scattering Grids for {}'.format(material))
#
# for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
#     set_t0 = time.perf_counter()
#     centres_m = centres_nm * 1e-9
#     print(f'Processing radius set {key}')
#
#     for λ0 in λ_centres:
#         λ_band_nm = np.arange(λ0-bandpass_nm/2,
#                               λ0+bandpass_nm/2 + λ_step_nm*0.5,
#                               λ_step_nm)
#         λ_band_m  = λ_band_nm * 1e-9
#         m_band    = m_at_lambda_arr(λ_band_nm)
#
#         # t0 = time.perf_counter()
#         # H,V,H45,V45 = make_scattering_grids(
#         #     m_band, λ_band_m, centres_m, weights)
#         # t1 = time.perf_counter()
#         #
#         # tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
#         # np.save(outdir+f"H_scat_{tag}",   H)
#         # np.save(outdir+f"V_scat_{tag}",   V)
#         # np.save(outdir+f"H45_scat_{tag}", H45)
#         # np.save(outdir+f"V45_scat_{tag}", V45)
#         # print(f"  λ = {λ0}±{bandpass_nm/2} nm "
#         #       f"({λ_band_nm.size} λ) in {t1-t0:5.2f} s")
#
#     set_t1 = time.perf_counter()
#     print(f'Radius set {key} finished in {set_t1-set_t0:5.2f} s\n')
#
# overall_t1 = time.perf_counter()
# print(f"All scattering grids finished in {(overall_t1-overall_t0)/60:.1f} min.")


############################

# # ───────── 4 · optical constants table ───────────────────────────────
# E_file = '/*/*/mcfost/utils/Dust/Al2O3-Jena_up.dat'
# λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=2).T;  λ_tab_nm *= 1000
# material     = 'Al2O3'
# outdir       = '/import/*1/snert/*/mie_scattering_grids/'
# os.makedirs(outdir, exist_ok=True)
#
# # ───────── 5 · main loop with timings ────────────────────────────────
# overall_t0 = time.perf_counter()
#
# print('Making Scattering Grids for {}'.format(material))
# for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
#     set_t0 = time.perf_counter()
#     centres_m = centres_nm * 1e-9
#     print(f'Processing radius set {key}')
#
#     for λ0 in λ_centres:
#         λ_band_nm = np.arange(λ0-bandpass_nm/2,
#                               λ0+bandpass_nm/2 + λ_step_nm*0.5,
#                               λ_step_nm)
#         λ_band_m  = λ_band_nm * 1e-9
#         m_band    = m_at_lambda_arr(λ_band_nm)
#
#         t0 = time.perf_counter()
#         H,V,H45,V45 = make_scattering_grids(
#             m_band, λ_band_m, centres_m, weights)
#         t1 = time.perf_counter()
#
#         tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
#         np.save(outdir+f"H_scat_{tag}",   H)
#         np.save(outdir+f"V_scat_{tag}",   V)
#         np.save(outdir+f"H45_scat_{tag}", H45)
#         np.save(outdir+f"V45_scat_{tag}", V45)
#         print(f"  λ = {λ0}±{bandpass_nm/2} nm "
#               f"({λ_band_nm.size} λ) in {t1-t0:5.2f} s")
#
#     set_t1 = time.perf_counter()
#     print(f'Radius set {key} finished in {set_t1-set_t0:5.2f} s\n')
#
# overall_t1 = time.perf_counter()
# print(f"All scattering grids finished in {(overall_t1-overall_t0)/60:.1f} min.")


############################

# ───────── 4 · optical constants table ───────────────────────────────
# E_file = '/*/*/mcfost/utils/Dust/Forsterite_wv.dat'
# λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=2).T;  λ_tab_nm *= 1000
# material     = 'Forsterite'
# outdir       = '/import/*1/snert/*/mie_scattering_grids/'
# os.makedirs(outdir, exist_ok=True)
#
# # ───────── 5 · main loop with timings ────────────────────────────────
# overall_t0 = time.perf_counter()
#
#
# print('Making Scattering Grids for {}'.format(material))
# for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
#     set_t0 = time.perf_counter()
#     centres_m = centres_nm * 1e-9
#     print(f'Processing radius set {key}')
#
#     for λ0 in λ_centres:
#         λ_band_nm = np.arange(λ0-bandpass_nm/2,
#                               λ0+bandpass_nm/2 + λ_step_nm*0.5,
#                               λ_step_nm)
#         λ_band_m  = λ_band_nm * 1e-9
#         m_band    = m_at_lambda_arr(λ_band_nm)
#
#         t0 = time.perf_counter()
#         H,V,H45,V45 = make_scattering_grids(
#             m_band, λ_band_m, centres_m, weights)
#         t1 = time.perf_counter()
#
#         tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
#         np.save(outdir+f"H_scat_{tag}",   H)
#         np.save(outdir+f"V_scat_{tag}",   V)
#         np.save(outdir+f"H45_scat_{tag}", H45)
#         np.save(outdir+f"V45_scat_{tag}", V45)
#         print(f"  λ = {λ0}±{bandpass_nm/2} nm "
#               f"({λ_band_nm.size} λ) in {t1-t0:5.2f} s")
#
#     set_t1 = time.perf_counter()
#     print(f'Radius set {key} finished in {set_t1-set_t0:5.2f} s\n')
#
# overall_t1 = time.perf_counter()
# print(f"All scattering grids finished in {(overall_t1-overall_t0)/60:.1f} min.")
#
# ############################
#
# # ───────── 4 · optical constants table ───────────────────────────────
# E_file = '/*/*/mcfost/utils/Dust/Draine_Si.dat'
# λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=12).T;  λ_tab_nm *= 1000
# material     = 'Draine'
# outdir       = '/import/*1/snert/*/mie_scattering_grids/'
# os.makedirs(outdir, exist_ok=True)
#
# # ───────── 5 · main loop with timings ────────────────────────────────
# overall_t0 = time.perf_counter()
#
#
# print('Making Scattering Grids for {}'.format(material))
#
# for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
#     set_t0 = time.perf_counter()
#     centres_m = centres_nm * 1e-9
#     print(f'Processing radius set {key}')
#
#     for λ0 in λ_centres:
#         λ_band_nm = np.arange(λ0-bandpass_nm/2,
#                               λ0+bandpass_nm/2 + λ_step_nm*0.5,
#                               λ_step_nm)
#         λ_band_m  = λ_band_nm * 1e-9
#         m_band    = m_at_lambda_arr(λ_band_nm)
#
#         t0 = time.perf_counter()
#         H,V,H45,V45 = make_scattering_grids(
#             m_band, λ_band_m, centres_m, weights)
#         t1 = time.perf_counter()
#
#         tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
#         np.save(outdir+f"H_scat_{tag}",   H)
#         np.save(outdir+f"V_scat_{tag}",   V)
#         np.save(outdir+f"H45_scat_{tag}", H45)
#         np.save(outdir+f"V45_scat_{tag}", V45)
#         print(f"  λ = {λ0}±{bandpass_nm/2} nm "
#               f"({λ_band_nm.size} λ) in {t1-t0:5.2f} s")
#
#     set_t1 = time.perf_counter()
#     print(f'Radius set {key} finished in {set_t1-set_t0:5.2f} s\n')
#
# overall_t1 = time.perf_counter()
# print(f"All scattering grids finished in {(overall_t1-overall_t0)/60:.1f} min.")

############################

# # ───────── 4 · optical constants table ───────────────────────────────
# E_file = '/*/*/mcfost/utils/Dust/Olivine.dat'
# λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=2).T;  λ_tab_nm *= 1000
# material     = 'Olivine'
# outdir       = '/import/*1/snert/*/mie_scattering_grids/'
# os.makedirs(outdir, exist_ok=True)
#
# # ───────── 5 · main loop with timings ────────────────────────────────
# overall_t0 = time.perf_counter()
#
# print('Making Scattering Grids for {}'.format(material))
#
# for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
#     set_t0 = time.perf_counter()
#     centres_m = centres_nm * 1e-9
#     print(f'Processing radius set {key}')
#
#     for λ0 in λ_centres:
#         λ_band_nm = np.arange(λ0-bandpass_nm/2,
#                               λ0+bandpass_nm/2 + λ_step_nm*0.5,
#                               λ_step_nm)
#         λ_band_m  = λ_band_nm * 1e-9
#         m_band    = m_at_lambda_arr(λ_band_nm)
#
#         t0 = time.perf_counter()
#         H,V,H45,V45 = make_scattering_grids(
#             m_band, λ_band_m, centres_m, weights)
#         t1 = time.perf_counter()
#
#         tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
#         np.save(outdir+f"H_scat_{tag}",   H)
#         np.save(outdir+f"V_scat_{tag}",   V)
#         np.save(outdir+f"H45_scat_{tag}", H45)
#         np.save(outdir+f"V45_scat_{tag}", V45)
#         print(f"  λ = {λ0}±{bandpass_nm/2} nm "
#               f"({λ_band_nm.size} λ) in {t1-t0:5.2f} s")
#
#     set_t1 = time.perf_counter()
#     print(f'Radius set {key} finished in {set_t1-set_t0:5.2f} s\n')
#
# overall_t1 = time.perf_counter()
# print(f"All scattering grids finished in {(overall_t1-overall_t0)/60:.1f} min.")
#


# ───────── 4 · optical constants table ───────────────────────────────
E_file = '/*/*/mcfost/utils/Dust/pyroxene_*.dat'
λ_tab_nm, n_tab, k_tab = np.loadtxt(E_file, skiprows=2).T;  λ_tab_nm *= 1000
material     = 'pyroxene'
outdir       = '/import/*1/snert/*/mie_scattering_grids/'
os.makedirs(outdir, exist_ok=True)

# ───────── 5 · main loop with timings ────────────────────────────────
overall_t0 = time.perf_counter()

print('Making Scattering Grids for {}'.format(material))

for key, (centres_nm, weights) in zip(save_rad_keys, compressed_sets):
    set_t0 = time.perf_counter()
    centres_m = centres_nm * 1e-9
    print(f'Processing radius set {key}')

    for λ0 in λ_centres:
        λ_band_nm = np.arange(λ0-bandpass_nm/2,
                              λ0+bandpass_nm/2 + λ_step_nm*0.5,
                              λ_step_nm)
        λ_band_m  = λ_band_nm * 1e-9
        m_band    = m_at_lambda_arr(λ_band_nm)

        t0 = time.perf_counter()
        H,V,H45,V45 = make_scattering_grids(
            m_band, λ_band_m, centres_m, weights)
        t1 = time.perf_counter()

        tag = f"m{material}_r{key}_w{λ0}_band{int(bandpass_nm)}"
        np.save(outdir+f"H_scat_{tag}",   H)
        np.save(outdir+f"V_scat_{tag}",   V)
        np.save(outdir+f"H45_scat_{tag}", H45)
        np.save(outdir+f"V45_scat_{tag}", V45)
        print(f"  λ = {λ0}±{bandpass_nm/2} nm "
              f"({λ_band_nm.size} λ) in {t1-t0:5.2f} s")

    set_t1 = time.perf_counter()
    print(f'Radius set {key} finished in {set_t1-set_t0:5.2f} s\n')

overall_t1 = time.perf_counter()
print(f"All scattering grids finished in {(overall_t1-overall_t0)/60:.1f} min.")

