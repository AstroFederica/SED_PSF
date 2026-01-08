import os, glob
import numpy as np

# -----------------------------
# 1) Carica i template COSMOS
# -----------------------------
def load_cosmos_templates(path="galaxy_seds"):
    templates = {}
    for fname in sorted(glob.glob(os.path.join(path, "*.csv"))):
        base = os.path.basename(fname) 
        idx = int(base.split("_")[0])   # "00_Ell1..." -> 0
        data = np.loadtxt(fname, delimiter=",")
        wave = data[:, 0]      # [Angstrom]
        flux = data[:, 1]      # F_lambda 
        templates[idx] = (wave, flux)
    return templates


def load_extinction_laws(path="galaxy_extincts", ebv_ref=0.2):
    # law 0 = no extinction
    noext_file = os.path.join(path, "0_flatnuspec_noext.csv")
    wave0, F0 = np.loadtxt(noext_file, delimiter=",").T

    laws = {}
    laws[0] = (wave0, np.zeros_like(F0))  # k(λ)=0

    for law_id in [1, 2, 3, 4]:
        fname = os.path.join(path, f"{law_id}_flatnuspec_*.csv")
        matches = glob.glob(fname)
        if not matches:
            raise FileNotFoundError(f"Extinction file for law {law_id} not found.")
        wave, Fext = np.loadtxt(matches[0], delimiter=",").T
        # interpolate F0 over the same grid, if needed
        F0i = np.interp(wave, wave0, F0)
        # A_lambda = -2.5 log10(F_ext/F_noext)  (per ebv_ref)
        A_lambda = -2.5 * np.log10(Fext / F0i)
        k_lambda = A_lambda / ebv_ref
        laws[law_id] = (wave, k_lambda)

    return laws


def load_filters(path="filters"):
    filters = {}
    for fname in glob.glob(os.path.join(path, "*.csv")):
        base = os.path.basename(fname)
        name = os.path.splitext(base)[0]   # es. "euclid_vis", "lsst_r"
        data = np.loadtxt(fname, delimiter=",")
        wave = data[:, 0]
        T = data[:, 1]
        filters[name] = (wave, T)
    return filters
