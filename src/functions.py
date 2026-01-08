import os, glob
import numpy as np

# 2) k(λ) for extinction laws
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

def apply_internal_extinction(wave, F_lambda, law_id, ebv, laws):
    if ebv == 0.0 or law_id == 0:
        return F_lambda.copy()
    w_law, k_law = laws[law_id]
    k = np.interp(wave, w_law, k_law, left=k_law[0], right=k_law[-1])
    A_lambda = k * ebv
    return F_lambda * 10**(-0.4 * A_lambda)


# -----------------------------
# 3) MW extinction
# -----------------------------
def apply_mw_extinction(wave, F_lambda, A_V):
    if A_V == 0.0:
        return F_lambda.copy()
    # Cardelli-like approximation, as above
    w = wave / 1e4
    k = 3.1 * (0.574 * w**(-1.61))
    A_lambda = k * (A_V / 3.1)
    return F_lambda * 10**(-0.4 * A_lambda)


# --------
# 4) SED
# --------
def build_flagship_sed(row, templates, ext_laws):
    z = float(row["observed_redshift_gal"])
    A_MW = float(row["mw_extinction"])

    i1 = int(row["sed_cosmos_1"])
    i2 = int(row["sed_cosmos_2"])
    law1 = int(row["ext_curve_cosmos_1"])
    law2 = int(row["ext_curve_cosmos_2"])
    ebv1 = float(row["ebv_cosmos_1"])
    ebv2 = float(row["ebv_cosmos_2"])
    f1 = float(row["frac_cosmos_1"])
    f2 = 1.0 - f1

    # template rest-frame
    wave1, F1 = templates[i1]
    wave2, F2 = templates[i2]
    wave = np.union1d(wave1, wave2)
    F1i = np.interp(wave, wave1, F1)
    F2i = np.interp(wave, wave2, F2)

    # internal extinction
    F1_att = apply_internal_extinction(wave, F1i, law1, ebv1, ext_laws)
    F2_att = apply_internal_extinction(wave, F2i, law2, ebv2, ext_laws)

    # mix of the two components
    F_rest = f1 * F1_att + f2 * F2_att   # F_lambda rest-frame

    # redshift all’osservato:
    wave_obs = wave * (1.0 + z)
    F_obs = F_rest / (1.0 + z)           # conservation of F_nu

    # Milky Way extinction
    F_obs = apply_mw_extinction(wave_obs, F_obs, A_MW)

    return wave_obs, F_obs


# Additional steps
def synth_photometry(wave, F_lambda, filt_wave, filt_T):
    T_interp = np.interp(wave, filt_wave, filt_T, left=0, right=0)
    num = np.trapz(F_lambda * T_interp, wave)
    den = np.trapz(T_interp, wave)
    return num/den if den > 0 else np.nan

rng = np.random.default_rng(123)

def make_noisy_realizations(wave_obs, F_obs, frac_sigma=0.05, n_real=10):
    sed_realizations = []
    for _ in range(n_real):
        noise = rng.normal(0.0, frac_sigma, size=F_obs.size)
        F_pert = F_obs * (1.0 + noise)
        sed_realizations.append((wave_obs.copy(), F_pert))
    return sed_realizations



# --------
# 4) SED TEST
# --------
def build_flagship_sed_test(row, templates, ext_laws):
    z = float(row["observed_redshift_gal"])
    A_MW = float(row["mw_extinction"])

    i1 = int(row["sed_cosmos_1"])
    i2 = int(row["sed_cosmos_2"])
    law1 = int(row["ext_curve_cosmos_1"])
    law2 = int(row["ext_curve_cosmos_2"])
    ebv1 = float(row["ebv_cosmos_1"])
    ebv2 = float(row["ebv_cosmos_2"])
    f1 = float(row["frac_cosmos_1"])
    f2 = 1.0 - f1

    # template rest-frame
    wave1, F1 = templates[i1]
    wave2, F2 = templates[i2]
    wave = np.union1d(wave1, wave2)
    F1i = np.interp(wave, wave1, F1)
    F2i = np.interp(wave, wave2, F2)

    # internal extinction
    F1_att = apply_internal_extinction(wave, F1i, law1, ebv1, ext_laws)
    F2_att = apply_internal_extinction(wave, F2i, law2, ebv2, ext_laws)

    # mix of the two components
    F_rest = f1 * F1_att + f2 * F2_att   # F_lambda rest-frame

    # redshift, observed:
    wave_obs = wave * (1.0 + z)
    F_obs = F_rest / (1.0 + z)           # conservazione di F_nu

    # Milky Way extinction
    F_obs = apply_mw_extinction(wave_obs, F_obs, A_MW)

    return wave_obs, F_obs, F1i, F2i, F1_att, F2_att, F_rest, wave
