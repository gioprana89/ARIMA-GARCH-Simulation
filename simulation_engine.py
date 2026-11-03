"""Simulation and conditional-MLE engine for an ARIMA(1,d,q)-GARCH(1,1) lab.

Supported d, q = 0 or 1, AR order fixed at 1.
No external data are required. All random numbers come from explicit RNG seeds.

The benchmark is ARIMA(1,d,q) with constant conditional innovation variance.
The alternative jointly estimates an ARIMA(1,d,q) mean and GARCH(1,1)
conditional variance, using either Gaussian or standardized Student-t innovations.
Both likelihoods condition on the first transformed observation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.special import gammaln, softmax
from scipy.stats import chi2, jarque_bera, norm, t as student_t
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from statsmodels.tsa.stattools import adfuller, acf, pacf

EPS = 1e-10
SCENARIOS = {
    "Homoskedastisitas": {"alpha": 0.0, "beta": 0.0, "innovation": "Normal", "break_factor": 1.0},
    "Persistensi rendah": {"alpha": 0.10, "beta": 0.40, "innovation": "Normal", "break_factor": 1.0},
    "Persistensi sedang": {"alpha": 0.10, "beta": 0.75, "innovation": "Normal", "break_factor": 1.0},
    "Persistensi tinggi": {"alpha": 0.08, "beta": 0.90, "innovation": "Normal", "break_factor": 1.0},
    "Heavy-tail Student-t": {"alpha": 0.10, "beta": 0.75, "innovation": "Student-t", "break_factor": 1.0},
    "Break struktural varians": {"alpha": 0.10, "beta": 0.75, "innovation": "Normal", "break_factor": 4.0},
}


@dataclass(frozen=True)
class SimulationConfig:
    n: int = 500
    d: int = 1
    q: int = 0
    mu: float = 0.0
    phi: float = 0.35
    theta: float = 0.0
    variance_target: float = 1.0
    alpha: float = 0.08
    beta: float = 0.90
    innovation: str = "Normal"
    df: float = 6.0
    break_factor: float = 1.0
    break_at: float = 0.85
    training_fraction: float = 0.80
    seed: int = 2026
    burn_in: int = 350
    level_start: float = 100.0


def config_from_scenario(name: str, n: int = 500, seed: int = 2026, **kwargs: Any) -> SimulationConfig:
    if name not in SCENARIOS:
        raise ValueError(f"Skenario tidak dikenal: {name}")
    values = {**SCENARIOS[name], **kwargs}
    return SimulationConfig(n=n, seed=seed, **values)


def validate_config(cfg: SimulationConfig) -> None:
    if cfg.n < 80:
        raise ValueError("Jumlah observasi minimal 80.")
    if cfg.d not in (0, 1) or cfg.q not in (0, 1):
        raise ValueError("Aplikasi hanya mendukung ARIMA(1,d,q), d dan q = 0 atau 1.")
    if not (-0.98 < cfg.phi < 0.98):
        raise ValueError("Parameter AR harus berada pada interval (-0,98; 0,98).")
    if cfg.q == 1 and not (-0.98 < cfg.theta < 0.98):
        raise ValueError("Parameter MA harus berada pada interval (-0,98; 0,98).")
    if cfg.variance_target <= 0:
        raise ValueError("Target varians harus positif.")
    if cfg.alpha < 0 or cfg.beta < 0 or cfg.alpha + cfg.beta >= 1:
        raise ValueError("Syarat GARCH: alpha >= 0, beta >= 0, dan alpha+beta < 1.")
    if cfg.innovation not in ("Normal", "Student-t"):
        raise ValueError("Distribusi inovasi tidak dikenali.")
    if cfg.innovation == "Student-t" and cfg.df <= 2:
        raise ValueError("Derajat bebas Student-t harus > 2 agar varians berhingga.")
    if cfg.break_factor < 1:
        raise ValueError("Faktor perubahan varians minimal 1.")
    if not (0.1 < cfg.break_at < 0.9):
        raise ValueError("Lokasi break harus antara 10% dan 90% data.")
    if not (0.5 <= cfg.training_fraction <= 0.95):
        raise ValueError("Proporsi latih harus antara 50% dan 95%.")
    if cfg.burn_in < 0:
        raise ValueError("Burn-in tidak boleh negatif.")


def simulate(cfg: SimulationConfig) -> pd.DataFrame:
    """Simulate observed y, transformed w, true eps, h, z and train-test split.

    Parameter omega = variance_target * (1-alpha-beta), so unconditional
    variance is held constant across stationary homogeneous regimes.
    The structural-break scenario multiplies omega after the break, and is
    *intentionally* outside a single constant-parameter GARCH DGP.
    """
    validate_config(cfg)
    rng = np.random.default_rng(int(cfg.seed))
    total = cfg.n + cfg.burn_in
    if cfg.innovation == "Normal":
        z = rng.standard_normal(total)
    else:
        z = rng.standard_t(cfg.df, size=total) * np.sqrt((cfg.df - 2.0) / cfg.df)
    h = np.empty(total, dtype=float)
    eps = np.empty(total, dtype=float)
    w = np.empty(total, dtype=float)
    regime = np.zeros(total, dtype=int)
    omega_series = np.empty(total, dtype=float)
    omega_base = cfg.variance_target * (1.0 - cfg.alpha - cfg.beta)
    break_index = cfg.burn_in + int(np.floor(cfg.break_at * cfg.n))
    intercept = (1.0 - cfg.phi) * cfg.mu
    prev_h = cfg.variance_target
    prev_eps = 0.0
    prev_w = cfg.mu
    for k in range(total):
        is_post_break = cfg.break_factor > 1.0 and k >= break_index
        omega_k = omega_base * (cfg.break_factor if is_post_break else 1.0)
        omega_series[k] = omega_k
        h[k] = omega_k + cfg.alpha * prev_eps**2 + cfg.beta * prev_h
        eps[k] = np.sqrt(h[k]) * z[k]
        w[k] = intercept + cfg.phi * prev_w + (cfg.theta * prev_eps if cfg.q else 0.0) + eps[k]
        regime[k] = int(is_post_break)
        prev_h, prev_eps, prev_w = h[k], eps[k], w[k]
    w = w[cfg.burn_in:]
    if cfg.d == 1:
        y = cfg.level_start + np.cumsum(w)
        y_lag1 = np.concatenate([[cfg.level_start], y[:-1]])
    else:
        y = w.copy()
        y_lag1 = np.full(cfg.n, np.nan)
    split = int(np.floor(cfg.n * cfg.training_fraction))
    df = pd.DataFrame({
        "t": np.arange(1, cfg.n + 1, dtype=int),
        "Y_t": y,
        "Y_lag1": y_lag1,
        "W_t": w,
        "mean_true": w - eps[cfg.burn_in:],
        "omega_t": omega_series[cfg.burn_in:],
        "Z_t": z[cfg.burn_in:],
        "epsilon_true": eps[cfg.burn_in:],
        "h_true": h[cfg.burn_in:],
        "sigma_true": np.sqrt(h[cfg.burn_in:]),
        "regime": regime[cfg.burn_in:],
        "partisi": np.where(np.arange(cfg.n) < split, "Training", "Testing"),
    })
    return df


def residuals_arima(w: np.ndarray, intercept: float, phi: float, theta: float = 0.0) -> np.ndarray:
    """Conditional innovations, with initial residual fixed to zero."""
    w = np.asarray(w, dtype=float)
    raw = np.empty_like(w)
    raw[0] = 0.0
    raw[1:] = w[1:] - intercept - phi * w[:-1]
    if abs(theta) < 1e-13:
        return raw
    return lfilter([1.0], [1.0, theta], raw)


def conditional_variance(eps: np.ndarray, omega: float, alpha: float, beta: float, h0: float) -> np.ndarray:
    """h[t] uses eps[t-1] and h[t-1], never the current eps[t]."""
    eps = np.asarray(eps, dtype=float)
    h = np.empty_like(eps)
    h[0] = max(float(h0), EPS)
    if len(eps) > 1:
        source = omega + alpha * eps[:-1] ** 2
        h[1:], _ = lfilter([1.0], [1.0, -beta], source, zi=[beta * h[0]])
    return h


def _norm_loss(eps: np.ndarray, variances: np.ndarray) -> float:
    v = np.maximum(variances[1:], EPS)
    e = eps[1:]
    if np.any(~np.isfinite(v)) or np.any(~np.isfinite(e)):
        return 1e15
    result = 0.5 * np.sum(np.log(2 * np.pi * v) + e**2 / v)
    return float(result) if np.isfinite(result) else 1e15


def _student_loss(eps: np.ndarray, variances: np.ndarray, nu: float) -> float:
    v = np.maximum(variances[1:], EPS)
    e2v = eps[1:]**2 / v
    if np.any(~np.isfinite(v)) or np.any(~np.isfinite(e2v)):
        return 1e15
    const = gammaln((nu + 1) / 2) - gammaln(nu / 2) - .5 * np.log(np.pi * (nu - 2))
    log_pdf = const - .5 * np.log(v) - .5 * (nu + 1) * np.log1p(e2v / (nu - 2))
    result = -np.sum(log_pdf)
    return float(result) if np.isfinite(result) else 1e15


def _base_initial(w: np.ndarray) -> tuple[float, float, float, float]:
    x = np.column_stack([np.ones(len(w) - 1), w[:-1]])
    coeff = np.linalg.lstsq(x, w[1:], rcond=None)[0]
    c0 = float(coeff[0])
    p0 = float(np.clip(coeff[1], -0.979, 0.979))
    eps = w[1:] - (c0 + p0*w[:-1])
    v0 = max(float(np.mean(eps**2)), 1e-6)
    return c0, p0, 0.0, v0


def fit_arima_baseline(w_train: np.ndarray, q: int = 0, maxiter: int = 120) -> dict[str, Any]:
    """Conditional Gaussian MLE for ARIMA(1,d,q) fitted on stationary W_t.

    For q=0 the conditional likelihood maximizer is the AR(1) OLS solution.
    For q=1 the ARMA conditional likelihood is optimized numerically.
    """
    w = np.asarray(w_train, float)
    c0, phi0, theta0, var0 = _base_initial(w)
    sd = max(float(w.std()), 0.01)
    if q == 0:
        eps = residuals_arima(w, c0, phi0)
        v = max(float(np.mean(eps[1:] ** 2)), EPS)
        nll = _norm_loss(eps, np.full(len(w), v))
        return {"success": True, "message": "MLE kondisional AR(1), solusi OLS", "c": c0, "phi": phi0,
                "theta": 0.0, "variance": v, "nll": nll, "aic": 2*3+2*nll, "bic": 3*np.log(len(w)-1)+2*nll, "k": 3}

    def objective(p: np.ndarray) -> float:
        c, phi, theta, logv = p
        eps = residuals_arima(w, c, phi, theta)
        return _norm_loss(eps, np.full(len(w), np.exp(logv)))

    bounds = [(-10 * sd - 5, 10 * sd + 5), (-0.979, .979), (-.979, .979),
              (np.log(max(var0*1e-5, 1e-8)), np.log(var0*1e3 + 1))]
    result = minimize(objective, [c0, phi0, theta0, np.log(var0)], bounds=bounds,
                      method="L-BFGS-B", options={"maxiter": maxiter, "ftol": 1e-8})
    c, phi, theta, logv = result.x
    nll = float(result.fun)
    k = 4
    return {"success": bool(result.success and np.isfinite(nll)), "message": str(result.message),
            "c": float(c), "phi": float(phi), "theta": float(theta), "variance": float(np.exp(logv)),
            "nll": nll, "aic": 2*k+2*nll, "bic": k*np.log(len(w)-1)+2*nll, "k": k}


def _to_raw_ab(alpha: float, beta: float) -> tuple[float, float]:
    rest = 0.999 - alpha - beta
    return float(np.log(alpha / rest)), float(np.log(beta / rest))


def _from_raw_ab(raw_a: float, raw_b: float) -> tuple[float, float]:
    s = softmax([0, raw_a, raw_b])
    return float(.999 * s[1]), float(.999 * s[2])


def fit_arima_garch(w_train: np.ndarray, q: int = 0, distribution: str = "Normal", maxiter: int = 180) -> dict[str, Any]:
    """Joint conditional likelihood for ARIMA(1,d,q)-GARCH(1,1)."""
    if distribution not in ("Normal", "Student-t"):
        raise ValueError("Distribusi estimasi harus Normal atau Student-t")
    w = np.asarray(w_train, float)
    base = fit_arima_baseline(w, q=q, maxiter=maxiter)
    if not base["success"]:
        # Still use finite conditional-mean estimates as optimizer initial values.
        pass
    v0 = max(float(base["variance"]), 1e-5)
    sd = max(float(w.std()), .01)
    h0 = max(float(np.var(w)), 1e-5)
    alpha0, beta0 = .08, .85
    ra, rb = _to_raw_ab(alpha0, beta0)
    start = [base["c"], base["phi"]]
    if q:
        start.append(base["theta"])
    start += [np.log(max(v0 * (1-alpha0-beta0), 1e-7)), ra, rb]
    if distribution == "Student-t":
        start.append(np.log(8.0 - 2.0))

    bounds = [(-10 * sd - 5, 10 * sd + 5), (-.979, .979)]
    if q:
        bounds.append((-.979, .979))
    bounds += [(np.log(max(v0 * 1e-6, 1e-9)), np.log(v0 * 10 + 1)), (-10, 10), (-10, 10)]
    if distribution == "Student-t":
        bounds.append((np.log(.15), np.log(100-2)))

    def unpack(raw: np.ndarray):
        c, phi = float(raw[0]), float(raw[1])
        ix = 2
        theta = float(raw[ix]) if q else 0.0
        ix += int(q)
        omega = float(np.exp(raw[ix]))
        alpha, beta = _from_raw_ab(float(raw[ix+1]), float(raw[ix+2]))
        nu = float(2 + np.exp(raw[ix+3])) if distribution == "Student-t" else np.nan
        return c, phi, theta, omega, alpha, beta, nu

    def objective(raw: np.ndarray) -> float:
        c, phi, theta, omega, alpha, beta, nu = unpack(raw)
        eps = residuals_arima(w, c, phi, theta)
        h = conditional_variance(eps, omega, alpha, beta, h0)
        if np.any(h[1:] <= 0):
            return 1e15
        return _student_loss(eps, h, nu) if distribution == "Student-t" else _norm_loss(eps, h)

    result = minimize(objective, start, method="L-BFGS-B", bounds=bounds,
                      options={"maxiter": maxiter, "ftol": 2e-8, "maxls": 30})
    if not result.success or not np.isfinite(result.fun):
        # Retry a lower-persistence initialization, useful for flat likelihoods.
        ra2, rb2 = _to_raw_ab(.16, .55)
        start2 = np.array(start)
        raw_offset = 2 + int(q)
        start2[raw_offset] = np.log(max(v0 * .29, 1e-7))
        start2[raw_offset+1] = ra2
        start2[raw_offset+2] = rb2
        result2 = minimize(objective, start2, method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": maxiter, "ftol": 2e-8, "maxls": 30})
        if (result2.success and not result.success) or (result2.fun < result.fun):
            result = result2
    c, phi, theta, omega, alpha, beta, nu = unpack(result.x)
    nll = float(result.fun)
    k = len(start)
    return {"success": bool(result.success and np.isfinite(nll)), "message": str(result.message),
            "c": c, "phi": phi, "theta": theta, "omega": omega, "alpha": alpha, "beta": beta,
            "persistence": alpha+beta, "nu": nu, "distribution": distribution,
            "nll": nll, "aic": 2*k+2*nll, "bic": k*np.log(len(w)-1)+2*nll, "k": k}


def predict_fixed_parameters(w_all: np.ndarray, fitted: dict, model: str, h0: float) -> pd.DataFrame:
    """One-step-ahead recursive predictions with parameters frozen after training.

    At t the forecast uses w up to t-1 only. Parameters were trained on
    the training data, never updated with test observations.
    """
    w_all = np.asarray(w_all, dtype=float)
    eps = residuals_arima(w_all, fitted["c"], fitted["phi"], fitted["theta"])
    means = np.full(len(w_all), np.nan, float)
    means[1:] = w_all[1:] - eps[1:]
    if model == "ARIMA":
        h = np.full(len(w_all), float(fitted["variance"]))
    else:
        h = conditional_variance(eps, fitted["omega"], fitted["alpha"], fitted["beta"], h0)
    return pd.DataFrame({"pred_mean_w": means, "pred_var": h, "residual": eps,
                         "z_standard": eps/np.sqrt(np.maximum(h, EPS))})


def _interval_quantile(distribution: str, nu: float = np.nan) -> float:
    if distribution == "Student-t" and np.isfinite(nu) and nu > 2:
        return float(student_t.ppf(.975, nu) * np.sqrt((nu-2)/nu))
    return float(norm.ppf(.975))


def _metrics(actual_mean: np.ndarray, prediction: pd.DataFrame, truth_var: np.ndarray,
             split: int, distribution: str = "Normal", nu: float = np.nan) -> dict[str, float]:
    p = prediction.iloc[split:]
    h_true = np.asarray(truth_var[split:], dtype=float)
    h_hat = np.maximum(p["pred_var"].to_numpy(float), EPS)
    mean_hat = p["pred_mean_w"].to_numpy(float)
    w_true = actual_mean[split:]
    errors = w_true - mean_hat
    zcrit = _interval_quantile(distribution, nu)
    lower = mean_hat - zcrit*np.sqrt(h_hat)
    upper = mean_hat + zcrit*np.sqrt(h_hat)
    return {
        "mse_var": float(np.mean((h_true-h_hat)**2)),
        "rmse_var": float(np.sqrt(np.mean((h_true-h_hat)**2))),
        "qlike": float(np.mean(np.log(h_hat)+h_true/h_hat)),
        "rmse_mean": float(np.sqrt(np.mean(errors**2))),
        "mae_mean": float(np.mean(np.abs(errors))),
        "coverage95": float(np.mean((w_true >= lower) & (w_true <= upper))),
        "interval_width95": float(np.mean(upper-lower)),
    }


def _safe_diagnostics(residuals: np.ndarray, variances: np.ndarray) -> dict[str, float]:
    z = np.asarray(residuals, float)[1:] / np.sqrt(np.maximum(np.asarray(variances, float)[1:], EPS))
    z = z[np.isfinite(z)]
    out = {"ljungbox_p": np.nan, "ljungbox_sq_p": np.nan, "arch_lm_p": np.nan, "jarque_bera_p": np.nan}
    if len(z) < 40:
        return out
    lag = min(10, len(z)//5)
    try:
        out["ljungbox_p"] = float(acorr_ljungbox(z, lags=[lag], return_df=True)["lb_pvalue"].iloc[0])
        out["ljungbox_sq_p"] = float(acorr_ljungbox(z**2, lags=[lag], return_df=True)["lb_pvalue"].iloc[0])
        out["arch_lm_p"] = float(het_arch(z, nlags=min(5, lag))[1])
        out["jarque_bera_p"] = float(jarque_bera(z).pvalue)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        pass
    return out


def evaluate(df: pd.DataFrame, cfg: SimulationConfig, estimated_distribution: str = "Normal",
             maxiter: int = 180) -> dict[str, Any]:
    """Fit each model once on training data, forecast test data sequentially."""
    w = df["W_t"].to_numpy(float)
    split = int(np.floor(cfg.n * cfg.training_fraction))
    base = fit_arima_baseline(w[:split], q=cfg.q, maxiter=maxiter)
    garch = fit_arima_garch(w[:split], q=cfg.q, distribution=estimated_distribution, maxiter=maxiter)
    result: dict[str, Any] = {"baseline": base, "garch": garch, "split": split}
    h0 = max(float(np.var(w[:split])), 1e-5)
    for key, fit in [("baseline", base), ("garch", garch)]:
        if fit["success"]:
            pred = predict_fixed_parameters(w, fit, "ARIMA" if key == "baseline" else "GARCH", h0)
            metric = _metrics(w, pred, df["h_true"].to_numpy(float), split,
                              distribution="Normal" if key == "baseline" else estimated_distribution,
                              nu=fit.get("nu", np.nan))
            diag = _safe_diagnostics(pred["residual"].to_numpy(float)[:split], pred["pred_var"].to_numpy(float)[:split])
            result[key+"_pred"] = pred
            result[key+"_metric"] = metric
            result[key+"_diagnostics"] = diag
        else:
            result[key+"_pred"] = None
            result[key+"_metric"] = {k: np.nan for k in ["mse_var", "rmse_var", "qlike", "rmse_mean", "mae_mean", "coverage95", "interval_width95"]}
            result[key+"_diagnostics"] = {k: np.nan for k in ["ljungbox_p", "ljungbox_sq_p", "arch_lm_p", "jarque_bera_p"]}
    return result


def one_replication(cfg: SimulationConfig, scenario: str, replication: int,
                    estimated_distribution: str = "Normal", maxiter: int = 120) -> dict[str, Any]:
    """Run one Monte Carlo unit. Failed fits are recorded, not silently dropped."""
    row: dict[str, Any] = {"scenario": scenario, "replication": replication, "seed": cfg.seed,
                           "n": cfg.n, "alpha_true": cfg.alpha, "beta_true": cfg.beta,
                           "persistence_true": cfg.alpha+cfg.beta,
                           "omega_true": cfg.variance_target*(1-cfg.alpha-cfg.beta),
                           "innovation_true": cfg.innovation, "break_factor": cfg.break_factor,
                           "d": cfg.d, "q": cfg.q, "estimated_distribution": estimated_distribution}
    try:
        df = simulate(cfg)
        res = evaluate(df, cfg, estimated_distribution=estimated_distribution, maxiter=maxiter)
        for key, prefix in [("baseline", "arima"), ("garch", "garch")]:
            f = res[key]
            row[prefix+"_success"] = bool(f["success"])
            row[prefix+"_message"] = f["message"]
            row[prefix+"_aic"] = f.get("aic", np.nan)
            for name, val in res[key+"_metric"].items():
                row[prefix+"_"+name] = val
        fit = res["garch"]
        for name in ("alpha", "beta", "omega", "phi", "theta", "persistence", "nu"):
            row[name+"_estimated"] = fit.get(name, np.nan) if fit["success"] else np.nan
        row["alpha_error"] = row["alpha_estimated"] - cfg.alpha
        row["beta_error"] = row["beta_estimated"] - cfg.beta
        row["garch_better_qlike"] = bool(row["garch_qlike"] < row["arima_qlike"]) if (row["garch_success"] and row["arima_success"]) else np.nan
        row["qlike_gain"] = row["arima_qlike"] - row["garch_qlike"]
        row["error"] = ""
    except Exception as e:
        row.update({"arima_success": False, "garch_success": False, "error": f"{type(e).__name__}: {e}"})
    return row


def summarize_monte_carlo(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate successes and metrics; report denominator to avoid selection bias."""
    if raw.empty:
        return pd.DataFrame()
    output = []
    for (scenario, n), group in raw.groupby(["scenario", "n"], sort=False):
        success = group.get("garch_success", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        both = success & group.get("arima_success", pd.Series(False, index=group.index)).fillna(False).astype(bool)
        subs = group.loc[both]
        valid_better = pd.to_numeric(subs.get("garch_better_qlike", pd.Series(dtype=float)), errors="coerce")
        output.append({
            "Skenario": scenario, "n": int(n), "Jumlah replikasi": len(group),
            "GARCH konvergen (%)": 100 * float(success.mean()),
            "Dua model valid": int(both.sum()),
            "ARIMA QLIKE rata-rata": float(pd.to_numeric(subs.get("arima_qlike", pd.Series(dtype=float)), errors="coerce").mean()),
            "GARCH QLIKE rata-rata": float(pd.to_numeric(subs.get("garch_qlike", pd.Series(dtype=float)), errors="coerce").mean()),
            "Keuntungan QLIKE rata-rata": float(pd.to_numeric(subs.get("qlike_gain", pd.Series(dtype=float)), errors="coerce").mean()),
            "GARCH unggul QLIKE (%)": 100 * float(valid_better.mean()) if len(valid_better) else np.nan,
            "GARCH RMSE varians rata-rata": float(pd.to_numeric(subs.get("garch_rmse_var", pd.Series(dtype=float)), errors="coerce").mean()),
            "GARCH cakupan 95% (%)": 100 * float(pd.to_numeric(subs.get("garch_coverage95", pd.Series(dtype=float)), errors="coerce").mean()),
            "Bias alpha": float(pd.to_numeric(subs.get("alpha_error", pd.Series(dtype=float)), errors="coerce").mean()),
            "Bias beta": float(pd.to_numeric(subs.get("beta_error", pd.Series(dtype=float)), errors="coerce").mean()),
        })
    return pd.DataFrame(output)


def run_monte_carlo(scenarios: list[str], sizes: list[int], replications: int,
                    base_config: SimulationConfig, estimated_distribution: str = "Normal",
                    progress_callback=None, maxiter: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not scenarios or not sizes or replications < 1:
        raise ValueError("Skenario, ukuran sampel, dan replikasi wajib diisi.")
    n_total = len(scenarios)*len(sizes)*replications
    records = []
    count = 0
    for si, name in enumerate(scenarios):
        if name not in SCENARIOS:
            raise ValueError("Skenario tidak dikenal")
        for n in sizes:
            for rep in range(1, replications+1):
                seed = int(np.random.SeedSequence([int(base_config.seed), si, int(n), rep]).generate_state(1)[0])
                cfg = replace(base_config, n=int(n), seed=seed, **SCENARIOS[name])
                records.append(one_replication(cfg, name, rep, estimated_distribution, maxiter))
                count += 1
                if progress_callback is not None:
                    progress_callback(count, n_total, name, n)
    raw = pd.DataFrame(records)
    return raw, summarize_monte_carlo(raw)


def test_stationarity(w: np.ndarray) -> dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = adfuller(w, autolag="AIC")
    return {"ADF statistic": float(r[0]), "p-value": float(r[1]), "lag": int(r[2])}


def config_as_dict(cfg: SimulationConfig) -> dict[str, Any]:
    return asdict(cfg)
