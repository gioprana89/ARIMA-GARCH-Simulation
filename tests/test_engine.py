"""Scientific reproducibility and no-lookahead tests."""
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest

from simulation_engine import (
    SimulationConfig, SCENARIOS, config_from_scenario, simulate,
    validate_config, residuals_arima, conditional_variance,
    fit_arima_baseline, fit_arima_garch, predict_fixed_parameters,
    evaluate, one_replication, summarize_monte_carlo, run_monte_carlo,
)


def test_same_seed_exact_data():
    cfg = SimulationConfig(n=250, seed=123)
    pd.testing.assert_frame_equal(simulate(cfg), simulate(cfg))
    assert not simulate(replace(cfg, seed=124)).equals(simulate(cfg))


def test_basic_truth_and_training_split():
    cfg = SimulationConfig(n=400, d=1, q=1, theta=.20, seed=21)
    df = simulate(cfg)
    assert len(df) == 400
    assert (df["h_true"] > 0).all()
    assert (df["sigma_true"] > 0).all()
    assert (df["partisi"].iloc[:320] == "Training").all()
    assert (df["partisi"].iloc[320:] == "Testing").all()
    np.testing.assert_allclose(df["Y_t"] - df["Y_lag1"], df["W_t"], atol=1e-10)
    np.testing.assert_allclose(df["epsilon_true"] / df["sigma_true"], df["Z_t"], atol=1e-11)
    np.testing.assert_allclose(df["mean_true"] + df["epsilon_true"], df["W_t"], atol=1e-11)


def test_garch_recursion_internal():
    eps = np.array([0., 2., -1., 3.])
    h = conditional_variance(eps, omega=.1, alpha=.2, beta=.7, h0=1.0)
    assert h[0] == 1.0
    assert h[1] == pytest.approx(.8)
    assert h[2] == pytest.approx(.1+.2*4+.7*.8)
    assert h[3] == pytest.approx(.1+.2*1+.7*h[2])


def test_homoskedastic_variance_exact():
    cfg = config_from_scenario("Homoskedastisitas", n=500, variance_target=2.5)
    df = simulate(cfg)
    np.testing.assert_allclose(df["h_true"], 2.5, atol=1e-13)


def test_structural_break_truth_starts_in_testing():
    cfg = config_from_scenario("Break struktural varians", n=400, training_fraction=.80)
    df = simulate(cfg)
    assert df["regime"].iloc[:340].sum() == 0
    assert (df["regime"].iloc[340:] == 1).all()
    assert df["omega_t"].iloc[340] == pytest.approx(4*df["omega_t"].iloc[339])


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        validate_config(SimulationConfig(alpha=.4,beta=.7))
    with pytest.raises(ValueError):
        validate_config(SimulationConfig(innovation="Student-t",df=1.5))


def test_student_t_variance_and_dgp():
    cfg = SimulationConfig(n=4000, innovation="Student-t", df=6, seed=14)
    df=simulate(cfg)
    assert abs(df["Z_t"].var()-1) < .20
    assert df["Z_t"].kurt() > 0


def test_no_future_information_in_one_step_forecast():
    cfg=SimulationConfig(n=300, seed=43)
    df=simulate(cfg)
    train=df["W_t"].to_numpy()[:240]
    fit=fit_arima_garch(train, q=0)
    assert fit["success"]
    all_w=df["W_t"].to_numpy()
    a=predict_fixed_parameters(all_w,fit,"GARCH",float(np.var(train)))
    changed=all_w.copy()
    changed[250:]=999.0
    b=predict_fixed_parameters(changed,fit,"GARCH",float(np.var(train)))
    # predictions at 250 depend on data through 249 only (therefore identical)
    np.testing.assert_allclose(a.iloc[:251][["pred_mean_w", "pred_var"]],
                               b.iloc[:251][["pred_mean_w", "pred_var"]],equal_nan=True)


def test_fit_one_dataset_and_mc_replay():
    cfg=SimulationConfig(n=250, q=0, seed=55)
    df=simulate(cfg)
    r=evaluate(df,cfg)
    assert r["baseline"]["success"]
    assert r["garch"]["success"]
    assert np.isfinite(r["garch_metric"]["qlike"])
    row=one_replication(cfg,"Persistensi tinggi",replication=1)
    assert "garch_success" in row
    raw,summary=run_monte_carlo(["Persistensi tinggi"],[250],2,cfg)
    raw2,summary2=run_monte_carlo(["Persistensi tinggi"],[250],2,cfg)
    pd.testing.assert_frame_equal(raw,raw2)
    pd.testing.assert_frame_equal(summary,summary2)
    assert summary["Jumlah replikasi"].iloc[0]==2


def test_arima_ma_and_student_estimator():
    cfg=SimulationConfig(n=350,q=1,theta=.25,innovation="Student-t",df=6,alpha=.1,beta=.75,seed=10)
    df=simulate(cfg)
    r=evaluate(df,cfg,estimated_distribution="Student-t")
    assert r["baseline"]["success"]
    assert r["garch"]["success"]
    assert r["garch"]["nu"]>2
    assert np.isfinite(r["garch_metric"]["coverage95"])
