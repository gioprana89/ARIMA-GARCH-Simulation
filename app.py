"""ARIMA–GARCH Simulation & Analysis Lab — Streamlit.

Run:
    pip install -r requirements.txt
    streamlit run app.py
All datasets are generated with deterministic Monte Carlo pseudo-random seeds.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tsa.stattools import acf, pacf

from simulation_engine import (
    SCENARIOS, SimulationConfig, config_as_dict, config_from_scenario,
    evaluate, predict_fixed_parameters, run_monte_carlo,
    simulate, summarize_monte_carlo, test_stationarity, validate_config,
)

st.set_page_config(page_title="ARIMA–GARCH Simulation Lab", page_icon="📈", layout="wide", initial_sidebar_state="expanded")

COLORS = {"blue": "#2067A8", "navy": "#143A60", "gold": "#E3A529", "muted": "#687B8E", "pale": "#F2F7FC"}
st.markdown("""
<style>
  .stApp {background: #fcfdfd; color: #203448;}
  [data-testid="stSidebar"] {background: #f5f9fd; border-right: 1px solid #e1eaf3;}
  h1,h2,h3 {color:#143A60; letter-spacing:-.02em;}
  [data-testid="stMetric"] {border:1px solid #e2ecf5;border-radius:14px;padding:13px 17px;background:white;}
  .lab-header {padding:22px 26px 19px;background:linear-gradient(110deg,#eff6fc 0%,#fff 68%);
               border:1px solid #e2ebf3;border-left:5px solid #E3A529;border-radius:16px;margin-bottom:18px;}
  .lab-kicker {font-size:12px;font-weight:800;letter-spacing:.1em;color:#BE8520;}
  .lab-header h1 {font-size:34px;margin:7px 0 5px;}
  .lab-sub {font-size:15px;color:#536b81;margin:0;}
  .tip {padding:13px 16px;background:#f5f9fd;border-left:3px solid #2067a8;border-radius:8px;color:#36516c;}
  div[data-testid="stTabs"] button {font-size:13px;}
</style>
""", unsafe_allow_html=True)


def fmt(x, digits=4):
    if x is None or (isinstance(x, (float, np.floating)) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{digits}f}"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_xlsx_bytes(sheets: dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)
            ws = writer.sheets[name[:31]]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions
            from openpyxl.styles import Font, PatternFill, Alignment
            for cell in ws[1]:
                cell.fill = PatternFill("solid", fgColor="143A60")
                cell.font = Font(color="FFFFFF", bold=True)
                cell.alignment = Alignment(wrap_text=True)
            for col in ws.columns:
                letter = col[0].column_letter
                widest = min(30, max(10, max(len(str(c.value or "")) for c in list(col)[:500]) + 2))
                ws.column_dimensions[letter].width = widest
    return buf.getvalue()


def metadata_table(cfg: SimulationConfig) -> pd.DataFrame:
    return pd.DataFrame([{"Parameter": k, "Nilai": v} for k, v in config_as_dict(cfg).items()])


def time_series_plot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t"], y=df["Y_t"], name="Y_t", mode="lines", line={"color": COLORS["blue"], "width": 1.8}))
    split_index = int((df["partisi"] == "Training").sum())
    if split_index < len(df):
        fig.add_vline(x=float(df["t"].iloc[split_index]), line_dash="dash", line_color=COLORS["gold"],
                      annotation_text="Awal testing", annotation_position="top left")
    if df["regime"].sum():
        mark = int(df.loc[df["regime"] == 1, "t"].iloc[0])
        fig.add_vline(x=mark, line_dash="dot", line_color="#C84A55", annotation_text="Break varians")
    fig.update_layout(height=350, margin=dict(l=12,r=10,t=20,b=10), xaxis_title="Waktu t", yaxis_title="Y_t", hovermode="x unified")
    return fig


def volatility_plot(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["t"], y=df["sigma_true"], line={"color": COLORS["gold"], "width": 1.5}, name="σ_t sebenarnya"))
    fig.update_layout(height=325, margin=dict(l=12,r=10,t=20,b=10), xaxis_title="Waktu t", yaxis_title="σ_t", hovermode="x unified")
    return fig


def correlogram_plot(values: np.ndarray, kind: str = "ACF") -> go.Figure:
    nlags = min(30, max(2, len(values)//4))
    corrs = acf(values, nlags=nlags, fft=True) if kind == "ACF" else pacf(values, nlags=nlags, method="ywm")
    fig = go.Figure(go.Bar(x=list(range(len(corrs))), y=corrs, marker_color=COLORS["blue"], name=kind))
    bound = 1.96 / np.sqrt(len(values))
    fig.add_hline(y=bound, line_dash="dash", line_color=COLORS["gold"])
    fig.add_hline(y=-bound, line_dash="dash", line_color=COLORS["gold"])
    fig.update_layout(height=280, margin=dict(l=10,r=5,t=10,b=5), xaxis_title="Lag", yaxis_title=kind)
    return fig


def forecast_frame(df: pd.DataFrame, result: dict) -> pd.DataFrame:
    out = df[["t", "partisi", "Y_t", "W_t", "h_true"]].copy()
    for key, prefix in [("baseline", "arima"), ("garch", "garch")]:
        pred = result[key+"_pred"]
        if pred is not None:
            out[prefix+"_mean_w_pred"] = pred["pred_mean_w"].to_numpy()
            out[prefix+"_h_pred"] = pred["pred_var"].to_numpy()
            if st.session_state.cfg.d == 1:
                # For t>=1: forecast of Y_t = Y_(t-1) + forecast of W_t.
                out[prefix+"_Y_pred"] = df["Y_t"].shift(1).to_numpy() + pred["pred_mean_w"].to_numpy()
            else:
                out[prefix+"_Y_pred"] = pred["pred_mean_w"].to_numpy()
    return out


def download_pair(df: pd.DataFrame, prefix: str, keyprefix: str, meta: SimulationConfig | None = None):
    a, b = st.columns(2)
    with a:
        st.download_button("⬇️ Unduh CSV", data=to_csv_bytes(df), file_name=prefix+".csv", mime="text/csv", key=keyprefix+"csv", use_container_width=True)
    with b:
        sheets = {"Data": df}
        if meta:
            sheets["Metadata"] = metadata_table(meta)
        st.download_button("⬇️ Unduh Excel", data=to_xlsx_bytes(sheets), file_name=prefix+".xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=keyprefix+"xlsx", use_container_width=True)


def score_table(result: dict) -> pd.DataFrame:
    fields = [("mse_var", "MSE varians ↓"), ("rmse_var", "RMSE varians ↓"), ("qlike", "QLIKE ↓"),
              ("rmse_mean", "RMSE prediksi mean ↓"), ("mae_mean", "MAE prediksi mean ↓"),
              ("coverage95", "Cakupan interval 95%"), ("interval_width95", "Lebar rata-rata interval 95%")]
    return pd.DataFrame([{"Metrik": display, "ARIMA": result["baseline_metric"].get(key, np.nan),
                          "ARIMA-GARCH": result["garch_metric"].get(key, np.nan)} for key, display in fields])


def report_png(df: pd.DataFrame, result: dict | None = None) -> bytes:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=140, constrained_layout=True)
    axes[0].plot(df["t"], df["Y_t"], color=COLORS["blue"], lw=1.2)
    axes[0].set(title="Dataset simulasi Y(t)", xlabel="Waktu t", ylabel="Y(t)")
    axes[1].plot(df["t"], df["h_true"], color=COLORS["gold"], lw=1.4, label="h(t) benar")
    if result and result.get("garch_pred") is not None:
        split = result["split"]
        axes[1].plot(df["t"].iloc[split:], result["garch_pred"]["pred_var"].iloc[split:],
                     color=COLORS["blue"], lw=1.2, label="Prediksi h(t) ARIMA-GARCH")
        axes[1].legend()
    axes[1].set(title="Varians bersyarat sebenarnya dan prediksi", xlabel="Waktu t", ylabel="h(t)")
    fig.suptitle("ARIMA-GARCH SIMULATION & ANALYSIS LAB", fontweight="bold", fontsize=14)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def build_archive(cfg, df, result, raw, summary) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("01_dataset_simulasi.csv", to_csv_bytes(df))
        z.writestr("02_parameter_simulasi.json", json.dumps(config_as_dict(cfg), indent=2, ensure_ascii=False))
        z.writestr("03_grafik_simulasi.png", report_png(df, result))
        if result:
            scores = score_table(result)
            z.writestr("04_evaluasi_peramalan.csv", to_csv_bytes(scores))
            z.writestr("05_prediksi_testing.csv", to_csv_bytes(forecast_frame(df, result).query("partisi == 'Testing'")))
        if raw is not None:
            z.writestr("06_monte_carlo_per_replikasi.csv", to_csv_bytes(raw))
            z.writestr("07_monte_carlo_ringkasan.csv", to_csv_bytes(summary))
        z.writestr("PANDUAN.txt", "Dataset dibuat secara simulasi, tidak berasal dari data riil.\n"
                   "Semua seed dan parameter tercatat.\n"
                   "QLIKE: rata-rata [log(h_pred)+h_true/h_pred] (semakin kecil semakin baik).\n"
                   "Evaluasi one-step-ahead menggunakan parameter yang ditaksir sekali pada training.\n"
                   "Tidak ada data testing yang digunakan untuk mengestimasi parameter.\n"
                   "Kegagalan konvergensi tetap dicatat dalam Monte Carlo.\n")
    return buf.getvalue()


# Default dataset is immediately visible. Fitting is fast and performed once per new dataset.
if "cfg" not in st.session_state:
    st.session_state.cfg = SimulationConfig()
    st.session_state.dataset = simulate(st.session_state.cfg)
    st.session_state.single_result = None
    st.session_state.mc_raw = None
    st.session_state.mc_summary = None
    st.session_state.mc_cfg = None

cfg: SimulationConfig = st.session_state.cfg
df: pd.DataFrame = st.session_state.dataset
result = st.session_state.single_result

st.markdown("""
<div class="lab-header">
  <div class="lab-kicker">STATISTIKA KOMPUTASI • SKRIPSI MATEMATIKA</div>
  <h1>ARIMA–GARCH Simulation &amp; Analysis Lab</h1>
  <p class="lab-sub">Simulasi Monte Carlo · Estimasi Model · Analisis Volatilitas · Evaluasi Out-of-Sample</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### 🔬 Laboratorium ARIMA–GARCH")
    st.caption("Seluruh data dibangkitkan secara simulasi. Tidak memerlukan unggah dataset.")
    st.markdown("**Dataset aktif**")
    st.write(f"Observasi: **{cfg.n:,}**")
    st.write(f"Model: **ARIMA(1,{cfg.d},{cfg.q})–GARCH(1,1)**")
    st.write(f"Persistensi: **{cfg.alpha+cfg.beta:.3f}**")
    st.write(f"Seed: **{cfg.seed}**")
    st.divider()
    st.info("**Target kajian:** karakteristik data → hasil estimasi → kinerja peramalan volatilitas.")
    st.caption("Model estimasi menggunakan likelihood kondisional. ARIMA(1,d,0/1) adalah bagian dari keluarga ARIMA.")

TABS = ["🏠 Panduan", "⚙️ Pengaturan", "📋 Dataset", "📊 Eksplorasi", "🧪 Uji Statistik",
        "📐 Estimasi", "🔍 Diagnostik", "📈 Peramalan", "🔁 Monte Carlo", "📦 Ekspor"]
tabs = st.tabs(TABS)

with tabs[0]:
    st.subheader("Tujuan penelitian dan alur eksperimen")
    st.markdown("Aplikasi ini menguji **kapan ARIMA–GARCH unggul dan kapan ia mengalami keterbatasan**, dengan menggunakan data yang karakteristik serta parameter pembangkitnya diketahui.")
    a,b,c = st.columns(3)
    a.metric("Data saat ini", f"{cfg.n:,} observasi")
    b.metric("Persistensi α + β", fmt(cfg.alpha+cfg.beta, 3))
    c.metric("Training / Testing", f"{cfg.training_fraction:.0%} / {1-cfg.training_fraction:.0%}")
    st.plotly_chart(time_series_plot(df), use_container_width=True)
    st.markdown("""<div class="tip"><b>Catatan:</b> ARIMA memodelkan komponen rata-rata, sedangkan GARCH memodelkan varians bersyarat. Evaluasi utama penelitian ini adalah kualitas peramalan varians, bukan hanya kesalahan ramalan titik.</div>""", unsafe_allow_html=True)
    st.markdown("#### Cara menggunakan aplikasi")
    st.markdown("1. Atur parameter pada **Pengaturan**, lalu bangkitkan dataset.\n2. Periksa semua observasi pada **Dataset**, grafik pada **Eksplorasi**, dan uji awal pada **Uji Statistik**.\n3. Jalankan estimasi ARIMA dan ARIMA–GARCH di **Estimasi**.\n4. Periksa **Diagnostik** dan **Peramalan** pada data testing.\n5. Jalankan banyak replikasi di **Monte Carlo** dan lihat pola kinerja.\n6. Simpan dataset, hasil, dan grafik di **Ekspor**.")
    with st.expander("Persamaan model yang digunakan", expanded=False):
        st.latex(r"W_t = c + \phi W_{t-1} + \theta\varepsilon_{t-1}+\varepsilon_t,\quad W_t=\Delta^d Y_t")
        st.latex(r"\varepsilon_t = \sqrt{h_t}z_t,\quad h_t=\omega+\alpha\varepsilon_{t-1}^2+\beta h_{t-1}")
        st.latex(r"\omega=V(1-\alpha-\beta),\quad \alpha+\beta<1")
        st.caption("Jika q=0 maka θ=0. Untuk Student-t, inovasi distandardisasi agar memiliki varians 1. Untuk perubahan struktural, ω berubah pada titik break.")

with tabs[1]:
    st.subheader("Pengaturan pembangkit data simulasi")
    st.write("Pilih preset atau atur parameter sendiri. Perubahan baru diterapkan setelah tombol **Bangkitkan Dataset** ditekan.")
    with st.form("form_simulasi"):
        c1, c2 = st.columns(2)
        with c1:
            mode = st.radio("Sumber parameter GARCH", ["Preset skenario", "Kustom"], horizontal=True)
            preset = st.selectbox("Pilih skenario", list(SCENARIOS), index=3, disabled=mode=="Kustom")
            alpha_input = st.number_input("α (ARCH)", min_value=0.0, max_value=0.99, value=float(cfg.alpha), step=0.01, disabled=mode!="Kustom", format="%.3f")
            beta_input = st.number_input("β (GARCH)", min_value=0.0, max_value=0.99, value=float(cfg.beta), step=0.01, disabled=mode!="Kustom", format="%.3f")
            innovation = st.selectbox("Distribusi inovasi", ["Normal", "Student-t"], index=0 if cfg.innovation=="Normal" else 1, disabled=mode!="Kustom")
            break_factor = st.number_input("Pengali omega setelah break (1 = tanpa break)", 1.0, 15.0, float(cfg.break_factor), 0.5, disabled=mode!="Kustom")
            student_df = st.number_input("Derajat bebas Student-t", 2.1, 50.0, float(cfg.df), 0.5)
            break_pct = st.slider("Titik perubahan struktural (% posisi sampel)", 55, 90, int(cfg.break_at*100), 5)
        with c2:
            n = st.number_input("Jumlah observasi (n)", 100, 5000, int(cfg.n), 50)
            d = st.selectbox("Ordo differencing (d)", [0,1], index=int(cfg.d))
            q = st.selectbox("Ordo moving average (q)", [0,1], index=int(cfg.q))
            phi = st.number_input("Koefisien AR (φ)", -0.95, 0.95, float(cfg.phi), 0.05, format="%.3f")
            theta = st.number_input("Koefisien MA (θ), jika q=1", -0.95, 0.95, float(cfg.theta), 0.05, disabled=q==0, format="%.3f")
            mu = st.number_input("Rata-rata jangka panjang W (μ)", value=float(cfg.mu), step=0.1)
            target_var = st.number_input("Varians tak bersyarat target (V)", 0.01, 1000.0, float(cfg.variance_target), 0.1)
            train_pct = st.slider("Proporsi training (%)", 50, 90, int(100*cfg.training_fraction), 5)
            seed = st.number_input("Random seed", 0, 2_000_000_000, int(cfg.seed), 1)
        submitted = st.form_submit_button("🎲 Bangkitkan Dataset Simulasi", use_container_width=True, type="primary")
    if submitted:
        try:
            params = SCENARIOS[preset] if mode == "Preset skenario" else {
                "alpha": alpha_input, "beta": beta_input, "innovation": innovation,
                "break_factor": break_factor,
            }
            new_cfg = SimulationConfig(n=n,d=d,q=q,phi=phi,theta=theta if q else 0.0,mu=mu,
                                      variance_target=target_var,training_fraction=train_pct/100,
                                      df=student_df,seed=seed,break_at=break_pct/100,**params)
            validate_config(new_cfg)
            st.session_state.cfg = new_cfg
            st.session_state.dataset = simulate(new_cfg)
            st.session_state.single_result = None
            st.session_state.mc_raw = None
            st.session_state.mc_summary = None
            st.session_state.mc_cfg = None
            st.success(f"Dataset simulasi berhasil dibangkitkan: {n:,} observasi, seed {seed}. Silakan buka tab Dataset.")
            st.rerun()
        except Exception as e:
            st.error(f"Parameter tidak valid: {e}")
    st.info("**Pengendalian eksperimen:** omega otomatis dihitung sebagai V × (1 − α − β). Hal ini menjaga varians tak bersyarat target tetap sama pada skenario tanpa break.")
    st.caption("Skenario break mengubah omega pada 85% panjang data (default; dalam periode testing apabila training 80%). Pengujian d=1 dilakukan pada Wₜ=ΔYₜ, sedangkan grafik deret waktu utama menampilkan Yₜ.")

with tabs[2]:
    st.subheader("Dataset hasil simulasi — seluruh observasi")
    dset = df
    config_dset = cfg
    if st.session_state.mc_raw is not None and not st.session_state.mc_raw.empty:
        source = st.radio("Sumber dataset", ["Dataset utama", "Replikasi Monte Carlo"], horizontal=True)
        if source == "Replikasi Monte Carlo":
            raw = st.session_state.mc_raw
            ix = st.selectbox("Pilih baris replikasi", options=list(range(len(raw))),
                              format_func=lambda i: f"{raw.iloc[i]['scenario']} | n={raw.iloc[i]['n']} | replikasi {raw.iloc[i]['replication']} | seed {raw.iloc[i]['seed']}")
            row = raw.iloc[ix]
            config_dset = replace(st.session_state.mc_cfg, n=int(row["n"]), seed=int(row["seed"]), **SCENARIOS[row["scenario"]])
            dset = simulate(config_dset)
            st.caption("Dataset ini diregenerasi dari seed yang disimpan. Tidak memerlukan menyimpan semua replikasi di RAM.")
    aa, bb, cc = st.columns(3)
    aa.metric("Observasi", f"{len(dset):,}")
    bb.metric("Training", f"{(dset['partisi']=='Training').sum():,}")
    cc.metric("Testing", f"{(dset['partisi']=='Testing').sum():,}")
    which = st.selectbox("Tampilkan observasi", ["Semua", "Training", "Testing"])
    show_df = dset if which=="Semua" else dset.loc[dset["partisi"]==which]
    st.dataframe(show_df, use_container_width=True, height=440, hide_index=True)
    st.markdown("**Kamus variabel**")
    st.caption("Y_t = deret waktu; Y_lag1 = level sebelumnya (untuk d=1); W_t = ΔᵈY_t; mean_true = rata-rata bersyarat sebenarnya; omega_t = konstanta varians pada periode t; Z_t = inovasi standar; epsilon_true = inovasi sebenarnya; h_true = varians bersyarat sebenarnya; sigma_true = akar varians; regime = 0 sebelum / 1 setelah break; partisi = training/testing.")
    download_pair(dset, "dataset_simulasi_arima_garch", "download_dataset_tab", config_dset)

with tabs[3]:
    st.subheader("Eksplorasi dan visualisasi data")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Rerata Wₜ", fmt(df["W_t"].mean()))
    c2.metric("SD Wₜ", fmt(df["W_t"].std()))
    c3.metric("Skewness Wₜ", fmt(df["W_t"].skew()))
    c4.metric("Kurtosis excess Wₜ", fmt(df["W_t"].kurt()))
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("**Deret waktu Yₜ**")
        st.plotly_chart(time_series_plot(df), use_container_width=True)
    with col2:
        st.markdown("**Volatilitas bersyarat sebenarnya σₜ**")
        st.plotly_chart(volatility_plot(df), use_container_width=True)
    c1,c2=st.columns(2)
    with c1:
        fig = px.line(df,x="t",y="W_t",title="Komponen stasioner Wₜ", color_discrete_sequence=[COLORS["blue"]])
        fig.update_layout(height=300,margin=dict(t=40,b=15))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df,x="W_t",nbins=45,title="Distribusi komponen stasioner Wₜ",color_discrete_sequence=[COLORS["gold"]])
        fig.update_layout(height=300,margin=dict(t=40,b=15))
        st.plotly_chart(fig,use_container_width=True)
    with st.expander("Statistik deskriptif lengkap", expanded=False):
        st.dataframe(df[["Y_t","W_t","epsilon_true","h_true","sigma_true"]].describe().T,use_container_width=True)

with tabs[4]:
    st.subheader("Uji awal dan identifikasi karakteristik data")
    st.caption("Uji ADF dilakukan pada Wₜ (setelah differencing jika d=1). Efek ARCH diuji pada residual model mean ARIMA hasil estimasi training, bukan langsung pada kurs mentah.")
    try:
        adf_w = test_stationarity(df["W_t"].to_numpy())
        a,b,c=st.columns(3)
        a.metric("ADF statistic Wₜ",fmt(adf_w["ADF statistic"]))
        b.metric("ADF p-value",fmt(adf_w["p-value"],6))
        c.metric("Keputusan (5%)","Stasioner" if adf_w["p-value"] < .05 else "Gagal menolak unit root")
        with st.expander("Uji ADF pada level Yₜ"):
            st.write(test_stationarity(df["Y_t"].to_numpy()))
    except Exception as e:
        st.warning(f"Uji ADF tidak dapat dihitung: {e}")
    v1,v2 = st.columns(2)
    with v1:
        st.markdown("**ACF Wₜ**")
        st.plotly_chart(correlogram_plot(df["W_t"].to_numpy(),"ACF"),use_container_width=True)
    with v2:
        st.markdown("**PACF Wₜ**")
        st.plotly_chart(correlogram_plot(df["W_t"].to_numpy(),"PACF"),use_container_width=True)
    v1,v2 = st.columns(2)
    with v1:
        st.markdown("**ACF inovasi sebenarnya kuadrat**")
        st.plotly_chart(correlogram_plot(df["epsilon_true"].to_numpy()**2,"ACF"),use_container_width=True)
    with v2:
        st.markdown("**Efek ARCH pada inovasi sebenarnya**")
        try:
            p_arch = float(het_arch(df["epsilon_true"].to_numpy(),nlags=5)[1])
            st.metric("ARCH-LM p-value",fmt(p_arch,6))
            st.write("Indikasi efek ARCH pada taraf 5%." if p_arch<.05 else "Tidak ditemukan bukti efek ARCH pada taraf 5%.")
        except Exception as e:
            st.warning(f"ARCH-LM gagal: {e}")
        st.caption("Ini merupakan pemeriksaan DGP (inovasi sebenarnya tersedia karena simulasi). Untuk analisis seperti data nyata, gunakan residual model mean pada tab Estimasi.")

with tabs[5]:
    st.subheader("Estimasi ARIMA dan ARIMA–GARCH")
    a,b=st.columns([2,1])
    with a:
        estimated_distribution = st.selectbox("Distribusi yang DIASUMSIKAN pada model ARIMA–GARCH", ["Normal","Student-t"],
                                              index=0, help="Bisa dibuat berbeda dari distribusi pembangkit untuk uji ketahanan model.")
    with b:
        maxiter_single = st.selectbox("Maksimum iterasi optimizer", [120,180,300],index=1)
    if st.button("▶️ Estimasi dua model (training saja)",type="primary",use_container_width=True):
        with st.spinner("Mengestimasi likelihood kondisional pada data training..."):
            try:
                st.session_state.single_result = evaluate(df,cfg,estimated_distribution,maxiter_single)
                st.session_state.fit_distribution = estimated_distribution
                st.success("Estimasi selesai. Hasil dapat dilihat di bawah dan tab Diagnostik/Peramalan.")
            except Exception as e:
                st.error(f"Estimasi tidak dapat diselesaikan: {type(e).__name__}: {e}")
    result = st.session_state.single_result
    if result is None:
        st.info("Tekan tombol estimasi untuk menghasilkan parameter model, AIC/BIC, dan diagnostik residual.")
    else:
        g = result["garch"]
        base = result["baseline"]
        c1,c2,c3=st.columns(3)
        c1.metric("ARIMA baseline", "Konvergen" if base["success"] else "Gagal")
        c2.metric("ARIMA–GARCH", "Konvergen" if g["success"] else "Gagal")
        c3.metric("Distribusi estimasi",g["distribution"])
        par_true={"c":(1-cfg.phi)*cfg.mu,"phi":cfg.phi,"theta":cfg.theta if cfg.q else 0.,
                  "omega":cfg.variance_target*(1-cfg.alpha-cfg.beta),"alpha":cfg.alpha,"beta":cfg.beta,
                  "persistence":cfg.alpha+cfg.beta}
        pars = pd.DataFrame([{"Parameter":key,"Parameter sebenarnya":par_true[key],
                               "Estimasi ARIMA–GARCH":g.get(key,np.nan) if g["success"] else np.nan,
                               "Selisih":g.get(key,np.nan)-par_true[key] if g["success"] else np.nan}
                             for key in par_true if key!="theta" or cfg.q==1])
        st.markdown("**Parameter pembangkit versus parameter hasil estimasi**")
        st.dataframe(pars,use_container_width=True,hide_index=True)
        st.caption("Pada skenario break struktural, nilai ω 'sebenarnya' yang ditampilkan adalah nilai sebelum break; setelah break nilainya berubah. Model estimasi tetap mengasumsikan parameter konstan.")
        fit_comp=pd.DataFrame([{"Model":"ARIMA","AIC":base["aic"],"BIC":base["bic"],"Status":base["message"]},
                               {"Model":"ARIMA–GARCH","AIC":g["aic"],"BIC":g["bic"],"Status":g["message"]}])
        st.markdown("**Kriteria informasi dan status optimisasi**")
        st.dataframe(fit_comp,use_container_width=True,hide_index=True)
        st.caption("Likelihood keduanya menggunakan pengondisian dan sampel training yang sama. AIC/BIC tidak menggantikan validasi out-of-sample. Status gagal konvergensi harus dilaporkan.")

with tabs[6]:
    st.subheader("Diagnostik residual pada data training")
    result = st.session_state.single_result
    if result is None:
        st.info("Jalankan estimasi pada tab Estimasi terlebih dahulu.")
    else:
        diagnostic=[]
        for key,label in [("baseline","ARIMA"),("garch","ARIMA–GARCH")]:
            dgn=result[key+"_diagnostics"]
            diagnostic.append({"Model":label,"Ljung-Box resid (p)":dgn["ljungbox_p"],
                               "Ljung-Box resid² (p)":dgn["ljungbox_sq_p"],"ARCH-LM (p)":dgn["arch_lm_p"],
                               "Jarque-Bera (p)":dgn["jarque_bera_p"]})
        st.dataframe(pd.DataFrame(diagnostic),use_container_width=True,hide_index=True)
        st.caption("Untuk GARCH digunakan residual terstandarisasi. Untuk model varians konstan residual dibagi simpangan baku konstan. Nilai-p > 0,05 menunjukkan tidak cukup bukti menolak H₀ pada uji terkait, bukan bukti H₀ pasti benar.")
        pred=result["garch_pred"]
        if pred is not None:
            split=result["split"]
            left,right=st.columns(2)
            with left:
                st.markdown("**Residual terstandarisasi (training)**")
                z=pred["z_standard"].iloc[1:split]
                f=go.Figure(go.Scatter(x=df["t"].iloc[1:split],y=z,mode="lines",line_color=COLORS["blue"]))
                f.add_hline(y=0,line_color="#9AA8B3")
                f.update_layout(height=300,margin=dict(t=20,b=10),xaxis_title="t",yaxis_title="z estimasi")
                st.plotly_chart(f,use_container_width=True)
            with right:
                st.markdown("**ACF kuadrat residual terstandarisasi**")
                st.plotly_chart(correlogram_plot(z.to_numpy()**2,"ACF"),use_container_width=True)
        st.warning("Ketidaknormalan residual bukan otomatis berarti ARIMA–GARCH gagal. Untuk inovasi berekor tebal, bandingkan asumsi Normal dan Student-t, lalu evaluasi diagnostik dan interval prediksinya.")

with tabs[7]:
    st.subheader("Evaluasi peramalan out-of-sample")
    result=st.session_state.single_result
    if result is None:
        st.info("Estimasi model terlebih dahulu di tab Estimasi.")
    else:
        scores=score_table(result)
        st.markdown("**Metrik prediksi pada data testing saja**")
        st.dataframe(scores.style.format({"ARIMA":"{:.5f}","ARIMA-GARCH":"{:.5f}"}),use_container_width=True,hide_index=True)
        st.caption("Untuk varians, QLIKE = mean[log(h_pred) + h_true/h_pred]. QLIKE lebih kecil lebih baik; angka negatif boleh terjadi. Karena h_true diketahui dari simulasi, tidak diperlukan proksi volatilitas.")
        pred_base=result["baseline_pred"]
        pred_g=result["garch_pred"]
        split=result["split"]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=df["t"].iloc[split:],y=df["h_true"].iloc[split:],name="h benar",line={"color":COLORS["navy"],"width":2}))
        if pred_base is not None:
            fig.add_trace(go.Scatter(x=df["t"].iloc[split:],y=pred_base["pred_var"].iloc[split:],name="ARIMA varians tetap",line={"color":COLORS["gold"],"dash":"dash"}))
        if pred_g is not None:
            fig.add_trace(go.Scatter(x=df["t"].iloc[split:],y=pred_g["pred_var"].iloc[split:],name="ARIMA–GARCH prediksi h",line={"color":COLORS["blue"]}))
        fig.update_layout(title="Varians sebenarnya vs one-step-ahead forecast (testing)",height=410,xaxis_title="t",yaxis_title="h",hovermode="x unified")
        st.plotly_chart(fig,use_container_width=True)
        if pred_g is not None:
            f=go.Figure()
            f.add_trace(go.Scatter(x=df["t"].iloc[split:],y=df["W_t"].iloc[split:],name="W aktual",line_color=COLORS["navy"]))
            f.add_trace(go.Scatter(x=df["t"].iloc[split:],y=pred_g["pred_mean_w"].iloc[split:],name="ARIMA–GARCH mean forecast",line_color=COLORS["gold"]))
            f.update_layout(title="Ramalan satu langkah untuk Wₜ (testing)",height=320,xaxis_title="t",yaxis_title="W")
            st.plotly_chart(f,use_container_width=True)
        st.markdown("**Tabel hasil prediksi lengkap**")
        forecasts=forecast_frame(df,result)
        st.dataframe(forecasts.query("partisi == 'Testing'"),use_container_width=True,height=310,hide_index=True)
        st.download_button("⬇️ Unduh hasil peramalan CSV",to_csv_bytes(forecasts),"hasil_peramalan.csv","text/csv",key="download_forecast_here")
        st.info("**Metode evaluasi:** parameter diestimasi satu kali pada training. Selama testing, prediksi satu langkah diperbarui memakai observasi masa lalu, tanpa mengestimasi ulang parameter. Ini adalah recursive/rolling one-step forecasting dengan parameter tetap, bukan rolling re-estimation.")

with tabs[8]:
    st.subheader("Eksperimen Monte Carlo dan pencarian pola")
    st.write("Jalankan banyak replikasi dengan skenario terkontrol. Setiap percobaan menyimpan seed, hasil estimasi, metrik akurasi, dan status konvergensi.")
    a,b,c = st.columns([2,1,1])
    with a:
        chosen_scenarios=st.multiselect("Skenario",list(SCENARIOS),
                 default=["Homoskedastisitas","Persistensi sedang","Persistensi tinggi","Break struktural varians"])
    with b:
        chosen_sizes=st.multiselect("Ukuran sampel",[150,250,500,1000,2000],default=[250,500])
    with c:
        reps=st.number_input("Replikasi per sel",min_value=2,max_value=500,value=10,step=5)
        mc_dist=st.selectbox("Asumsi distribusi estimasi",["Normal","Student-t"],index=0)
    runs=len(chosen_scenarios)*len(chosen_sizes)*reps
    st.caption(f"Total estimasi per model: {runs:,} fit (skenario × ukuran × replikasi). Model ARIMA dan ARIMA–GARCH diestimasi untuk setiap replikasi.")
    if runs > 1000:
        st.warning("Eksperimen besar dapat memerlukan waktu dan memori cukup banyak; jalankan bertahap atau kurangi jumlah sel/replikasi.")
    if st.button("🔁 Jalankan Monte Carlo",type="primary",disabled=not chosen_scenarios or not chosen_sizes, use_container_width=True):
        progress=st.progress(0,text="Memulai eksperimen ...")
        current=st.empty()
        def update(k,total,name,n_current):
            progress.progress(k/total,text=f"Simulasi {k:,} dari {total:,}")
            if k==1 or k%5==0 or k==total:
                current.caption(f"Skenario: {name} · n={n_current} · {k}/{total} selesai")
        try:
            mc_cfg=replace(cfg)
            raw,summary=run_monte_carlo(chosen_scenarios,chosen_sizes,int(reps),mc_cfg,
                                        estimated_distribution=mc_dist,progress_callback=update,maxiter=120)
            st.session_state.mc_raw=raw
            st.session_state.mc_summary=summary
            st.session_state.mc_cfg=mc_cfg
            progress.empty()
            current.empty()
            st.success(f"Monte Carlo selesai: {len(raw):,} replikasi. Hasil tetap tersedia selama sesi Streamlit aktif.")
        except Exception as e:
            st.error(f"Eksperimen gagal: {type(e).__name__}: {e}")
    raw=st.session_state.mc_raw
    summary=st.session_state.mc_summary
    if raw is not None and not raw.empty:
        st.divider()
        st.markdown("### Ringkasan pola hasil simulasi")
        st.dataframe(summary,use_container_width=True,hide_index=True)
        if len(summary):
            a,b=st.columns(2)
            with a:
                piv=summary.pivot(index="Skenario",columns="n",values="Keuntungan QLIKE rata-rata")
                heat=px.imshow(piv,aspect="auto",color_continuous_scale="RdBu",color_continuous_midpoint=0,
                        labels={"color":"ARIMA QLIKE − GARCH QLIKE", "x":"n", "y":"Skenario"},title="Keuntungan QLIKE (positif = GARCH unggul)")
                heat.update_layout(height=410,margin=dict(t=55,b=30))
                st.plotly_chart(heat,use_container_width=True)
            with b:
                bar=px.bar(summary,x="Skenario",y="GARCH unggul QLIKE (%)",color="n",barmode="group",
                           title="Persentase replikasi dengan GARCH lebih baik",text_auto=".1f")
                bar.update_layout(height=410,margin=dict(t=55,b=80),xaxis_tickangle=-30)
                st.plotly_chart(bar,use_container_width=True)
        st.markdown("### Perbedaan kinerja antar replikasi")
        clean=raw.loc[raw["garch_success"].fillna(False) & raw["arima_success"].fillna(False)].copy()
        if not clean.empty and "qlike_gain" in clean:
            box=px.box(clean,x="scenario",y="qlike_gain",color="n",points="outliers",
                       title="Sebaran keuntungan QLIKE (positif = ARIMA–GARCH lebih baik)")
            box.add_hline(y=0,line_dash="dash",line_color="#B17D20")
            box.update_layout(height=420,xaxis_tickangle=-20)
            st.plotly_chart(box,use_container_width=True)
        fail=int((~raw["garch_success"].fillna(False).astype(bool)).sum())
        st.caption(f"Jumlah gagal konvergensi / kegagalan proses ARIMA–GARCH: {fail:,} dari {len(raw):,} replikasi. Ringkasan rata-rata dihitung hanya untuk pasangan model yang valid; persentase konvergensi dilaporkan agar bias seleksi terlihat.")
        st.markdown("### Data setiap replikasi (tidak hanya ringkasan)")
        st.dataframe(raw,use_container_width=True,height=310,hide_index=True)
        st.info("Buka tab **Dataset** → **Replikasi Monte Carlo** untuk menampilkan dan mengunduh dataset dari salah satu replikasi. Dataset diregenerasi dengan seed yang tersimpan.")
        download_pair(summary,"ringkasan_monte_carlo","mc_summary_dl")
        download_pair(raw,"hasil_per_replikasi_monte_carlo","mc_raw_dl")
    else:
        st.info("Mulai dengan 2–4 skenario, dua ukuran sampel, dan 10 replikasi untuk pemeriksaan awal. Untuk pelaporan skripsi, tingkatkan jumlah replikasi dan periksa stabilitas hasil.")

with tabs[9]:
    st.subheader("Ekspor dataset dan keluaran penelitian")
    st.write("Seluruh data simulasi dapat disimpan agar pemodelan dan kesimpulan skripsi dapat direplikasi.")
    st.markdown("#### 1. Dataset aktif dan parameter pembangkit")
    download_pair(df,"dataset_simulasi_utama","exp_main",cfg)
    st.markdown("#### 2. Grafik PNG")
    result=st.session_state.single_result
    st.download_button("🖼️ Unduh grafik dataset & volatilitas (PNG)",data=report_png(df,result),
                       file_name="grafik_simulasi_arima_garch.png",mime="image/png",key="png_exp")
    if result:
        st.markdown("#### 3. Estimasi, evaluasi, dan prediksi")
        sheets={"Evaluasi":score_table(result),"Prediksi":forecast_frame(df,result),
                "Parameter":pd.DataFrame([{"Model":model,**result[key]} for model,key in [("ARIMA","baseline"),("ARIMA-GARCH","garch")]])}
        st.download_button("⬇️ Unduh hasil analisis (Excel)",to_xlsx_bytes(sheets),
                           "hasil_estimasi_dan_peramalan.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="exp_analysis")
    if st.session_state.mc_raw is not None:
        st.markdown("#### 4. Hasil Monte Carlo")
        st.download_button("⬇️ Unduh seluruh hasil Monte Carlo (Excel)",
                           to_xlsx_bytes({"Per replikasi":st.session_state.mc_raw,"Ringkasan":st.session_state.mc_summary}),
                           "hasil_monte_carlo.xlsx","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",key="exp_mc")
    st.markdown("#### 5. Paket arsip penelitian")
    archive=build_archive(cfg,df,result,st.session_state.mc_raw,st.session_state.mc_summary)
    st.download_button("📦 Unduh semua hasil sebagai ZIP",archive,"paket_hasil_arima_garch.zip","application/zip",key="exp_archive",type="primary")
    st.caption("Keterangan: file ZIP mencakup data simulasi CSV, konfigurasi JSON, grafik PNG, serta hasil prediksi/Monte Carlo jika telah dijalankan. Semua variabel asli dan parameter simulasi tetap disimpan.")

st.divider()
st.caption("ARIMA–GARCH Simulation & Analysis Lab · Penelitian simulasi Monte Carlo · Model: ARIMA(1,d,q)–GARCH(1,1), d,q∈{0,1} · Evaluasi satu langkah pada data uji.")
