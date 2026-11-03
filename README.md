# ARIMA–GARCH Simulation & Analysis Lab

Aplikasi Python Streamlit untuk penelitian skripsi Matematika: **mengidentifikasi karakteristik deret waktu yang memengaruhi kinerja peramalan ARIMA–GARCH melalui simulasi Monte Carlo.** Seluruh data dibangkitkan secara sintetis, tanpa dataset eksternal.

## Memulai

Gunakan Python 3.10 atau lebih baru. Buka terminal di dalam folder ini:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Browser biasanya otomatis membuka aplikasi di `http://localhost:8501`.

Untuk Streamlit Community Cloud, unggah `app.py`, `simulation_engine.py`, `requirements.txt`, dan `.streamlit/config.toml` ke repositori GitHub; pilih `app.py` sebagai entry point. **Tidak perlu dataset Excel** karena aplikasi membangkitkan data sendiri.

## Apa yang ada di aplikasi?

1. **Panduan**: persamaan, tujuan penelitian, alur eksperimen.
2. **Pengaturan**: pilih skenario, parameter pembangkit, ukuran sampel, seed, dan proporsi training.
3. **Dataset**: seluruh observasi simulasi; termasuk dataset dari replikasi Monte Carlo tertentu.
4. **Eksplorasi**: grafik nilai deret waktu, volatilitas, histogram, statistik deskriptif.
5. **Uji Statistik**: ADF, ACF, PACF, ARCH-LM.
6. **Estimasi**: parameter ARIMA dan ARIMA–GARCH, AIC/BIC, konvergensi.
7. **Diagnostik**: Ljung-Box residual, Ljung-Box kuadrat residual, ARCH-LM, Jarque-Bera.
8. **Peramalan**: QLIKE/MSE untuk varians, RMSE/MAE untuk mean, interval 95% dan plot testing.
9. **Monte Carlo**: banyak skenario, ukuran sampel dan replikasi, heatmap kinerja, boxplot, tingkat konvergensi, hasil per replikasi.
10. **Ekspor**: CSV, Excel, grafik PNG dan paket ZIP dengan seed serta metadata.

## Model matematis

Model pembangkit dan estimasi awal dibatasi pada **ARIMA(1,d,q)–GARCH(1,1)**, dengan `d ∈ {0,1}`, `q ∈ {0,1}`. Ini mencakup ARIMA(1,0,0), ARIMA(1,1,0), ARIMA(1,0,1), dan ARIMA(1,1,1) sebagai model mean. **Tidak mengklaim mendukung semua kombinasi ordo ARIMA(p,d,q).**

Untuk `W_t = Δ^d Y_t`:

```text
W_t = c + φ W_(t-1) + θ ε_(t-1) + ε_t
ε_t = sqrt(h_t) * z_t
h_t = ω + α ε_(t-1)^2 + β h_(t-1)
ω   = V * (1 - α - β)
```

Di mana `θ=0` jika `q=0`, `c=(1−φ)μ`, `α≥0`, `β≥0`, dan `α+β<1` pada rezim stasioner. `z_t` Normal(0,1) atau Student-t yang distandardisasi agar variansnya satu (df>2). Metode likelihood kondisional menggunakan residual awal nol dan varians awal berbasis sampel ketika mengestimasi; simulasi menggunakan burn-in untuk mengurangi efek inisialisasi.

**Persamaan break struktural:** pada 85% panjang dataset secara default, konstanta varians `ω` dikalikan 4 (dapat dikustomisasi). Ini sengaja menghasilkan *ketidaksesuaian spesifikasi* terhadap GARCH parameter konstan. Break dapat ditempatkan di luar atau di dalam testing dengan mengubah posisinya.

## Dua model yang dibandingkan

- **ARIMA(1,d,q)**: mean ARIMA, inovasi Gaussian dengan varians konstan. Estimasi via *conditional maximum likelihood* (CMLE), solusi OLS kondisional untuk q=0.
- **ARIMA(1,d,q)–GARCH(1,1)**: mean ARIMA dan varians GARCH ditaksir **secara gabungan** dengan CMLE numerik SciPy (`L-BFGS-B`), inovasi Normal atau Student-t terstandarisasi. Parameter positif/ketidak-negatifan dan α+β<1 dipaksakan melalui transformasi softmax.

Keduanya dilatih pada subset **training** yang sama. Parameter dibekukan setelah training. Pada **testing**, dilakukan peramalan **satu langkah ke depan secara sekuensial** menggunakan observasi sebelumnya, tanpa *rolling re-estimation*. Maka istilah yang tepat adalah **recursive / rolling one-step forecasting dengan parameter tetap**.

### Metrik

- `MSE varians = mean[(h_true − h_pred)^2]`
- `RMSE varians = sqrt(MSE varians)`
- `QLIKE = mean[log(h_pred) + h_true/h_pred]` (lebih kecil lebih baik; boleh negatif)
- RMSE dan MAE untuk ramalan mean (terpisah dari kualitas model varians)
- Cakupan interval prediksi nominal 95%, dan rata-rata lebar interval
- Bias estimasi α dan β, tingkat konvergensi, serta persentase kemenangan berdasarkan QLIKE.

Karena `h_true` diketahui pada simulasi, evaluasi varians tidak membutuhkan proksi volatilitas. **Model GARCH tidak otomatis mengungguli model mean ARIMA dalam RMSE prediksi titik**. Perbedaan utama biasanya muncul pada pemodelan varians dan interval prediksi.

## Catatan keterbatasan metodologi

- Model hanya mencakup komponen AR dengan ordo 1 dan GARCH(1,1). Model alternatif EGARCH/GJR-GARCH belum diimplementasikan.
- Data simulasi murni, bukan data BI atau Yahoo Finance; hasil tidak boleh diklaim sebagai karakteristik empiris rupiah.
- Pada model Student-t ber-df kecil, beberapa momen tinggi mungkin tidak berhingga; hasil sampling dapat bervariasi antar seed.
- Hasil yang tidak konvergen **dicatat dan dihitung dalam rasio kegagalan**. Metrik hanya dirata-ratakan atas pasangan fit yang berhasil, sehingga interpretasikan bersama tingkat keberhasilan.
- `AIC/BIC` dihitung dari likelihood kondisional pada sampel training yang sama dan tidak menggantikan evaluasi out-of-sample.
- Untuk hasil skripsi final, jalankan lebih banyak replikasi (misal 200 atau 500 per sel jika sumber daya cukup) dan uji kestabilan hasil menurut random seed.
- Hasil dalam Streamlit `st.session_state` bertahan selama sesi browser aktif saja; **unduh ZIP** setelah eksperimen.

## Struktur berkas

```text
arima_garch_lab/
├── app.py                 # Streamlit 10 tab
├── simulation_engine.py   # Simulasi, CMLE, uji, Monte Carlo
├── requirements.txt
├── README.md
├── .streamlit/config.toml
└── tests/test_engine.py
```

## Pengujian

```bash
python -m pytest -q tests
```

Pengujian memeriksa reproduksibilitas, rumus h(t), kesesuaian pembangkitan W/Y, distribusi Student-t, posisi break, pencegahan *look-ahead*, estimasi, dan pemutaran ulang Monte Carlo dari seed.

### Windows: jalankan lebih mudah

Anda juga dapat klik dua kali `JALANKAN_DI_WINDOWS.bat`. File tersebut akan memasang dependensi dan membuka server Streamlit. Membutuhkan Python yang sudah terpasang dan internet saat instalasi pertama.
