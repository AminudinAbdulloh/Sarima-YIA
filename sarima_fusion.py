# -*- coding: utf-8 -*-
"""
=============================================================================
  FUSIONSARIMA-UPG
  Prediksi Jumlah Penumpang Domestik Bandara Sultan Hasanuddin Makassar
  Metode: Seasonal AutoRegressive Integrated Moving Average (SARIMA)
=============================================================================
  Gabungan terbaik dari SarimaAir & VoluMax-Air:
    [+] Penanganan data & akurasi    -- SarimaAir
    [+] Analisis statistik lengkap   -- VoluMax-Air
    [+] Deteksi outlier otomatis     -- Fitur baru (Modified Z-score + IQR)
    [+] Visualisasi premium (8 plt)  -- VoluMax-Air + extended
    [+] Output CSV + Excel 5-sheet   -- SarimaAir + extended
    [+] requirements.txt             -- VoluMax-Air
=============================================================================
  Python   : 3.9+
  Data     : lalinud_UPG_domestik.csv
  Output   : folder output/
=============================================================================
"""
import sys, io
# Paksa stdout ke UTF-8 agar emoji/karakter khusus aman di semua platform
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 0 – IMPORT & KONFIGURASI GLOBAL
# ─────────────────────────────────────────────────────────────────────────────
import os
import glob
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-GUI backend, aman di server / tanpa display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
import seaborn as sns

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller, kpss
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pmdarima as pm
from scipy import stats
from scipy.stats import pearsonr

# ── Path & Direktori ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Auto-detect CSV (glob pattern cerdas dari SarimaAir)
_candidates = (
    glob.glob(os.path.join(BASE_DIR, "*UPG*domestik*.csv")) +
    glob.glob(os.path.join(BASE_DIR, "*domestik*.csv")) +
    glob.glob(os.path.join(BASE_DIR, "*UPG*.csv")) +
    glob.glob(os.path.join(BASE_DIR, "*.csv"))
)
if not _candidates:
    raise FileNotFoundError(
        "Tidak ada file CSV di folder:\n  {}\n"
        "Pastikan lalinud_UPG_domestik.csv ada di folder yang sama.".format(BASE_DIR)
    )
DATA_FILE = _candidates[0]

# ── Palet Warna Premium ───────────────────────────────────────────────────────
C_PRIMARY   = "#2563EB"
C_SECONDARY = "#F59E0B"
C_ACCENT    = "#10B981"
C_DANGER    = "#EF4444"
C_DARK      = "#1E293B"
C_LIGHT     = "#F8FAFC"
C_GRID      = "#E2E8F0"

plt.rcParams.update({
    "figure.facecolor":  C_LIGHT,
    "axes.facecolor":    "white",
    "axes.edgecolor":    C_GRID,
    "axes.labelcolor":   C_DARK,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.titlepad":     12,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "axes.grid":         True,
    "grid.color":        C_GRID,
    "grid.linewidth":    0.7,
    "xtick.color":       C_DARK,
    "ytick.color":       C_DARK,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "lines.linewidth":   2,
    "font.family":       "DejaVu Sans",
    "legend.framealpha": 0.9,
    "legend.fontsize":   9,
})

SEP = "=" * 70

# ── Helper Functions ──────────────────────────────────────────────────────────
def fmt(n):
    return "{:,.0f}".format(n)

def save_fig(name, dpi=200):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=plt.gcf().get_facecolor())
    plt.close()
    print("  [OK] Gambar -> {}".format(path))

def section(n, title):
    print("\n" + SEP)
    print("  STEP {} - {}".format(n, title))
    print(SEP)


# =============================================================================
# STEP 1 - LOAD & PREPROCESSING DATA
# =============================================================================
section(1, "LOAD & PREPROCESSING DATA")

print("\n  [>] File data  : {}".format(DATA_FILE))
print("  [>] Output dir : {}\n".format(OUTPUT_DIR))

df_raw = pd.read_csv(DATA_FILE)
print("  Jumlah baris  : {}".format(len(df_raw)))
print("  Kolom         : {}".format(list(df_raw.columns)))

df_raw["Periode"] = pd.to_datetime(df_raw["Periode"], format="%Y-%m")
df_raw = df_raw.sort_values("Periode").reset_index(drop=True)

df_raw["Total_Penumpang"]  = df_raw["Penumpang_Datang"]         + df_raw["Penumpang_Berangkat"]
df_raw["Total_Pesawat"]    = df_raw["Pesawat_Datang"]           + df_raw["Pesawat_Berangkat"]
df_raw["Total_Transit"]    = df_raw["Penumpang Transit_Datang"] + df_raw["Penumpang Transit_Berangkat"]

# Hapus Mei 2026 (data parsial, hanya ~10% normal)
df_raw = df_raw[df_raw["Periode"] < "2026-05-01"].copy()

ts_raw = df_raw.set_index("Periode")["Total_Penumpang"].asfreq("MS")

print("\n  Periode data  : {} - {}".format(
    ts_raw.index[0].strftime("%B %Y"), ts_raw.index[-1].strftime("%B %Y")))
print("  Jumlah obs.   : {}".format(len(ts_raw)))
print("  Min penumpang : {}".format(fmt(ts_raw.min())))
print("  Max penumpang : {}".format(fmt(ts_raw.max())))
print("  Rata-rata     : {}".format(fmt(ts_raw.mean())))
print("  Std. deviasi  : {}".format(fmt(ts_raw.std())))


# =============================================================================
# STEP 2 - DETEKSI & TREATMENT OUTLIER (Modified Z-score + IQR)
# =============================================================================
section(2, "DETEKSI & TREATMENT OUTLIER")

# Modified Z-score (Iglewicz & Hoaglin, 1993) -- robust terhadap outlier
ts_median = ts_raw.median()
ts_mad    = (ts_raw - ts_median).abs().median()
mod_z     = 0.6745 * (ts_raw - ts_median) / (ts_mad + 1e-9)
outlier_mask_z = mod_z.abs() > 3.5

# IQR x1.5 (ketat)
Q1  = ts_raw.quantile(0.25)
Q3  = ts_raw.quantile(0.75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
outlier_mask_iqr = (ts_raw < lower_fence) | (ts_raw > upper_fence)

outlier_mask  = outlier_mask_z | outlier_mask_iqr
outlier_dates = ts_raw[outlier_mask].index.tolist()

print("\n  Metode           : Modified Z-score (threshold=3.5) + IQR (x1.5)")
print("  Median           : {}".format(fmt(ts_median)))
print("  MAD              : {}".format(fmt(ts_mad)))
print("  IQR Batas Bawah  : {}".format(fmt(lower_fence)))
print("  IQR Batas Atas   : {}".format(fmt(upper_fence)))
print("\n  Outlier Terdeteksi ({} bulan):".format(len(outlier_dates)))
for d in outlier_dates:
    print("    >> {}  :  {} penumpang  (mod-Z={:.2f})".format(
        d.strftime("%B %Y"), fmt(ts_raw[d]), mod_z[d]))
if not outlier_dates:
    print("    (Tidak ada outlier signifikan)")

# Treatment: Interpolasi linear
ts = ts_raw.copy()
ts[outlier_mask] = np.nan
ts = ts.interpolate(method="time")
ts = ts.bfill().ffill()   # aman untuk pandas versi baru

print("\n  Strategi      : Interpolasi linear (time-based)")
print("  Total imputed : {} nilai".format(int(outlier_mask.sum())))
print("  Rata-rata baru: {}".format(fmt(ts.mean())))

df_raw["is_outlier"] = df_raw["Periode"].isin(outlier_dates)

# -- Plot Outlier ---------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
fig.suptitle(
    "Deteksi & Treatment Outlier - Total Penumpang Domestik UPG\n"
    "(Modified Z-score + IQR)",
    fontsize=14, fontweight="bold", color=C_DARK, y=0.99
)

ax = axes[0]
ax.plot(ts_raw.index, ts_raw.values, color=C_PRIMARY, linewidth=1.8, label="Data Asli")
ax.fill_between(ts_raw.index, ts_raw.values, alpha=0.08, color=C_PRIMARY)
ax.axhline(lower_fence, color=C_DANGER, linewidth=1.2, linestyle="--",
           label="Batas Bawah IQR ({})".format(fmt(lower_fence)))
ax.axhline(upper_fence, color=C_DANGER, linewidth=1.2, linestyle=":",
           label="Batas Atas IQR ({})".format(fmt(upper_fence)))
if outlier_dates:
    ax.scatter(outlier_dates, ts_raw[outlier_dates].values,
               color=C_DANGER, zorder=5, s=80, label="Outlier Terdeteksi")
    for d in outlier_dates:
        ax.annotate(d.strftime("%b\n%Y"), xy=(d, ts_raw[d]),
                    xytext=(0, -32), textcoords="offset points",
                    ha="center", fontsize=7.5, color=C_DANGER, fontweight="bold")
ax.set_title("Data Asli + Batas Outlier", fontweight="bold")
ax.set_ylabel("Jumlah Penumpang")
ax.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: "{:.2f}M".format(x/1e6) if x >= 1e6 else "{:.0f}K".format(x/1e3)))
ax.legend(loc="upper left", fontsize=8.5)

ax2 = axes[1]
ax2.plot(ts_raw.index, ts_raw.values, color=C_DARK, linewidth=1.6,
         linestyle="--", alpha=0.5, label="Sebelum Treatment")
ax2.plot(ts.index, ts.values, color=C_ACCENT, linewidth=2, label="Sesudah Interpolasi")
if outlier_dates:
    ax2.scatter(outlier_dates, ts[outlier_dates].values,
                color=C_ACCENT, zorder=5, s=80, marker="^", label="Nilai Interpolasi")
ax2.set_title("Sebelum vs Sesudah Treatment Outlier", fontweight="bold")
ax2.set_ylabel("Jumlah Penumpang")
ax2.set_xlabel("Periode")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: "{:.2f}M".format(x/1e6) if x >= 1e6 else "{:.0f}K".format(x/1e3)))
ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_fig("02_outlier_detection.png")


# =============================================================================
# STEP 3 - EKSPLORASI DATA (EDA)
# =============================================================================
section(3, "EKSPLORASI DATA (EDA)")

fig, axes = plt.subplots(3, 1, figsize=(14, 13))
fig.suptitle(
    "Analisis Data Penumpang Domestik\nBandara Sultan Hasanuddin Makassar (UPG)",
    fontsize=15, fontweight="bold", color=C_DARK, y=0.99
)

ax = axes[0]
ax.fill_between(ts.index, ts.values, alpha=0.15, color=C_PRIMARY)
ax.plot(ts.index, ts.values, color=C_PRIMARY, linewidth=2, marker="o",
        markersize=3.5, label="Total Penumpang (setelah treatment)")
ax.plot(ts_raw.index, ts_raw.values, color=C_DARK, linewidth=1.2,
        linestyle="--", alpha=0.4, label="Data Asli")
ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2020-12-31"),
           alpha=0.07, color=C_DANGER, label="Periode COVID-19")
ax.axvline(pd.Timestamp("2022-01-01"), color=C_ACCENT, linewidth=1.2,
           linestyle="--", label="Pemulihan 2022")
ax.set_title("Total Penumpang per Bulan (Datang + Berangkat)", fontweight="bold")
ax.set_ylabel("Jumlah Penumpang")
ax.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: "{:.1f}M".format(x/1e6) if x >= 1e6 else "{:.0f}K".format(x/1e3)))
ax.legend(loc="upper left", fontsize=8.5)

ax2 = axes[1]
ax2.plot(df_raw["Periode"], df_raw["Penumpang_Datang"],
         color=C_PRIMARY, label="Penumpang Datang", linewidth=2)
ax2.plot(df_raw["Periode"], df_raw["Penumpang_Berangkat"],
         color=C_SECONDARY, label="Penumpang Berangkat", linewidth=2)
ax2.set_title("Penumpang Datang vs Berangkat", fontweight="bold")
ax2.set_ylabel("Jumlah Penumpang")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0f}K".format(x/1e3)))
ax2.legend()

ax3 = axes[2]
ax3.bar(df_raw["Periode"], df_raw["Total_Pesawat"],
        width=20, color=C_ACCENT, alpha=0.78, label="Total Pergerakan Pesawat",
        edgecolor="white")
ax3.set_title("Total Pergerakan Pesawat per Bulan", fontweight="bold")
ax3.set_ylabel("Jumlah Pergerakan")
ax3.set_xlabel("Periode")
ax3.legend()

plt.tight_layout(rect=[0, 0, 1, 0.96])
save_fig("01_eda_overview.png")

print("\n  [i] Statistik Total Penumpang per Tahun (Data Asli):")
df_raw["Tahun"] = df_raw["Periode"].dt.year
yearly = df_raw.groupby("Tahun")["Total_Penumpang"].agg(["sum","mean","min","max"])
yearly.columns = ["Total", "Rata-rata/Bulan", "Minimum", "Maksimum"]
for col in yearly.columns:
    yearly[col] = yearly[col].apply(lambda x: fmt(x))
print(yearly.to_string())


# =============================================================================
# STEP 4 - UJI STASIONERITAS LENGKAP (ADF + KPSS)
# =============================================================================
section(4, "UJI STASIONERITAS (ADF + KPSS)")

def adf_test(series, title=""):
    res = adfuller(series.dropna(), autolag="AIC")
    ok  = res[1] <= 0.05
    print("\n  [ADF] {}".format(title))
    print("    ADF Statistic : {:.4f}".format(res[0]))
    print("    p-value       : {:.4f}".format(res[1]))
    print("    Lag           : {}".format(res[2]))
    for k, v in res[4].items():
        print("    Critical ({})  : {:.4f}".format(k, v))
    print("    => {}".format("STASIONER [OK]" if ok else "TIDAK STASIONER [!!]"))
    return ok

def kpss_test(series, title=""):
    stat, pv, _, _ = kpss(series.dropna(), regression="c", nlags="auto")
    ok = pv > 0.05
    print("\n  [KPSS] {}".format(title))
    print("    KPSS Statistic: {:.4f}".format(stat))
    print("    p-value       : {:.4f}".format(pv))
    print("    => {}".format("STASIONER [OK]" if ok else "TIDAK STASIONER [!!]"))
    return ok

adf_test(ts,                   "Total Penumpang (Original)")
kpss_test(ts,                  "Total Penumpang (Original)")
adf_test(ts.diff().dropna(),   "Total Penumpang (Diff-1)")
kpss_test(ts.diff().dropna(),  "Total Penumpang (Diff-1)")
adf_test(ts.diff(12).dropna(), "Total Penumpang (Seasonal Diff-12)")


# =============================================================================
# STEP 5 - DEKOMPOSISI TIME SERIES
# =============================================================================
section(5, "DEKOMPOSISI TIME SERIES")

decomp = seasonal_decompose(ts, model="additive", period=12, extrapolate_trend="freq")

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle(
    "Dekomposisi Time Series - Total Penumpang Domestik UPG\n"
    "(Additive | Trend | Seasonal | Residual)",
    fontsize=14, fontweight="bold", color=C_DARK
)

components = [
    (ts,              "Data Asli (Total Penumpang)", C_PRIMARY),
    (decomp.trend,    "Komponen Trend",              C_SECONDARY),
    (decomp.seasonal, "Komponen Musiman (Seasonal)", C_ACCENT),
    (decomp.resid,    "Komponen Residual",           C_DANGER),
]
for i, (data, title, color) in enumerate(components):
    ax = axes[i]
    ax.plot(data.index, data.values, color=color, linewidth=1.8)
    if i == 0:
        ax.fill_between(data.index, data.values, alpha=0.1, color=color)
    if i == 3:
        ax.axhline(0, color=C_DARK, linewidth=0.8, linestyle="--")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.set_ylabel("Nilai")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0f}K".format(x/1e3)))

axes[-1].set_xlabel("Periode")
plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig("03_decomposition.png")
print("  Dekomposisi selesai.")


# =============================================================================
# STEP 6 - ANALISIS ACF & PACF (Sebelum & Sesudah Differencing)
# =============================================================================
section(6, "ANALISIS ACF & PACF")

ts_diff = ts.diff().diff(12).dropna()

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle(
    "Analisis ACF & PACF - Pemilihan Parameter SARIMA\n"
    "Bandara Sultan Hasanuddin Makassar",
    fontsize=14, fontweight="bold", color=C_DARK
)

plot_acf(ts,      lags=30, ax=axes[0, 0], color=C_PRIMARY, alpha=0.05)
axes[0, 0].set_title("ACF - Data Asli", fontweight="bold")
plot_pacf(ts,     lags=30, ax=axes[0, 1], color=C_PRIMARY, alpha=0.05)
axes[0, 1].set_title("PACF - Data Asli", fontweight="bold")
plot_acf(ts_diff,  lags=30, ax=axes[1, 0], color=C_ACCENT, alpha=0.05)
axes[1, 0].set_title("ACF - Setelah Differencing (d=1, D=1)", fontweight="bold")
plot_pacf(ts_diff, lags=30, ax=axes[1, 1], color=C_ACCENT, alpha=0.05)
axes[1, 1].set_title("PACF - Setelah Differencing (d=1, D=1)", fontweight="bold")

for ax in axes.flat:
    ax.set_xlabel("Lag")
plt.tight_layout(rect=[0, 0, 1, 0.93])
save_fig("04_acf_pacf.png")


# =============================================================================
# STEP 7 - PEMILIHAN PARAMETER OPTIMAL (AUTO ARIMA)
# =============================================================================
section(7, "PEMILIHAN PARAMETER OPTIMAL (AUTO ARIMA)")
print("  Sedang mencari parameter terbaik... (harap tunggu)\n")

# 12 bulan terakhir sebagai test -- strategi SarimaAir (lebih andal)
TRAIN_END  = ts.index[-13]
TEST_START = ts.index[-12]

train = ts[:TRAIN_END]
test  = ts[TEST_START:]

print("  Data training : {} obs ({} - {})".format(
    len(train),
    train.index[0].strftime("%b %Y"),
    train.index[-1].strftime("%b %Y")))
print("  Data testing  : {} obs  ({} - {})".format(
    len(test),
    test.index[0].strftime("%b %Y"),
    test.index[-1].strftime("%b %Y")))

auto_model = pm.auto_arima(
    train,
    start_p=0, max_p=3,
    start_q=0, max_q=3,
    d=None,
    start_P=0, max_P=2,
    start_Q=0, max_Q=2,
    D=1, m=12,
    seasonal=True,
    information_criterion="aic",
    stepwise=True,
    trace=True,
    error_action="ignore",
    suppress_warnings=True,
)

order          = auto_model.order
seasonal_order = auto_model.seasonal_order

print("\n  [*] Parameter Terpilih:")
print("    SARIMA order          : p={}, d={}, q={}".format(*order))
print("    Seasonal order (s=12) : P={}, D={}, Q={}".format(*seasonal_order[:3]))
print("    AIC                   : {:.4f}".format(auto_model.aic()))
print("    BIC                   : {:.4f}".format(auto_model.bic()))


# =============================================================================
# STEP 8 - FITTING MODEL SARIMA
# =============================================================================
section(8, "FITTING MODEL SARIMA")

model_train = SARIMAX(
    train,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
result_train = model_train.fit(disp=False)
print(result_train.summary())


# =============================================================================
# STEP 9 - EVALUASI MODEL
# =============================================================================
section(9, "EVALUASI MODEL")

pred_test = result_train.get_forecast(steps=len(test))
pred_mean  = pred_test.predicted_mean
pred_ci    = pred_test.conf_int(alpha=0.05)

mae     = mean_absolute_error(test, pred_mean)
rmse    = np.sqrt(mean_squared_error(test, pred_mean))
mape    = np.mean(np.abs((test.values - pred_mean.values) / test.values)) * 100
akurasi = 100 - mape

sep45 = "-" * 45
print("\n  +{}+".format(sep45))
print("  |       HASIL EVALUASI MODEL SARIMA         |")
print("  +{}+".format(sep45))
print("  |  MAE    (Mean Absolute Error)  : {:>12} |".format(fmt(mae)))
print("  |  RMSE   (Root Mean Sq. Error)  : {:>12} |".format(fmt(rmse)))
print("  |  MAPE   (Mean Abs % Error)     : {:>11.2f}% |".format(mape))
print("  |  Akurasi                       : {:>11.2f}% |".format(akurasi))
print("  |  AIC                           : {:>12.2f} |".format(auto_model.aic()))
print("  |  BIC                           : {:>12.2f} |".format(auto_model.bic()))
print("  +{}+".format(sep45))


# =============================================================================
# STEP 10 - DIAGNOSTIK RESIDUAL LENGKAP
# =============================================================================
section(10, "DIAGNOSTIK RESIDUAL")

residuals = result_train.resid

lb_test = acorr_ljungbox(residuals, lags=[12], return_df=True)
lb_stat = lb_test["lb_stat"].values[0]
lb_p    = lb_test["lb_pvalue"].values[0]
print("\n  Ljung-Box Test (lag=12):")
print("    Statistik : {:.4f}".format(lb_stat))
print("    p-value   : {:.4f}".format(lb_p))
print("    => {}".format(
    "Residual TIDAK ada autokorelasi [OK]" if lb_p > 0.05
    else "Residual masih ada autokorelasi [!!]"))

_, norm_p = stats.shapiro(residuals)
print("\n  Shapiro-Wilk Normality Test:")
print("    p-value   : {:.4f}".format(norm_p))
print("    => {}".format(
    "Residual berdistribusi NORMAL [OK]" if norm_p > 0.05
    else "Tidak normal (umum untuk time series) [i]"))

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    "Diagnostik Residual Model SARIMA\nBandara Sultan Hasanuddin Makassar (UPG)",
    fontsize=14, fontweight="bold", color=C_DARK
)

axes[0, 0].plot(residuals.index, residuals.values, color=C_DANGER, linewidth=1.5)
axes[0, 0].axhline(0, color=C_DARK, linewidth=1, linestyle="--")
axes[0, 0].set_title("Residual vs Waktu", fontweight="bold")
axes[0, 0].set_ylabel("Residual")

axes[0, 1].hist(residuals, bins=15, color=C_PRIMARY, alpha=0.75, edgecolor="white", density=True)
xmin, xmax = axes[0, 1].get_xlim()
x = np.linspace(xmin, xmax, 100)
p_norm = stats.norm.pdf(x, residuals.mean(), residuals.std())
axes[0, 1].plot(x, p_norm, color=C_DANGER, linewidth=2, label="Normal dist.")
axes[0, 1].set_title("Histogram Residual", fontweight="bold")
axes[0, 1].set_xlabel("Residual")
axes[0, 1].legend()

stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title("Q-Q Plot Residual", fontweight="bold")
axes[1, 0].get_lines()[0].set(color=C_PRIMARY, markerfacecolor=C_PRIMARY, markersize=5)
axes[1, 0].get_lines()[1].set(color=C_SECONDARY, linewidth=1.8)

plot_acf(residuals, lags=24, ax=axes[1, 1], color=C_ACCENT, alpha=0.05)
axes[1, 1].set_title("ACF Residual", fontweight="bold")
axes[1, 1].set_xlabel("Lag")

plt.tight_layout(rect=[0, 0, 1, 0.93])
save_fig("05_diagnostics.png")


# =============================================================================
# STEP 11 - REFIT SEMUA DATA & PREDIKSI 12 BULAN KE DEPAN
# =============================================================================
section(11, "REFIT & PREDIKSI 12 BULAN KE DEPAN")

model_full = SARIMAX(
    ts,
    order=order,
    seasonal_order=seasonal_order,
    enforce_stationarity=False,
    enforce_invertibility=False,
)
result_full = model_full.fit(disp=False)

N_FORECAST  = 12
future_fc   = result_full.get_forecast(steps=N_FORECAST)
future_mean = future_fc.predicted_mean
future_ci   = future_fc.conf_int(alpha=0.05)

future_index = pd.date_range(
    start=ts.index[-1] + pd.DateOffset(months=1),
    periods=N_FORECAST, freq="MS"
)
future_mean.index = future_index
future_ci.index   = future_index

BULAN = {1:"Januari",2:"Februari",3:"Maret",4:"April",5:"Mei",6:"Juni",
         7:"Juli",8:"Agustus",9:"September",10:"Oktober",11:"November",12:"Desember"}

print("\n  Prediksi Jumlah Penumpang Domestik Bandara UPG:")
print("  {:<18} {:>14} {:>15} {:>15}".format(
    "Periode","Prediksi","Lower 95% CI","Upper 95% CI"))
print("  " + "-"*64)

forecast_rows = []
total_pred = 0.0
for date, val, lo, hi in zip(
    future_mean.index,
    future_mean.values,
    future_ci.iloc[:, 0].values,
    future_ci.iloc[:, 1].values,
):
    label   = "{} {}".format(BULAN[date.month], date.year)
    lo_disp = max(lo, 0)
    print("  {:<18} {:>14} {:>15} {:>15}".format(
        label, fmt(val), fmt(lo_disp), fmt(hi)))
    forecast_rows.append({
        "Periode":    date.strftime("%Y-%m"),
        "Bulan":      label,
        "Prediksi":   round(val),
        "Lower_95CI": round(lo),
        "Upper_95CI": round(hi),
    })
    total_pred += val

print("\n  Total Prediksi 12 Bulan : {} penumpang".format(fmt(total_pred)))


# =============================================================================
# STEP 12 - VISUALISASI PREDIKSI UTAMA (2 Panel)
# =============================================================================
section(12, "VISUALISASI PREDIKSI UTAMA")

fitted_full = result_full.fittedvalues

fig, axes = plt.subplots(2, 1, figsize=(16, 14),
                         gridspec_kw={"height_ratios": [2, 1.2]})
fig.suptitle(
    "Prediksi Jumlah Penumpang Domestik\n"
    "Bandara Sultan Hasanuddin Makassar (UPG) - SARIMA{}x{}[s=12]".format(
        order, seasonal_order[:3]),
    fontsize=15, fontweight="bold", color=C_DARK, y=0.99
)

ax = axes[0]
ax.fill_between(ts.index, ts.values, alpha=0.08, color=C_PRIMARY)
ax.plot(train.index, train.values, color=C_PRIMARY, linewidth=2,
        label="Data Training", zorder=3)
ax.plot(test.index, test.values, color=C_DARK, linewidth=2,
        linestyle="--", label="Data Aktual (Test)", zorder=3)
ax.plot(fitted_full.index, fitted_full.values, color=C_ACCENT,
        linewidth=1.3, linestyle=":", label="Fitted SARIMA", zorder=4, alpha=0.8)
ax.plot(pred_mean.index, pred_mean.values, color=C_SECONDARY, linewidth=2,
        label="Prediksi Test (MAPE={:.2f}%)".format(mape), zorder=4)
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1],
                alpha=0.18, color=C_SECONDARY, label="95% CI (Test)")
ax.plot(future_mean.index, future_mean.values, color=C_ACCENT, linewidth=2.5,
        marker="o", markersize=5.5, label="Prediksi ke Depan", zorder=5)
ax.fill_between(future_ci.index, future_ci.iloc[:, 0], future_ci.iloc[:, 1],
                alpha=0.18, color=C_ACCENT, label="95% CI (Forecast)")
ax.axvline(test.index[0], color=C_SECONDARY, linewidth=1.2, linestyle=":", alpha=0.8)
ax.axvline(future_mean.index[0], color=C_ACCENT, linewidth=1.5, linestyle="-.", alpha=0.9)
ax.annotate("<- Test",     xy=(test.index[0],        ts.max() * 0.97), fontsize=8.5,
            color=C_SECONDARY, fontweight="bold")
ax.annotate("<- Forecast", xy=(future_mean.index[0], ts.max() * 0.97), fontsize=8.5,
            color=C_ACCENT, fontweight="bold")
ax.set_title("Historis & Prediksi  |  MAPE: {:.2f}%  |  Akurasi: {:.2f}%".format(
    mape, akurasi), fontweight="bold")
ax.set_ylabel("Jumlah Penumpang")
ax.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: "{:.2f}M".format(x/1e6) if x >= 1e6 else "{:.0f}K".format(x/1e3)))
ax.legend(loc="upper left", ncol=2, fontsize=8.5)
ax.grid(True, alpha=0.4)

ax2 = axes[1]
hist_ctx = ts.iloc[-24:]
ax2.fill_between(hist_ctx.index, hist_ctx.values, alpha=0.08, color=C_PRIMARY)
ax2.plot(hist_ctx.index, hist_ctx.values, color=C_PRIMARY, linewidth=2,
         label="Historis (24 bln terakhir)", zorder=3)
ax2.plot(future_mean.index, future_mean.values, color=C_ACCENT, linewidth=2.5,
         marker="o", markersize=7, zorder=5, label="Prediksi")
ax2.fill_between(future_ci.index, future_ci.iloc[:, 0], future_ci.iloc[:, 1],
                 alpha=0.22, color=C_ACCENT, label="Interval Kepercayaan 95%")
for i, (date, val) in enumerate(zip(future_mean.index, future_mean.values)):
    offset = 14 if i % 2 == 0 else -22
    ax2.annotate("{:.0f}K".format(val/1e3),
                 xy=(date, val), xytext=(0, offset), textcoords="offset points",
                 fontsize=8, fontweight="bold", color=C_DARK, ha="center",
                 arrowprops=dict(arrowstyle="-", color=C_DARK, lw=0.6))
ax2.axvline(future_mean.index[0], color=C_ACCENT, linewidth=1.5, linestyle="--", alpha=0.7)
ax2.set_title("Close-up: Prediksi 12 Bulan ke Depan (+ 24 Bulan Historis)",
              fontweight="bold")
ax2.set_ylabel("Jumlah Penumpang")
ax2.set_xlabel("Periode")
ax2.yaxis.set_major_formatter(plt.FuncFormatter(
    lambda x, _: "{:.2f}M".format(x/1e6) if x >= 1e6 else "{:.0f}K".format(x/1e3)))
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.4)

plt.tight_layout(rect=[0, 0, 1, 0.97])
save_fig("06_forecast_main.png", dpi=220)


# =============================================================================
# STEP 13 - VALIDASI: AKTUAL VS PREDIKSI
# =============================================================================
section(13, "VALIDASI AKTUAL VS PREDIKSI")

r, r_p = pearsonr(test.values, pred_mean.values)
print("  Pearson R     : {:.4f}".format(r))
print("  R-squared     : {:.4f}".format(r**2))
print("  p-value (R)   : {:.4f}".format(r_p))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle(
    "Validasi Model: Aktual vs Prediksi (Periode Test)\n"
    "Bandara Sultan Hasanuddin Makassar",
    fontsize=13, fontweight="bold", color=C_DARK
)

ax = axes[0]
ax.plot(test.index, test.values, color=C_DARK, linewidth=2.2,
        label="Aktual", marker="o", markersize=5.5)
ax.plot(pred_mean.index, pred_mean.values, color=C_SECONDARY, linewidth=2.2,
        label="Prediksi", marker="s", markersize=5.5, linestyle="--")
ax.fill_between(pred_ci.index, pred_ci.iloc[:, 0], pred_ci.iloc[:, 1],
                alpha=0.15, color=C_SECONDARY)
ax.set_title("Line Chart: Aktual vs Prediksi", fontweight="bold")
ax.set_ylabel("Jumlah Penumpang")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0f}K".format(x/1e3)))
ax.legend()
ax.tick_params(axis="x", rotation=35)

ax2 = axes[1]
ax2.scatter(test.values, pred_mean.values, color=C_PRIMARY, alpha=0.85, s=80, zorder=3)
lims = [min(test.min(), pred_mean.min()) * 0.92,
        max(test.max(), pred_mean.max()) * 1.06]
ax2.plot(lims, lims, color=C_DANGER, linewidth=1.5, linestyle="--",
         label="Garis Sempurna (45 deg)")
ax2.set_xlim(lims); ax2.set_ylim(lims)
ax2.set_title("Scatter: Aktual vs Prediksi", fontweight="bold")
ax2.set_xlabel("Nilai Aktual")
ax2.set_ylabel("Nilai Prediksi")
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0f}K".format(x/1e3)))
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0f}K".format(x/1e3)))
ax2.text(0.05, 0.90,
         "R  = {:.4f}\nR2 = {:.4f}\nMAPE = {:.2f}%".format(r, r**2, mape),
         transform=ax2.transAxes, fontsize=10, color=C_DARK,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.87))
ax2.legend()

plt.tight_layout(rect=[0, 0, 1, 0.93])
save_fig("07_actual_vs_predicted.png")


# =============================================================================
# STEP 14 - POLA MUSIMAN BULANAN
# =============================================================================
section(14, "POLA MUSIMAN BULANAN")

monthly_avg = ts.groupby(ts.index.month).mean()
month_names = [BULAN[m] for m in monthly_avg.index]
avg_all     = monthly_avg.mean()
colors_bar  = [C_ACCENT if v >= avg_all else C_SECONDARY for v in monthly_avg.values]

fig, ax = plt.subplots(figsize=(13, 5))
fig.suptitle(
    "Pola Musiman Bulanan - Rata-rata Total Penumpang Domestik UPG",
    fontsize=13, fontweight="bold", color=C_DARK
)
bars = ax.bar(month_names, monthly_avg.values, color=colors_bar,
              alpha=0.85, edgecolor="white", linewidth=0.5)
ax.axhline(avg_all, color=C_DANGER, linewidth=1.5, linestyle="--",
           label="Rata-rata: {}".format(fmt(avg_all)))
for bar, val in zip(bars, monthly_avg.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 4000,
            "{:.0f}K".format(val/1e3),
            ha="center", va="bottom", fontsize=9, fontweight="bold", color=C_DARK)

above = mpatches.Patch(color=C_ACCENT,    label="Di atas rata-rata")
below = mpatches.Patch(color=C_SECONDARY, label="Di bawah rata-rata")
ax.legend(handles=[above, below,
                   plt.Line2D([0], [0], color=C_DANGER, linestyle="--",
                              label="Rata-rata: {}".format(fmt(avg_all)))],
          loc="lower right")
ax.set_ylabel("Rata-rata Penumpang")
ax.set_xlabel("Bulan")
ax.tick_params(axis="x", rotation=20)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: "{:.0f}K".format(x/1e3)))

plt.tight_layout(rect=[0, 0, 1, 0.95])
save_fig("08_seasonal_pattern.png")


# =============================================================================
# STEP 15 - SIMPAN OUTPUT & RINGKASAN AKHIR
# =============================================================================
section(15, "SIMPAN OUTPUT & RINGKASAN AKHIR")

df_eval = pd.DataFrame({
    "Periode":     test.index.strftime("%Y-%m"),
    "Aktual":      test.values.astype(int),
    "Prediksi":    pred_mean.values.astype(int),
    "CI_Bawah_95": pred_ci.iloc[:, 0].values.astype(int),
    "CI_Atas_95":  pred_ci.iloc[:, 1].values.astype(int),
    "Error":       (test.values - pred_mean.values).astype(int),
    "APE (%)":     np.round(np.abs((test.values - pred_mean.values)
                                    / test.values) * 100, 2),
})

df_forecast = pd.DataFrame(forecast_rows)

df_outlier = pd.DataFrame({
    "Periode":       [d.strftime("%Y-%m") for d in outlier_dates],
    "Bulan":         [d.strftime("%B %Y") for d in outlier_dates],
    "Nilai_Asli":    [int(ts_raw[d]) for d in outlier_dates],
    "Nilai_Imputed": [int(ts[d])     for d in outlier_dates],
    "Selisih":       [int(ts[d] - ts_raw[d]) for d in outlier_dates],
})

df_summary = pd.DataFrame({
    "Parameter": [
        "Model", "Order (p,d,q)", "Seasonal Order (P,D,Q,s)",
        "AIC", "BIC", "MAE", "RMSE", "MAPE (%)", "Akurasi (%)",
        "Pearson R", "R-squared",
        "Data Training (obs)", "Data Testing (obs)",
        "Outlier Ditemukan", "Total Prediksi 12 Bln"
    ],
    "Nilai": [
        "SARIMA{}x{}[s=12]".format(order, seasonal_order[:3]),
        str(order), str(seasonal_order),
        round(auto_model.aic(), 2), round(auto_model.bic(), 2),
        round(mae, 0), round(rmse, 0),
        round(mape, 2), round(akurasi, 2),
        round(r, 4), round(r**2, 4),
        len(train), len(test),
        len(outlier_dates),
        round(total_pred, 0)
    ]
})

eval_csv     = os.path.join(OUTPUT_DIR, "evaluasi_test.csv")
forecast_csv = os.path.join(OUTPUT_DIR, "prediksi_kedepan.csv")
df_eval.to_csv(eval_csv,     index=False, encoding="utf-8-sig")
df_forecast.to_csv(forecast_csv, index=False, encoding="utf-8-sig")
print("  [OK] evaluasi_test.csv    -> {}".format(eval_csv))
print("  [OK] prediksi_kedepan.csv -> {}".format(forecast_csv))

xlsx_path = os.path.join(OUTPUT_DIR, "hasil_fusion_sarima.xlsx")
with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
    df_raw[["Periode","Total_Penumpang","Penumpang_Datang",
            "Penumpang_Berangkat","Total_Pesawat","is_outlier"]]\
        .to_excel(writer, sheet_name="Data_Historis", index=False)
    df_outlier.to_excel(writer,  sheet_name="Outlier_Log",    index=False)
    df_eval.to_excel(writer,     sheet_name="Evaluasi_Test",  index=False)
    df_forecast.to_excel(writer, sheet_name="Prediksi_12_Bln",index=False)
    df_summary.to_excel(writer,  sheet_name="Ringkasan_Model",index=False)
print("  [OK] hasil_fusion_sarima.xlsx -> {}".format(xlsx_path))

# -- Ringkasan Akhir di Console ------------------------------------------------
print("\n" + SEP)
print("  RINGKASAN HASIL AKHIR - FusionSARIMA-UPG")
print(SEP)
print()
print("  Dataset     : Bandara Sultan Hasanuddin Makassar (UPG)")
print("  Variabel    : Total Penumpang Domestik (Datang + Berangkat)")
print("  Periode     : {} - {}".format(
    ts.index[0].strftime("%B %Y"), ts.index[-1].strftime("%B %Y")))
print("  Observasi   : {} bulan".format(len(ts)))
print()
print("  Outlier     : {} bulan dideteksi & diinterpolasi".format(len(outlier_dates)))
if outlier_dates:
    print("               {}".format(", ".join(d.strftime("%b %Y") for d in outlier_dates)))
print()
print("  Model       : SARIMA{} x {}[s=12]".format(order, seasonal_order[:3]))
print("  Data Train  : {} obs ({} - {})".format(
    len(train), train.index[0].strftime("%b %Y"), train.index[-1].strftime("%b %Y")))
print("  Data Test   : {} obs  ({} - {})".format(
    len(test), test.index[0].strftime("%b %Y"), test.index[-1].strftime("%b %Y")))
print()
print("  Evaluasi:")
print("    MAE        : {} penumpang".format(fmt(mae)))
print("    RMSE       : {} penumpang".format(fmt(rmse)))
print("    MAPE       : {:.2f}%".format(mape))
print("    Akurasi    : {:.2f}%".format(akurasi))
print("    Pearson R  : {:.4f}".format(r))
print("    R-squared  : {:.4f}".format(r**2))
print("    AIC        : {:.2f}".format(auto_model.aic()))
print("    BIC        : {:.2f}".format(auto_model.bic()))
print()
print("  Prediksi 12 Bulan ke Depan:")
for row in forecast_rows:
    lo = max(row["Lower_95CI"], 0)
    print("    {:<18} : {:>10}  [{:>10} - {:>10}]".format(
        row["Bulan"], fmt(row["Prediksi"]), fmt(lo), fmt(row["Upper_95CI"])))
print()
print("  Total : {} penumpang".format(fmt(total_pred)))
print()
print("  Output (folder output/):")
for f in ["01_eda_overview.png","02_outlier_detection.png","03_decomposition.png",
          "04_acf_pacf.png","05_diagnostics.png","06_forecast_main.png",
          "07_actual_vs_predicted.png","08_seasonal_pattern.png",
          "evaluasi_test.csv","prediksi_kedepan.csv",
          "hasil_fusion_sarima.xlsx (5 sheet)"]:
    print("    - {}".format(f))
print()
print("  [DONE] FusionSARIMA-UPG selesai! Semua output tersimpan di output/")
print()
print(SEP)
