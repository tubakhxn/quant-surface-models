import sys
import subprocess
import importlib

def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# List of required packages (pip names)
REQUIRED_PACKAGES = [
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("sklearn", "scikit-learn"),
    ("scipy", "scipy"),
]

for pkg, pip_name in REQUIRED_PACKAGES:
    install_and_import(pip_name)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from scipy.ndimage import gaussian_filter
from mpl_toolkits.mplot3d import Axes3D

# 1. DATA GENERATION
np.random.seed(42)
N = 1000

dates = pd.date_range("2020-01-01", periods=N, freq="B")
price = np.cumprod(1 + np.random.normal(0, 0.01, N)) * 100

df = pd.DataFrame({"date": dates, "price": price})
df.set_index("date", inplace=True)

# Returns
RET_LAG = 1
df["return"] = df["price"].pct_change(RET_LAG)

# Rolling volatility
VOL_WIN = 20
df["volatility"] = df["return"].rolling(VOL_WIN).std()

# Momentum (past 10-day return)
MOM_WIN = 10
df["momentum"] = df["price"].pct_change(MOM_WIN)

# Mean reversion (distance from 20-day MA)
MA_WIN = 20
df["ma"] = df["price"].rolling(MA_WIN).mean()
df["mean_reversion"] = df["price"] - df["ma"]

# Target: future return (5 days ahead), non-linear interaction
FUTURE = 5
mom = df["momentum"].shift(-FUTURE)
vol = df["volatility"].shift(-FUTURE)
mr = df["mean_reversion"].shift(-FUTURE)
noise = np.random.normal(0, 0.01, N)
df["future_return"] = (
    0.5 * mom +
    0.3 * np.sin(3 * vol) +
    0.2 * np.sign(-mr) * np.abs(mr) ** 0.7 +
    0.15 * mom * vol +
    0.1 * mom * mr +
    0.1 * vol * mr +
    noise
)

# Drop NaNs
features = ["momentum", "volatility", "mean_reversion"]
df = df.dropna(subset=features + ["future_return"])

# 2. MODELING
X = df[features].values
y = df["future_return"].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestRegressor(n_estimators=200, max_depth=5, random_state=42)
model.fit(X_scaled, y)

y_pred = model.predict(X_scaled)

# 3. VISUALIZATION
sns.set(style="whitegrid", font_scale=1.1)

# A. SCATTER + SMOOTH CURVE (momentum vs future return)
plt.figure(figsize=(7, 5))
sns.scatterplot(x=df["momentum"], y=df["future_return"], alpha=0.3, color="tab:blue", label="Data")
# Smooth curve
sort_idx = np.argsort(df["momentum"].values)
x_sorted = df["momentum"].values[sort_idx]
y_sorted = df["future_return"].values[sort_idx]
y_smooth = gaussian_filter(y_sorted, sigma=5)
plt.plot(x_sorted, y_smooth, color="crimson", lw=2, label="Smooth Curve")
plt.xlabel("Momentum (10-day return)")
plt.ylabel("Future Return (5-day ahead)")
plt.title("Momentum vs Future Return")
plt.legend()
plt.tight_layout()

# B. SECOND CURVE (volatility vs future return)
plt.figure(figsize=(7, 5))
sns.scatterplot(x=df["volatility"], y=df["future_return"], alpha=0.3, color="tab:green", label="Data")
sort_idx = np.argsort(df["volatility"].values)
x_sorted = df["volatility"].values[sort_idx]
y_sorted = df["future_return"].values[sort_idx]
y_smooth = gaussian_filter(y_sorted, sigma=5)
plt.plot(x_sorted, y_smooth, color="darkorange", lw=2, label="Smooth Curve")
plt.xlabel("Volatility (20-day rolling std)")
plt.ylabel("Future Return (5-day ahead)")
plt.title("Volatility vs Future Return")
plt.legend()
plt.tight_layout()

# C. 3D SURFACE (momentum, volatility -> predicted return)
plt.figure(figsize=(9, 7))
ax = plt.subplot(111, projection="3d")

# Meshgrid for surface
mom_grid = np.linspace(df["momentum"].min(), df["momentum"].max(), 40)
vol_grid = np.linspace(df["volatility"].min(), df["volatility"].max(), 40)
mg, vg = np.meshgrid(mom_grid, vol_grid)

# Use mean value for mean_reversion
mr_mean = df["mean_reversion"].mean()
X_surf = np.column_stack([
    mg.ravel(),
    vg.ravel(),
    np.full(mg.size, mr_mean)
])
X_surf_scaled = scaler.transform(X_surf)
Z = model.predict(X_surf_scaled).reshape(mg.shape)
Z_smooth = gaussian_filter(Z, sigma=1.5)

surf = ax.plot_surface(mg, vg, Z_smooth, cmap="viridis", alpha=0.85, edgecolor='none')
contour = ax.contourf(mg, vg, Z_smooth, zdir='z', offset=Z_smooth.min()-0.02, cmap="viridis", alpha=0.5)
ax.set_xlabel("Momentum")
ax.set_ylabel("Volatility")
ax.set_zlabel("Predicted Return")
ax.set_title("Alpha Interaction Surface (Momentum, Volatility → Return)")
fig = plt.gcf()
fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.1, label="Predicted Return")
plt.tight_layout()

# D. INTERACTION HEATMAP (momentum x volatility)
plt.figure(figsize=(7, 6))
heatmap = plt.gca()
Z_heat = Z_smooth
sns.heatmap(Z_heat, xticklabels=6, yticklabels=6, cmap="coolwarm", cbar_kws={"label": "Predicted Return"}, ax=heatmap)
heatmap.set_xlabel("Momentum")
heatmap.set_ylabel("Volatility")
heatmap.set_title("Interaction Heatmap: Momentum vs Volatility")
plt.tight_layout()

# E. FEATURE IMPORTANCE BAR CHART
plt.figure(figsize=(6, 4))
perm = permutation_importance(model, X_scaled, y, n_repeats=10, random_state=42)
importances = perm.importances_mean
feat_names = features
sns.barplot(x=importances, y=feat_names, palette="Blues_r")
plt.xlabel("Importance")
plt.title("Feature Importance (Permutation)")
plt.tight_layout()

# FINAL DISPLAY
plt.tight_layout()
plt.show()

print("Done. Alpha interaction surface complete.")
