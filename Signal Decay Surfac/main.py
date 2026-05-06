import sys
import subprocess
import importlib

# --- AUTO-INSTALL REQUIRED PACKAGES ---
REQUIRED = ['numpy', 'matplotlib', 'seaborn', 'scikit-learn']
for pkg in REQUIRED:
    try:
        importlib.import_module(pkg if pkg != 'scikit-learn' else 'sklearn')
    except ImportError:
        print(f'Installing {pkg}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg])

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from scipy.interpolate import griddata

# --- DATA GENERATION ---
np.random.seed(42)
N = 300
T = np.linspace(0, 10, N)
alpha_signal = np.random.normal(loc=1.0, scale=0.3, size=N)
decay_rate = 0.4
signal = alpha_signal * np.exp(-decay_rate * T) + np.random.normal(0, 0.05, N)

# Future returns: correlated with signal, decays with time, plus noise
future_return = 0.8 * signal + 0.2 * np.exp(-0.2 * T) + np.random.normal(0, 0.08, N)

# --- MODELING ---
X = np.column_stack([signal, T])
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
reg = LinearRegression().fit(X_poly, future_return)
pred_return = reg.predict(X_poly)

# --- VISUALIZATION ---
# 1. Scatter + Smooth Regression Curve
plt.figure(figsize=(7, 5))
sns.scatterplot(x=signal, y=future_return, alpha=0.6, label='Data', color='tab:blue')
# Fit smooth curve (polyfit)
sort_idx = np.argsort(signal)
signal_sorted = signal[sort_idx]
return_sorted = future_return[sort_idx]
coefs = np.polyfit(signal_sorted, return_sorted, 2)
poly_curve = np.poly1d(coefs)
plt.plot(signal_sorted, poly_curve(signal_sorted), color='red', lw=2, label='Smooth Curve')
plt.xlabel('Signal Strength')
plt.ylabel('Future Return')
plt.title('Signal Strength vs Future Return')
plt.legend()

# 2. Decay Curve
plt.figure(figsize=(7, 5))
plt.plot(T, signal, color='purple', lw=2)
plt.xlabel('Time')
plt.ylabel('Signal Strength')
plt.title('Signal Decay Over Time')

# 3. 3D Surface: Predicted Return vs (Time, Signal)
from mpl_toolkits.mplot3d import Axes3D
plt.figure(figsize=(8, 6))
ax = plt.axes(projection='3d')
# Meshgrid for smooth surface
grid_t = np.linspace(T.min(), T.max(), 40)
grid_s = np.linspace(signal.min(), signal.max(), 40)
TT, SS = np.meshgrid(grid_t, grid_s)
X_grid = np.column_stack([SS.ravel(), TT.ravel()])
X_grid_poly = poly.transform(X_grid)
Z_pred = reg.predict(X_grid_poly).reshape(TT.shape)
# Plot surface
surf = ax.plot_surface(TT, SS, Z_pred, cmap='viridis', alpha=0.8, edgecolor='none')
ax.set_xlabel('Time')
ax.set_ylabel('Signal Strength')
ax.set_zlabel('Predicted Return')
ax.set_title('Predicted Return Surface')
plt.colorbar(surf, shrink=0.5, aspect=10, pad=0.1)

# 4. Heatmap: Correlation Matrix
plt.figure(figsize=(5, 4))
data = np.column_stack([signal, T, future_return])
corr = np.corrcoef(data, rowvar=False)
sns.heatmap(corr, annot=True, cmap='coolwarm', xticklabels=['Signal', 'Time', 'Return'], yticklabels=['Signal', 'Time', 'Return'])
plt.title('Correlation Heatmap')

# --- DISPLAY FIX ---
plt.tight_layout()
plt.show()

print('Done. Signal decay analysis complete.')
