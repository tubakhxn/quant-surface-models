import sys
import subprocess
import importlib

def install_and_import(package):
    try:
        importlib.import_module(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    finally:
        globals()[package] = importlib.import_module(package)

for pkg in ["numpy", "pandas", "matplotlib", "seaborn", "sklearn"]:
    install_and_import(pkg)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import KBinsDiscretizer

# 1. Generate synthetic price series
def generate_price_series(n=1000, seed=42):
    np.random.seed(seed)
    # Simulate regime switching volatility
    regimes = np.random.choice([0.01, 0.03, 0.07], size=n, p=[0.5, 0.3, 0.2])
    returns = np.random.normal(loc=0.0005, scale=regimes)
    price = 100 * np.exp(np.cumsum(returns))
    return price, returns, regimes

# 2. Compute returns and rolling volatility
def compute_features(price, returns, window=30):
    df = pd.DataFrame({
        'price': price,
        'returns': returns
    })
    df['volatility'] = df['returns'].rolling(window).std()
    df['volatility'] = df['volatility'].bfill()
    return df

# 3. Label regimes
def label_regimes(volatility):
    # Use quantiles for regime boundaries
    bins = np.quantile(volatility, [0, 1/3, 2/3, 1])
    labels = ['Low', 'Medium', 'High']
    regime = pd.cut(volatility, bins=bins, labels=labels, include_lowest=True)
    return regime

# 4. Fit smooth curve (LOWESS or polynomial)
def fit_smooth_curve(x, y, method='lowess'):
    if method == 'lowess':
        try:
            from statsmodels.nonparametric.smoothers_lowess import lowess
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "statsmodels"])
            from statsmodels.nonparametric.smoothers_lowess import lowess
        smoothed = lowess(y, x, frac=0.2, return_sorted=True)
        return smoothed[:,0], smoothed[:,1]
    else:
        # Polynomial fit (degree 3)
        coeffs = np.polyfit(x, y, 3)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(np.min(x), np.max(x), 200)
        y_fit = poly(x_fit)
        return x_fit, y_fit

# MAIN EXECUTION
if __name__ == "__main__":
    # Data generation
    price, returns, true_regimes = generate_price_series(n=1000)
    df = compute_features(price, returns)
    df['regime'] = label_regimes(df['volatility'])
    df['time'] = np.arange(len(df))

    # 1. SCATTER + SMOOTH CURVE (MAIN)
    plt.figure(figsize=(8,6))
    palette = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
    sns.scatterplot(x='volatility', y='returns', hue='regime', data=df, palette=palette, alpha=0.6, edgecolor=None)
    x_smooth, y_smooth = fit_smooth_curve(df['volatility'], df['returns'], method='lowess')
    plt.plot(x_smooth, y_smooth, color='blue', linewidth=2, label='Smooth Curve')
    plt.xlabel('Volatility (Rolling Std)')
    plt.ylabel('Returns')
    plt.title('Returns vs Volatility with Regime Coloring')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 2. Rolling Volatility Plot
    plt.figure(figsize=(10,4))
    for regime, color in palette.items():
        mask = df['regime'] == regime
        plt.plot(df['time'][mask], df['volatility'][mask], '.', label=regime, color=color, alpha=0.7)
    plt.xlabel('Time')
    plt.ylabel('Volatility (Rolling Std)')
    plt.title('Rolling Volatility and Regime Transitions')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 3. OPTIONAL 3D Plot
    try:
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111, projection='3d')
        for regime, color in palette.items():
            mask = df['regime'] == regime
            ax.scatter(df['time'][mask], df['volatility'][mask], df['returns'][mask], c=color, label=regime, alpha=0.5)
        ax.set_xlabel('Time')
        ax.set_ylabel('Volatility')
        ax.set_zlabel('Returns')
        ax.set_title('3D Regime Transition Curve (Optional)')
        ax.legend()
        plt.tight_layout()
        plt.show()
    except Exception:
        pass

    print("Done. Regime transition analysis complete.")
