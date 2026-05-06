## Dev/Creator = tubakhxn

# Experimental Quant Research Systems

This repository contains three experimental quantitative trading systems focused on modeling, visualization, and research-style analysis using synthetic data.

Each project emphasizes clean visual outputs such as smooth curves, 3D surfaces, and heatmaps to explore how financial signals behave in a controlled environment.

These systems are designed for experimentation and learning, not real trading.

---

# 1. Alpha Interaction Surface

## Overview

Models how multiple alpha factors interact (momentum, volatility, mean reversion) to influence future returns using non-linear relationships.

## What it does

* Generates synthetic financial data
* Creates multiple alpha factors
* Fits a machine learning model
* Visualizes:

  * smooth regression curves
  * 3D interaction surface
  * heatmaps
  * feature importance

## How to run

pip install numpy pandas matplotlib seaborn scikit-learn scipy
python main.py

---

# 2. Regime Transition Curve System

## Overview

Analyzes how market behavior changes across volatility regimes using smooth curve fitting and structured visualization.

## What it does

* Generates synthetic price data
* Computes returns and volatility
* Identifies regimes (low, medium, high)
* Visualizes:

  * scatter plots with smooth curves
  * regime-colored data
  * volatility transitions over time

## How to run

pip install numpy pandas matplotlib seaborn scikit-learn
python regime_transition_curve_system.py

---

# 3. Signal Decay Surface

## Overview

Explores how trading signals lose predictive power over time using decay modeling and regression.

## What it does

* Simulates alpha signal decay
* Models relationship with future returns
* Visualizes:

  * decay curves
  * scatter + smooth regression
  * 3D surface plots
  * correlation heatmaps

## How to run

pip install numpy matplotlib seaborn scikit-learn
python main.py

---

# Disclaimer

These projects are for educational and research purposes only.
They are not intended for financial advice, trading, or investment use.

---

# Notes

* All data is synthetically generated
* No external datasets required
* Runs entirely on CPU
* Built for experimentation and visualization

---

# License

MIT License (recommended)
