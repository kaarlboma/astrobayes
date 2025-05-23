# Exoplanet Radius Regression with Gibbs Sampling

This project applies Bayesian Linear Regression to predict exoplanet radii from host star properties using Gibbs Sampler with conjugate priors. The model is implemented from scratch in Rust, and all computations are performed without using external crates apart from linear algebra and arrays.

## Features

- Bayesian linear model with Gaussian priors and likelihood
- Gibbs Sampler implemented in Rust
- Posterior sampling and uncertainty quantification
- Exploratory data analysis and visualization in Python
- Posterior Predictive Checks

## Data

Data is sourced from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/). Key features used:
- Planet Mass
- Star Effective Temperature
- Equilibrium Temperature
- Stellar Mass
- Distance
- Stellar Metallicity


## Results

- R^2 of 0.53
- 10,000 samples generated from Gibbs Sampling
- Quantified uncertainty through credible intervals

## Output

- Features array and target array
- Prediction for the first row of the data + 95% Credible Interval
- Prediction array for all the rows of the data
- MMSE Estimates of Beta after Gibbs Sampling
- 95% Credible Interval for the predictors

## Usage

1. Clone the repo and build the Rust project:
   ```bash
   cargo build --release
   cargo run --release
