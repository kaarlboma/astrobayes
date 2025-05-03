use crate::BayesianLinearRegressor;
use ndarray::array;

/*
Testing new() method
    - Dimensions of both data and model parameters
    - Storing the attributes
 */
#[test]
fn test_new_with_defaults() {
    let x = array![[1.0, 2.0], [3.0, 4.0]];
    let y = array![1.0, 2.0];

    let model = BayesianLinearRegressor::new(
        x.clone(),
        y.clone(),
        None, None, None, None, None, None, None
    );

    assert_eq!(model.x.nrows(), 2);
    assert_eq!(model.x.ncols(), 3); // With the intercept term
    assert_eq!(model.y, y);
    assert_eq!(model.beta.len(), 3);
    assert_eq!(model.beta_prior_mean.len(), 3);
    assert_eq!(model.beta_prior_cov.shape(), &[3, 3]);
}

/*
Testing if the sample variances are positive for every sample
 */
#[test]
fn test_sample_variance_positive() {
    let x = array![[1.0], [2.0]];
    let y = array![1.0, 2.0];
    let mut model = BayesianLinearRegressor::new(x, y, None, None, None, None, None, None, None);

    let sample = model.sample_variance(2.0, 2.0);
    assert!(sample > 0.0);
}

/*
Test if the sample prior method provides a positive variance and the size
of beta is correct
 */
#[test]
fn test_sample_prior_updates_model() {
    let x = array![[1.0], [2.0]];
    let y = array![1.0, 2.0];
    let mut model = BayesianLinearRegressor::new(x, y, None, None, None, None, None, None, None);
    model.sample_prior();

    assert_eq!(model.beta.len(), 2); // intercept + 1 feature
    assert!(model.variance > 0.0);
}

/*
Test if the MCMC is working
    - Beta samples are being populated
    - Variance samples are being populated
    - Correct number of samples for both
 */
#[test]
fn test_run_mcmc_produces_samples() {
    let x = array![[1.0], [2.0], [3.0], [4.0]];
    let y = array![2.0, 3.9, 6.1, 7.8];
    let mut model = BayesianLinearRegressor::new(
        x, y,
        None, None, None, None,
        Some(100), Some(10), Some(5)
    );
    model.run_mcmc();

    assert!(!model.beta_samples.is_empty());
    assert!(!model.variance_samples.is_empty());
    assert_eq!(model.beta_samples.len(), 100);
    assert_eq!(model.variance_samples.len(), 100);
}
