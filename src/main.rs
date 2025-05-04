use fileio::fileio::read_csv;
mod fileio;
use blr::blr::BayesianLinearRegressor;
mod blr;

#[cfg(test)]
mod tests;

fn main() {
    // Read features and targets from the CSV file
    let (features, targets) = read_csv("data/cleaned_exoplanet_data.csv").unwrap(); // Use `?` to propagate errors
    
    // Print the features and targets
    println!("Features:\n{}", features);
    println!("Targets:\n{}", targets);
    println!("------------------------------------------------");

    // Initialize the Bayesian Linear Regressor with the data
    let mut model = BayesianLinearRegressor::new(features.clone(), targets.clone(), None, None, None, None, None);
    
    // Run MCMC to sample the posterior
    model.run_gibbs_sampling();

    // Export the beta values as csv
    model.export_beta_samples_to_csv("beta_samples.csv");
    
    // Predicting with just one observation (first row of the data)
    // Printing 95% credible interval too
    let x_test = features.row(0).to_owned();
    let (mean_pred, lower, upper) = model.predict_single_with_credible_interval(x_test);
    println!("Prediction for first data point: {:.3}, 95% Credible Interval: ({:.3}, {:.3})", mean_pred, lower, upper);
    println!("");

    // Predicting with several data points (the full data)
    let predictions = model.predict_multiple(features.clone());
    println!("Predictions for multiple data points:\n{}", predictions);
    println!("");

    // Get the MMSE estimates for beta
    println!("MMSE Estimates for Beta: {:?}", model.beta.clone().into_raw_vec());
    println!("");

    // Get the 95% credible intervals for each predictor
    let intervals = model.get_beta_credible_interval();
    println!("{:?}", intervals);
    println!("");

    println!("R^2 for the model: {}", model.r_squared());
}
