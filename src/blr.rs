// Module containing the core functionality of the model
pub mod blr {
    use ndarray::{concatenate, Array, Array1, Array2, Axis, array};
    use rand_distr::{Gamma, Normal, Distribution};
    use rand::thread_rng;
    use ndarray_linalg::cholesky::Cholesky;
    use ndarray_linalg::{UPLO, Inverse};
    use std::fs::File;
    use std::io::{BufWriter, Write};

    /*
    Explanation of each type/attribute:
    - x: training data features
    - y: training data target
    - beta: array of the estimated coefficients
    - variance: variance term for normal distribution of error terms
    - beta_prior_mean: prior of the beta values
    - beta_prior_cov: prior of the covariance matrix of the beta values
    - a: shape for the inverse gamma (for variance)
    - b: scale for the inverse gamma (for variance)
    - beta_samples: vector of arrays containing the samples from Gibbs sampling
    - variance_samples: vector of floats containing variance samples from Gibbs sampling
    - num_samples: total number of samples the model should generate
    - burn_in: MCMC burn in period before sampling from posterior (CHANGED TO GIBBS)
    - thin: value that ensures that the previous value from MCMC is not too dependent (CHANGED TO GIBBS)
     */
    pub struct BayesianLinearRegressor {
        // Model Data
        x: Array2<f64>,
        y: Array1<f64>, 

        // Model Parameters
        beta: Array1<f64>,
        variance: f64,

        // Priors and Hyperparamters
        beta_prior_mean: Array1<f64>,
        beta_prior_cov: Array2<f64>,
        a: f64,
        b: f64,

        // MCMC Settings
        pub beta_samples: Vec<Array1<f64>>,
        pub variance_samples: Vec<f64>,
        num_samples: usize,
        burn_in: usize,
        thin: usize,
    }

    impl BayesianLinearRegressor {
        /*
        What it does: Initializes a new instance of the model with required parameters 
        and optional parameters for users to input with defaults if not inputted of the data

        Inputs: Data and optionally, parameters for prior, likelihood, and MCMC settings
        Outputs: Returns an instance of the model

        High-level logic: Unwrapping with defaults and adding intercept term
         */
        pub fn new(
            x: Array2<f64>,
            y: Array1<f64>,
            // Optional parameters with default values
            beta_prior_mean: Option<Array1<f64>>,
            beta_prior_cov: Option<Array2<f64>>,
            a: Option<f64>,
            b: Option<f64>,
            num_samples: Option<usize>,
            burn_in: Option<usize>,
            thin: Option<usize>,
        ) -> Self {
            assert_eq!(x.nrows(), y.len(), "Number of data points need to be the same");

            let intercept = Array::ones((x.nrows(), 1)); // Shape: [1436, 1]
            let intercept_x = concatenate![Axis(1), intercept, x]; // Shape: [1436, 7]

            let n_features = intercept_x.ncols(); // Now this should work
        
            // Set default prior mean to zero vector if None
            let beta_prior_mean = beta_prior_mean.unwrap_or_else(|| Array1::zeros(n_features));
        
            // Set default prior covariance to lambda * I if None
            // This is a weakly informative prior so it is a good default in case
            // Users do not have much information about the data
            let default_lambda = 10.0;
            let beta_prior_cov = beta_prior_cov.unwrap_or_else(|| Array2::eye(n_features) * default_lambda);
        
            // Set default values for hyperparameters if None
            let a = a.unwrap_or(1.0);
            let b = b.unwrap_or(1.0);
        
            // Set default values for MCMC settings if None
            let beta_samples: Vec<Array1<f64>> = Vec::new();
            let variance_samples: Vec<f64> = Vec::new();
            let num_samples = num_samples.unwrap_or(10000);
            let burn_in = burn_in.unwrap_or(200);
            let thin = thin.unwrap_or(10);
        
            // Initialize model parameters
            let beta = Array1::zeros(n_features);
            let variance = 1.0;

            BayesianLinearRegressor { x: intercept_x, y: y, beta: beta, variance: variance, beta_prior_mean: beta_prior_mean, beta_prior_cov: beta_prior_cov, a: a, b: b, beta_samples, variance_samples, num_samples: num_samples, burn_in: burn_in, thin: thin }
        }

        /*
        What it does: Generates samples of variance from the inverse gamma distribution

        Inputs: Parameters of the distribution, a and b since they can differ
        Outputs: A sample from the inverse gamma distribution with the inputted params

        High-level logic: 
        - Using RNG and Gammma struct from rand_distr crate to gennerate 
          samples.
        - Inverse gamma is taken by doing 1 / gamma sample
         */
        pub fn sample_variance(&mut self, a: f64, b: f64) -> f64 {
            let gamma = Gamma::new(a, 1.0 / b).expect("Invalid Gamma parameters");
            let mut rng = thread_rng();
            let gamma_sample = gamma.sample(&mut rng);
            1.0 / gamma_sample  
        }

        // Sample the beta values from a N(0, 1)
        /*
        What it does: Generates samples of beta and variance to initialize the MCMC step

        Inputs: Reference to self
        Outputs: None as it just updates the model attributes

        High-level logic:
        - Sample_beta function:
            - Takes in number of features, a sampled variance, prior for the beta
              means and covariance. 
            - Populates an array of beta values by sampling from Normal distribution
            - Uses Cholesky decomposition A = (L)(L_transpose) at the end to update
              the values in the Array
        - Updates beta and variance by calling the sample_beta and sample_variance methods
          and passes in the model attributes to do so
         */
        pub fn sample_prior(&mut self) {
            fn sample_beta(n_features: usize, sampled_variance: f64, beta_prior_mean: Array1<f64>, beta_prior_cov: Array2<f64>) -> Array1<f64> {
                // Initialize beta of 0s
                let mut z = Array1::<f64>::zeros(n_features);

                // Initialize the normal distribution
                let normal = Normal::new(0.0, 1.0).expect("Invalid Normal parameters");
                let mut rng = thread_rng();

                // Sample each beta from N(0, 1)
                for i in 0..z.len() {
                    let normal_sample = normal.sample(&mut rng);
                    z[i] = normal_sample
                }

                // Calculates the lower triangular matrix from the Cholesky decomposition
                let l = beta_prior_cov.cholesky(UPLO::Lower).unwrap();

                beta_prior_mean + (sampled_variance.sqrt() * l.dot(&z))
            }

            let sampled_variance = self.sample_variance(self.a, self.b);
            let n_features = self.x.ncols();
            self.beta = sample_beta(n_features, sampled_variance, self.beta_prior_mean.clone(), self.beta_prior_cov.clone());
            self.variance = sampled_variance;
        }

        /*
        What it does: Samples beta values from the posterior distribution

        Inputs: Reference to self
        Outputs: 1D Array where each element is a Beta value from a sample

        High-level logic:
        - Sample a value from the N(0,1) and then transform it into the
          a sample from the multivariate normal distribution N(Mu_n, Sigma_n)
        - Equations for computing the mean vector and covariance matrix are in
          the write up
        - Uses rand_distr::Normal to generate random samples from N(0,1)
        - Transformation to Multivariate Normal sample is 
          B_sample = mu_n + L * z where L is from the Cholesky decomposition
          and z is the N(0,1) samples of beta 
         */
        pub fn sample_beta_posterior(&mut self) -> Array1<f64> {
            // Get the parameters for the multivariate normal posterior
            let xt_x = self.x.t().dot(&self.x);
            let xt_y = self.x.t().dot(&self.y);

            // Equation for sigma_n is ((X_transpose * X) / variance + sigma_not)^-1
            let sigma_0 = self.beta_prior_cov.inv().expect("Failed to invert matrix");
            let sigma_n = (xt_x.mapv(|v| v / self.variance) + &sigma_0).inv().expect("Failed to invert matrix");

            // Equation for mu_n is Sigma_n * ((X_transpose * y) / variance) + (Sigma_not * mu_not))
            let mu_n = sigma_n.dot(&(xt_y.mapv(|v| v / self.variance) + sigma_0.dot(&self.beta_prior_mean)));
            
            let l = sigma_n.cholesky(UPLO::Lower).unwrap();
            let z = {
                let mut z = Array1::<f64>::zeros(self.x.ncols());

                // Initialize the normal distribution
                let normal = Normal::new(0.0, 1.0).expect("Invalid Normal parameters");
                let mut rng = thread_rng();

                // Sample each beta from N(0, 1)
                for i in 0..z.len() {
                    let normal_sample = normal.sample(&mut rng);
                    z[i] = normal_sample
                }
                z
            };

            mu_n + l.dot(&z)
        }

        /*
        What it does: Generates a sample of variance from its posterior distribution

        Inputs: Reference to self
        Outputs: Sample from the posterior distribution

        High-level logic:
        - Calculates residuals from y - y_hat
        - Calculates SSR by doing ||residuals||^2
        - Uses conjugate priors so the Inverse Gamma posterior has the following updated parameters
            - a = a + (n / 2)
            - b = b + (SSR / 2)
        - Passes in those updated param values to the sample_variance to generate a new sample
          from the posterior distribution
         */
        pub fn sample_variance_posterior(&mut self) -> f64 {
            let residuals = self.y.clone() - self.x.dot(&self.beta);
            let ssr = residuals.dot(&residuals);

            let new_a = (self.a) + (self.x.nrows() as f64 / 2.0);
            let new_b = self.b + (0.5 * ssr);

            self.sample_variance(new_a, new_b)

        }

        /*
        What it does: Updates the values of beta and variance for each step in MCMC

        Inputs: Reference to self
        Outputs: None as we just update the model attributes

        High-level logic:
        - Updates beta and variance by calling the respective methods of sampling
          from the posterior distribution
         */
        pub fn mcmc_step(&mut self) {
            self.beta = self.sample_beta_posterior();
            self.variance = self.sample_variance_posterior();
        }

        /*
        What it does: Uses Gibbs Sampling + MCMC aspects to populate the beta and 
        variance sample vectors

        Inputs: Reference to self
        Outputs: None as it just updates the model attributes

        High-level logic:
        - Calculates iterations by doing num_samples * thin and then + the burn_in period
        - For loop to iterate through the number of iterations and calls the mcmc_step method
          to generate the samples. If the iteration is past the burn in and is factor of
          the thin then populate the sample
        - Update the current beta and variance at the end of the iterations to be the 
          MMSE estimator of the samples; the average
        - Beta update
            - Iterates through the beta_samples to get the total betas and then apply
              closure of dividing by number of samples to get the mean
        - Variance update
            - Iterates through the variance samples, gets the sum, and divides by the
              number of samples to get the mean
         */
        pub fn run_mcmc(&mut self) {
            self.sample_prior();

            let mut beta_samples: Vec<Array1<f64>> = Vec::new();
            let mut variance_samples: Vec<f64> = Vec::new();

            let iterations = self.num_samples * self.thin + self.burn_in;

            for i in 0..iterations {
                self.mcmc_step();

                if i >= self.burn_in {
                    if i % self.thin == 0 {
                        beta_samples.push(self.beta.clone());
                        variance_samples.push(self.variance.clone());
                    }
                }
            }
            self.beta_samples = beta_samples;
            self.variance_samples = variance_samples;

            self.beta = {
                let mut mean_beta = Array1::<f64>::zeros(self.beta_samples[0].len());
                for beta in &self.beta_samples {
                    mean_beta = &mean_beta + beta;
                }
                mean_beta = mean_beta.mapv(|v| v / self.beta_samples.len() as f64);
                mean_beta
            };

            self.variance = {
                let mut mean_variance:f64 = 0.0;
                for var in &self.variance_samples {
                    mean_variance += var;
                }
                mean_variance = mean_variance / self.variance_samples.len() as f64;
                mean_variance
            };
        }

        /*
        -- Asked ChatGPT how to get the covariance matrix from the sampled beta from my MCMC sample --

        What it does: Calculates the covariance matrix from the beta samples

        Inputs: Reference to self
        Outputs: 2D covariance matrix

        High-level logic: 
        - Variables
            - n: number of samples from Gibbs sampling
            - d: number of regression coefficients
            - mean: MMSE estimator of the beta coefficients
        - Initializes an d x d covariance matrix of 0s
        - Converts the delta row vector into column vector with insert_axis
        - Iterates through each sample from the beta samples vector, calculates the 
          difference from the mean and squares it by taking the dot product
        - Adds the dot product to the covariance matrix and then divide by n - 1 to get
          the sample variance.
         */
        pub fn empirical_beta_covariance(&self) -> Array2<f64> {
            let n = self.beta_samples.len();
            let d = self.beta_samples[0].len();
            let mean = &self.beta;
            let mut cov = Array2::<f64>::zeros((d, d));
        
            for beta in &self.beta_samples {
                let delta = beta - mean;
                // Convert the delta vector into column vector with insert_axis and view
                let delta_2d = delta.view().insert_axis(Axis(1));
                cov = cov + &delta_2d.dot(&delta_2d.t());
            }
        
            cov.mapv(|v| v / (n as f64 - 1.0))
        }

        /*
        What it does: Makes a prediction on a new data point and provides the 95% 
        credible interval

        Inputs: Reference to self, new data point
        Outputs: Tuple of (mean, lower_bound, upper_bound)

        High-level logic:
        - Adds intercept term of 1 to the data point
        - Calculates mean prediction by computing dot product of Beta and the data point
        - Calculates the prediction variance (Equation in the comments below) and since the
          linear regression model assumes the data to be normally distributed, gets the upper
          and lower bounds with 1.96 (z-score) of the prediction variance
         */
        pub fn predict_single_with_credible_interval(&self, x_test: Array1<f64>) -> (f64, f64, f64) {
            // Predict with the MMMSE estimator of the posterior for beta
            let x_test_intercept = concatenate![Axis(0), array![1.0], x_test];
            let mean_prediction = x_test_intercept.dot(&self.beta);
    
            // Compute the variance for the prediction: sigma^2 + x_test * Sigma_beta * x_test
            let sigma_squared = self.variance;
            let sigma_beta = self.empirical_beta_covariance();
            let prediction_variance = sigma_squared + x_test_intercept.dot(&(sigma_beta.dot(&x_test_intercept)));
    
            // Calculate the 95% credible interval: mean ± 1.96 * sqrt(variance)
            let lower_bound = mean_prediction - 1.96 * prediction_variance.sqrt();
            let upper_bound = mean_prediction + 1.96 * prediction_variance.sqrt();
    
            (mean_prediction, lower_bound, upper_bound)
        }
    
        /*
        What it does: Makes prediction on multiple data points and returns the predictions

        Input: Reference to self, test data
        Outputs: Array of the predictions

        High-level logic:
        - Adds an column of 1s for the intercept term with concatenate
        - Computes the prediction by doing the dot product of the Beta coefficients and the test data
         */
        pub fn predict_multiple_with_mean_beta(&self, x_test: Array2<f64>) -> Array1<f64> {
            // Predict using the mean of the beta estimates (mean of posterior)
            let intercept = Array::ones((x_test.nrows(), 1));
            let x_test_intercept = concatenate![Axis(1), intercept, x_test];
            x_test_intercept.dot(&self.beta)
        }

        /*
        What it does: Returns a vector of tuples where each tuple is the 95% credible interval
        for the beta coefficients

        Inputs: Reference to self
        Outputs: Vector of (f64, f64) tuple

        High-level logic:
        - Initializes empty vector
        - Uses for loop to update each tuple for the predictor
            - Gets a vector of the ith column of the beta samples with into_iter() and then map with closure
              of the ith column and then collects it into vector
            - Uses sort_by since f64 does not support normal comparison
            - Gets the lower_index and upper index by getting the 2.5 and 97.5 percentiles of the data
              and then rounds them to nearest 3 decimal poitns and pushes them to the tuple and to the vector
         */
        pub fn get_beta_credible_interval(&self) -> Vec<(f64, f64)> {
            let mut credible_intervals: Vec<(f64, f64)> = Vec::new();

            for i in 0..self.beta.len() {
                let mut beta_i: Vec<f64> = self.beta_samples.clone().into_iter().map(|beta| beta[i]).collect();
                // Sorts by making a less than b with partial_cmp
                beta_i.sort_by(|a, b| a.partial_cmp(b).unwrap());

                // Floor to get integer of the index
                let lower_index = (0.025 * beta_i.len() as f64).floor() as usize;
                let upper_index = (0.975 * beta_i.len() as f64).floor() as usize;

                // Round to nearest 3 decimal points by doing (value * 1000.0).round() / 1000.0
                credible_intervals.push(((beta_i[lower_index] * 1000.0).round() / 1000.0, (beta_i[upper_index] * 1000.0).round() / 1000.0));
            }
            credible_intervals
        }

        /*
        -- Asked ChatGPT how to export the beta samples as a CSV file --
        What it does: Exports the beta samples as a CSV file to use for post-processing or analysis

        Inputs: Reference to self, file path
        Outputs: None but it creates the CSV file in the project root folder

        High-level logic:
        - Creates a file and wraps it in the BufWriter
        - Iterates through each row in the beta_samples, converts them to strings with closure
          and then collects them into a vector of strings and joins them with "," for CSV notation
         */
        pub fn export_beta_samples_to_csv(&self, filepath: &str) { 
            let file = File::create(filepath).expect("Unable to create file");
            let mut writer = BufWriter::new(file);
    
            for sample in &self.beta_samples {
                let line = sample.iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<String>>()
                    .join(",");
                writeln!(writer, "{}", line).expect("Failed to write line");
            }
        }

        /*
        What it does: Calculates the R^2 for the data

        Inputs: Reference to self
        Outputs: f64 R^2 value

        High-level logic:
        - Computes the predicted values for each row in the training data by doing the
          dot product with the beta coefficients and the training data
        - Computes residual: residual = y - y_pred
        - Computes Sum of Squared residuals: residuals * residuals
        - Computes the total sum of squares: 
            - Calculates the mean of the y variable and then iterates through each y
              and applies a closure of subtracting the mean from it and then doing squaring
              it and finally summing it and casts to f64 to get the variance
        - Computes the R^2: 1 - (SSR / TSS)
         */
        pub fn r_squared(&self) -> f64 {

            // Compute predicted values
            let y_pred = self.x.dot(&self.beta);
            let residuals = &self.y - &y_pred;
            let ssr = residuals.dot(&residuals);
    
            // Compute total sum of squares (SST)
            let y_mean = self.y.mean().unwrap();
            // Iterate through each y and for each y value, calculate the difference from mean and then square it
            // And cast to f64 for variance. This is normal equation for calculating the variance
            let total_sum_of_squares = self.y.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>();
    
            // Compute R^2
            1.0 - (ssr / total_sum_of_squares)
        }
    }
}