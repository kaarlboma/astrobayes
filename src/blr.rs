pub mod blr {
    use ndarray::{concatenate, Array, Array1, Array2, Axis, array};
    use rand_distr::{Gamma, Normal, Distribution};
    use rand::thread_rng;
    use ndarray_linalg::cholesky::Cholesky;
    use ndarray_linalg::{UPLO, Inverse};
    use std::fs::File;
    use std::io::{BufWriter, Write};

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

        // Samples variance from the inverse gamma distribution since we assume
        // The variance terms are inverse gamma distributed with parameters a, b
        pub fn sample_variance(&mut self, a: f64, b: f64) -> f64 {
            let gamma = Gamma::new(a, 1.0 / b).expect("Invalid Gamma parameters");
            let mut rng = thread_rng();
            let gamma_sample = gamma.sample(&mut rng);
            1.0 / gamma_sample  
        }

        // Sample the beta values from a N(0, 1)
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

                let l = beta_prior_cov.cholesky(UPLO::Lower).unwrap();

                beta_prior_mean + (sampled_variance.sqrt() * l.dot(&z))
            }

            let sampled_variance = self.sample_variance(self.a, self.b);
            let n_features = self.x.ncols();
            self.beta = sample_beta(n_features, sampled_variance, self.beta_prior_mean.clone(), self.beta_prior_cov.clone());
            self.variance = sampled_variance;
        }

        // Sample beta values from the multivariate posterior distribution
        pub fn sample_beta_posterior(&mut self) -> Array1<f64> {
            // Get the parameters for the multivariate normal posterior
            let xt_x = self.x.t().dot(&self.x);
            let xt_y = self.x.t().dot(&self.y);

            let sigma_0 = self.beta_prior_cov.inv().expect("Failed to invert matrix");
            let sigma_n = (xt_x.mapv(|v| v / self.variance) + &sigma_0).inv().expect("Failed to invert matrix");

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

        // Sample variance values from the inverse gamma posterior with updated parameters
        pub fn sample_variance_posterior(&mut self) -> f64 {
            let residuals = self.y.clone() - self.x.dot(&self.beta);
            let ssr = residuals.dot(&residuals);

            let new_a = (self.a) + (self.x.nrows() as f64 / 2.0);
            let new_b = self.b + (0.5 * ssr);

            self.sample_variance(new_a, new_b)

        }

        // Each step of MCMC where we update the value of beta and variance at each step
        pub fn mcmc_step(&mut self) {
            self.beta = self.sample_beta_posterior();
            self.variance = self.sample_variance_posterior();
        }

        // Run MCMC and add the samples from after burn-in
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

        // Asked ChatGPT how to get the covariance matrix from the sampled beta from MCMC
        pub fn empirical_beta_covariance(&self) -> Array2<f64> {
            let n = self.beta_samples.len();
            let d = self.beta_samples[0].len();
            let mean = &self.beta;
            let mut cov = Array2::<f64>::zeros((d, d));
        
            for beta in &self.beta_samples {
                let delta = beta - mean;
                let delta_2d = delta.view().insert_axis(Axis(1));
                cov = cov + &delta_2d.dot(&delta_2d.t());
            }
        
            cov.mapv(|v| v / (n as f64 - 1.0))
        }

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
    
        // Predict for multiple data points using the mean of the beta estimates
        pub fn predict_multiple_with_mean_beta(&self, x_test: Array2<f64>) -> Array1<f64> {
            // Predict using the mean of the beta estimates (mean of posterior)
            let intercept = Array::ones((x_test.nrows(), 1));
            let x_test_intercept = concatenate![Axis(1), intercept, x_test];
            x_test_intercept.dot(&self.beta)
        }

        pub fn get_beta_credible_interval(&self) -> Vec<(f64, f64)> {
            let mut credible_intervals: Vec<(f64, f64)> = Vec::new();

            for i in 0..self.beta.len() {
                let mut beta_i: Vec<f64> = self.beta_samples.clone().into_iter().map(|beta| beta[i]).collect();
                beta_i.sort_by(|a, b| a.partial_cmp(b).unwrap());

                let lower_index = (0.025 * beta_i.len() as f64).floor() as usize;
                let upper_index = (0.975 * beta_i.len() as f64).floor() as usize;

                credible_intervals.push(((beta_i[lower_index] * 1000.0).round() / 1000.0, (beta_i[upper_index] * 1000.0).round() / 1000.0));
            }
            credible_intervals
        }

        pub fn export_beta_samples_to_csv(&self, filepath: &str) { // Asked ChatGPT how to export csv file of the array
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

        pub fn r_squared(&self) -> f64 {
            // Equation for R^2 is 1 - (SSR / TSS)

            // Compute predicted values
            let y_pred = self.x.dot(&self.beta);
            let residuals = &self.y - &y_pred;
            let ssr = residuals.dot(&residuals);
    
            // Compute total sum of squares (SST)
            let y_mean = self.y.mean().unwrap();
            let total_sum_of_squares = self.y.iter().map(|y| (y - y_mean).powi(2)).sum::<f64>();
    
            // Compute R^2
            1.0 - (ssr / total_sum_of_squares)
        }
    }
}