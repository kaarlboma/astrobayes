use fileio::fileio::read_csv;
mod fileio;

fn main() {
    let (features, targets) = read_csv("data/cleaned_exoplanet_data.csv").unwrap(); // Use `?` to propagate errors

    // Print the features and targets for testing
    println!("Features:\n{}", features);
    println!("Targets:\n{}", targets);
}