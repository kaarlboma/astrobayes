pub mod fileio {
    use csv::ReaderBuilder;
    use ndarray::{Array2, Array1};
    use std::error::Error;
    
    pub fn read_csv(filename: &str) -> Result<(Array2<f64>, Array1<f64>), Box<dyn Error>> {
        let mut rdr = ReaderBuilder::new().has_headers(true).from_path(filename)?;
    
        let mut features: Vec<Vec<f64>> = Vec::new();
        let mut targets: Vec<f64> = Vec::new();
    
        for result in rdr.records() {
            let record = result?;
            
            let target: f64 = record.get(0).unwrap().parse()?;  // `pl_rade` is the first column
            let feature: Vec<f64> = record.iter().skip(1) // After the first column
                                          .map(|s| s.parse().unwrap())
                                          .collect();
            
            features.push(feature);
            targets.push(target);
        }
    
        // Convert to ndarray types
        let features_array = Array2::from_shape_vec((features.len(), features[0].len()), features.concat())?;
        let targets_array = Array1::from_vec(targets);
    
        Ok((features_array, targets_array))
    }
}