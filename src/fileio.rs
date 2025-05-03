pub mod fileio {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader};

    pub fn load_index(path: &str) -> HashMap<u32, String> {
        let file = File::open(path).expect("Could not open PLD index file");
        let reader = BufReader::new(file);
        let mut map = HashMap::new();

        for line in reader.lines() {
            if let Ok(entry) = line {
                let parts: Vec<&str> = entry.trim().split('\t').collect();
                if parts.len() == 2 {
                    if let Ok(id) = parts[1].parse::<u32>() {
                        map.insert(id, parts[0].to_string());
                    }
                }
            }
        }

        map
    }
}