use fileio::fileio::load_index;

mod fileio;

fn main() {
    let index_pld = fileio::fileio::load_index("data/index.pld");
    let index_host = load_index("data/index.host");
    println!("{:?}", index_host.values());
}
