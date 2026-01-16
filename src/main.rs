use clap::Parser;
use spec_spars::lap_test;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Location of input file
    #[arg(short, long, default_value_t = ("/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx".to_string()))]
    input_file: String,

    /// name of dataset
    #[arg(short, long, default_value_t = ("virus".to_string()))]
    dataset_name: String,

    #[arg(short, long, default_value_t = 0.5)]
    epsilon: f64,

    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    #[arg(short, long, default_value_t = 0)]
    seed: u64,

    #[arg(short, long, default_value_t = true)]
    benchmark: bool,
}

fn main() {
    let args = Args::parse();

    println!("Arguments are input file = {}, dataset name = {}, epsilon = {}, verbose = {}, seed = {}, benchmark = {}",
        args.input_file, args.dataset_name, args.epsilon, args.verbose, args.seed, args.benchmark);
    
    lap_test(&args.input_file, &args.dataset_name, args.epsilon, args.verbose, args.seed, args.benchmark);
}

//     //let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
//     //let dataset_name = "virus";

//     let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/mouse_gene/mouse_gene.mtx";
//     let dataset_name = "mouse_gene";

//     let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/human_gene1/human_gene1.mtx";
//     let dataset_name = "human_gene1";

//     let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/human_gene2/human_gene2.mtx";
//     let dataset_name = "human_gene2";
