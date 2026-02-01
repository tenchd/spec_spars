#![allow(unused)]
use clap::Parser;
use spec_spars::lap_test;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {

    /// allows user to specify location of input file
    #[arg(short, long, default_value_t = ("/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx".to_string()))]
    input_file: String,

    // allows user to specify name of dataset
    #[arg(short, long, default_value_t = ("virus".to_string()))]
    dataset_name: String,

    // allows the user to set epsilon to control approximation quality.
    #[arg(short, long, default_value_t = 0.5)]
    epsilon: f64,

    // default: do not print out debug information about sparsification run.
    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    // allows the user to specify a seed for reproducibility.
    #[arg(short, long, default_value_t = 0)]
    seed: u64,

    // default: benchmark the run. if flag is set, skips benchmarking.
    #[arg(short, long, default_value_t = false)]
    benchmark_skip: bool,

    // if set, ignores input_file and dataset_name parameters and sparsifies a set list of datasets according to other command line arguments.
    #[arg(short, long, default_value_t = false)]
    process_all: bool,
}

// allows for bulk processing of a list of datasets. feel free to customize for bulk jobs.
fn process_standard_datasets(args: Args) {
    let input_files: Vec<&str> = vec!["/global/cfs/cdirs/m1982/david/bulk_to_process/virus/virus.mtx", 
                                        "/global/cfs/cdirs/m1982/david/bulk_to_process/mouse_gene/mouse_gene.mtx", 
                                        "/global/cfs/cdirs/m1982/david/bulk_to_process/human_gene1/human_gene1.mtx", 
                                        "/global/cfs/cdirs/m1982/david/bulk_to_process/human_gene2/human_gene2.mtx", 
                                        ];
    
    let dataset_names: Vec<&str> = vec!["virus",
                                        "mouse",
                                        "human1",
                                        "human2",
                                        ];

    for i in 0..input_files.len() {
        let current_input_file = input_files[i];
        let current_dataset_name = dataset_names[i];
        println!("============================================================");
        println!("            Processing {} dataset", current_dataset_name);
        println!("============================================================");
        lap_test(current_input_file, current_dataset_name, args.epsilon, args.verbose, args.seed, !args.benchmark_skip);
        println!("============================================================");
        println!("             Finished {} dataset", current_dataset_name);
        println!("============================================================");
        println!("");
    }

}

fn main() {
    let args = Args::parse();

    println!("Arguments are input file = {}, dataset name = {}, epsilon = {}, verbose = {}, seed = {}, benchmark skip = {}, process all = {}",
        args.input_file, args.dataset_name, args.epsilon, args.verbose, args.seed, args.benchmark_skip, args.process_all);
    if args.process_all {
        process_standard_datasets(args);
    }
    else {
        lap_test(&args.input_file, &args.dataset_name, args.epsilon, args.verbose, args.seed, !args.benchmark_skip);
    }
    
}

