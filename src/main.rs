#![allow(unused)]
use clap::Parser;
use spec_spars::{sparsify_dataset, run_basic_experiment, run_jl_scaling_experiment, run_jl_dim_experiment, run_space_use_experiment};

const INPUT_FILENAME_VIRUS: &str = "data/virus.mtx";
const INPUT_FILENAME_MOUSE: &str = "data/mouse.mtx";
const INPUT_FILENAME_HUMAN1: &str = "data/human1.mtx";
const INPUT_FILENAME_HUMAN2: &str = "data/human2.mtx";

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {

    // allows user to specify location of input file
    #[arg(short, long, default_value_t = (INPUT_FILENAME_VIRUS.to_string()))]
    input_file: String,

    // allows user to specify location of output file
    #[arg(short, long, default_value_t = ("".to_string()))]
    output_file: String,

    // allows user to specify name of dataset
    #[arg(short, long, default_value_t = ("virus".to_string()))]
    dataset_name: String,

    // allows the user to set epsilon to control approximation quality.
    #[arg(short, long, default_value_t = 0.5)]
    epsilon: f64,

    // default: do not print out debug information about sparsification run.
    #[arg(short, long, default_value_t = false)]
    verbose: bool,

    // allows the user to specify a seed for jl sketching, for reproducibility.
    #[arg(short, long, default_value_t = 0)]
    sketch_seed: u64,    
    
    // allows the user to specify a seed for sampling, for reproducibility.
    #[arg(short, long, default_value_t = 0)]
    sampling_seed: u64,

    // default: benchmark the run. if flag is set, skips benchmarking.
    #[arg(short, long, default_value_t = false)]
    benchmark_skip: bool,

    // if set, ignores input_file and dataset_name parameters and sparsifies a set list of datasets according to other command line arguments.
    #[arg(short, long, default_value_t = false)]
    process_all: bool,

    // if set, ignores everything else and runs currently staged experiment.
    #[arg(short, long, default_value_t = false)]
    run_experiment: bool,

    // selects experiment.
    #[arg(short, long, default_value_t = 0)]
    choose_experiment: usize,
}

// allows for bulk processing of a list of datasets. feel free to customize for bulk jobs.
fn process_standard_datasets(args: Args) {
    let input_files: Vec<&str> = vec![INPUT_FILENAME_VIRUS,
                                        INPUT_FILENAME_MOUSE,
                                        INPUT_FILENAME_HUMAN1,
                                        INPUT_FILENAME_HUMAN2,
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
        sparsify_dataset(current_input_file, current_dataset_name, "", args.epsilon, args.verbose, args.sketch_seed, args.sampling_seed, !args.benchmark_skip);
        println!("============================================================");
        println!("             Finished {} dataset", current_dataset_name);
        println!("============================================================");
        println!("");
    }

}

fn run_basic() {
    println!("running basic experiment on specified input file. ignoring parameter input flags.");
    run_basic_experiment();
}

fn run_jl_scaling() {
    println!("running JL scaling experiment on specified input file. ignoring parameter input flags.");
    run_jl_scaling_experiment();
}

fn run_jl_dim() {
    println!("running JL dimension sensitivity experiment on specified input file. ignoring parameter input flags.");
    run_jl_dim_experiment();
}

fn main() {
    let args = Args::parse();

    println!("Command line arguments are: 
        input file = {}, 
        output file = {},
        dataset name = {}, 
        epsilon = {}, 
        verbose = {}, 
        sketch seed = {}, 
        sampling seed = {}, 
        benchmark skip = {}, 
        process all = {}, 
        run_experiment = {},
        experiment selector = {}",

        args.input_file, args.output_file, args.dataset_name, args.epsilon, args.verbose, args.sketch_seed, args.sampling_seed, 
        args.benchmark_skip, args.process_all, args.run_experiment, args.choose_experiment);

    if args.run_experiment {
        match args.choose_experiment {
            0 => run_basic(),
            1 => run_jl_scaling(),
            2 => run_jl_dim(),
            3 => run_space_use_experiment(),
            _ => println!("invalid experiment selector. currently only supports 0 (basic), 1 (jl scaling), and 2 (jl dimension)."),
        }

    }
    else {    
        if args.process_all {
            println!("running demo sparsifications. ignoring all other input flags.");
            process_standard_datasets(args);
        }
        else {
            sparsify_dataset(&args.input_file, &args.dataset_name, &args.output_file, args.epsilon, args.verbose, args.sketch_seed, args.sampling_seed, !args.benchmark_skip);
        }
    }
}



