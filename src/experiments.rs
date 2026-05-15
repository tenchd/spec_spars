use crate::stream::InputStream;
use crate::sparsifier::{Sparsifier,SparsifierParameters};
use crate::utils::{CustomIndex, CustomIndex::from_int, BenchmarkPoint};
use crate::tests::make_random_vector;
use ndarray::Array1;
use petgraph::algo::connected_components;
use petgraph::Graph;
use std::fs::File;
use std::fs::OpenOptions;
use std::path::Path;
use csv::Writer;


// handles writing experiment results as csv file
struct ExperimentResult {
    output_filename: String,
    column_headers: Vec<String>,
    writer: csv::Writer<std::fs::File>,
}

impl ExperimentResult {
    // constructor that opens a file and writes csv column headers (if file doesn't already exist)
    fn new(output_filename: String, column_headers: Vec<String>) -> Self {
        //let file = File::create(&output_filename).expect("Could not create file");
        let path: &Path = output_filename.as_ref();
        let file_exists = path.exists();
        let file = OpenOptions::new()
            .write(true)
            .create(true)
            .append(true)
            .open(&output_filename)
            .expect("Could not open or create file");
        let mut writer = Writer::from_writer(file);
        if !file_exists {
            writer.write_record(&column_headers).expect("Could not write column headers");
            writer.flush().expect("Could not flush writer");
        }
        ExperimentResult {
            output_filename,
            column_headers,
            writer,
        }
    }

    // call this function to write an experiment result as a line in the csv.
    fn record_result(&mut self, result: Vec<String>) {
        self.writer.write_record(&result).expect("Could not write record");
        self.writer.flush().expect("Could not flush writer");
    }

    // call this function when you're done to ensure everything is written to the file.
    fn finalize(&mut self) {
        self.writer.flush().expect("Could not flush writer");
    }
}

// returns whether the sparsified graph has the same number of CCs as the original graph (stored in the InputStream object)
fn verify_ccs<IndexType: CustomIndex>(stream: &InputStream, sparsifier: &Sparsifier<IndexType>) -> bool {
    let original_graph = stream.get_input_graph();
    let sparsified_graph = sparsifier.to_petgraph();
    let original_ccs = connected_components(&original_graph);
    let sparsified_ccs = connected_components(&sparsified_graph);
    return original_ccs == sparsified_ccs;
}

// stores the result of testing the quadratic normal form of the sparsifier on different randomly chosen vectors.
struct QuadraticFormProbeResult {
        upper_bound_violations: usize,
        lower_bound_violations: usize,
        mean_relative_error: f64,
        std_dev_relative_error: f64,
        max_relative_error: f64,
}

impl QuadraticFormProbeResult {
    fn new(upper_bound_violations: usize, lower_bound_violations: usize, mean_relative_error: f64, std_dev_relative_error: f64, max_relative_error: f64) -> Self {
        QuadraticFormProbeResult {
            upper_bound_violations: upper_bound_violations,
            lower_bound_violations: lower_bound_violations,
            mean_relative_error: mean_relative_error,
            std_dev_relative_error: std_dev_relative_error,
            max_relative_error: max_relative_error,
        }
    }
}

// randomly samples vectors to measure the difference between the quadratic normal form w.r.t. the sparsifer vs that of the original laplacian.
fn probe_quadratic_form(stream: &InputStream, sparsifier: &Sparsifier<i32>, dataset_name: &str, probe_iterations: usize) -> QuadraticFormProbeResult {
    let original_state = stream.produce_laplacian();
    let mut errors = Array1::zeros(probe_iterations);
    let mut upper_bound_violations = 0;
    let mut lower_bound_violations = 0;
    for i in 0..probe_iterations {
        let probe_vector = make_random_vector::<i32>(from_int(sparsifier.num_nodes));
        let original_product = original_state.compute_quadratic_form(&probe_vector);
        let sparsified_product = sparsifier.compute_quadratic_form(&probe_vector);
        let upper_bound = original_product * (1.0 + sparsifier.epsilon);
        let lower_bound = original_product * (1.0 - sparsifier.epsilon);
        let relative_error = (sparsified_product - original_product).abs() / original_product.abs();
        //println!("original quadratic form is {}, sparsified quadratic form is {}, upper bound is {}, lower bound is {}",
        //    original_product, sparsified_product, upper_bound, lower_bound);
        if sparsified_product >= upper_bound {
            upper_bound_violations += 1;
        }
        if sparsified_product <= lower_bound {
            lower_bound_violations += 1;
        }
        errors[i] = relative_error;
    }
    let mut sketch_type = "discrete";
    if sparsifier.sketch_uniform {
        sketch_type = "uniform";
    }
    let max_error = errors.iter().copied().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
    println!("for dataset {}, with epsilon {} and sketch type {},
        mean rel error ------ {}
        std dev rel error --- {}
        max rel error ------- {}",
        dataset_name, sparsifier.epsilon, sketch_type, errors.mean().unwrap(), errors.std(0.0), max_error);
    println!("upper bound violations: {}, lower bound violations: {}", upper_bound_violations, lower_bound_violations);
    return QuadraticFormProbeResult::new(upper_bound_violations, lower_bound_violations, errors.mean().unwrap(), errors.std(0.0), max_error);
}

// first experiment: for given datasets, vary epsilon and sketch type. track the follwing for each run:
// dataset (for now all mouse)
// epsilon
// sketch type (discrete, uniform)
// runtime (broken down by task)
// whether observe CC failure, 
// quadratic form statistics (# failures, mean/std/max of relative error)
// sparsification rate
pub fn basic_exploration(input_filenames: &[&str], dataset_names: &[&str], writeout: bool) {
    let epsilons = [0.75, 0.5, 0.25];
    let sketch_types = [true, false]; //true for uniform, false for discrete (fix this later)
    let output_filename = format!("experiment_results/basic_exploration/basic_experiment_results.csv");
    let dataset_stats_filename = format!("experiment_results/basic_exploration/dataset_stats.csv");
    let column_headers = vec![
        "dataset".to_string(),
        "epsilon".to_string(),
        "sketch_type".to_string(),
        "evim_time".to_string(),
        "jl_time".to_string(),
        "solve_time".to_string(),
        "diff_norm_time".to_string(),
        "reweight_time".to_string(),
        "cc_success".to_string(),
        "upper_bound_violations".to_string(),
        "lower_bound_violations".to_string(),
        "mean_rel_error".to_string(),
        "std_dev_rel_error".to_string(),
        "max_rel_error".to_string(),
        "sparsification_rate".to_string(),
    ];

    let mut experiment_result = ExperimentResult::new(output_filename, column_headers);

    let dataset_stats_column_headers = vec![
        "dataset".to_string(),
        "num_nodes".to_string(),
        "num_edges".to_string(),
        "notes".to_string(),
    ];

    let mut dataset_stats = ExperimentResult::new(dataset_stats_filename, dataset_stats_column_headers);
        
    for (input_filename, dataset_name) in input_filenames.iter().zip(dataset_names.iter()) {
        // track if it's the first experiment for this input file. if so, write its stats into the dataset sets csv file.
        let mut first = true;
        for epsilon in epsilons {
            for sketch_type in sketch_types {
                let mut sketch_name = "discrete";
                if sketch_type {sketch_name = "uniform";}
                println!("------------------------------------------------------------------------");
                println!("    Sparsifying {} with epsilon {} and {} sketch", dataset_name, epsilon, sketch_name);
                println!("------------------------------------------------------------------------");
                let mut parameters = SparsifierParameters::new_default(true);
                parameters.jl_factor = 4.0;
                parameters.sketch_seed = 0;
                parameters.sampling_seed = 0;
                parameters.epsilon = epsilon;
                parameters.sketch_uniform = sketch_type;

                let stream = InputStream::new(input_filename, dataset_name);
                let (sparsifier, sparsification_stats) = stream.run_stream(&parameters, false, false);

                let cc_success = verify_ccs(&stream, &sparsifier);
                let quadratic_form_result = probe_quadratic_form(&stream, &sparsifier, dataset_name, 100);
                println!("------------------------------------------------------------------------");
                println!("      Finished {} with epsilon {} and {} sketch", dataset_name, epsilon, sketch_name);
                println!("------------------------------------------------------------------------");
                experiment_result.record_result(vec![
                    dataset_name.to_string(),
                    epsilon.to_string(),
                    sketch_name.to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::EvimComplete, BenchmarkPoint::Initialize).to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::JlSketchComplete, BenchmarkPoint::EvimComplete).to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::SolvesComplete, BenchmarkPoint::JlSketchComplete).to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::DiffNormsComplete, BenchmarkPoint::SolvesComplete).to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::ReweightingsComplete, BenchmarkPoint::DiffNormsComplete).to_string(),
                    cc_success.to_string(),
                    quadratic_form_result.upper_bound_violations.to_string(),
                    quadratic_form_result.lower_bound_violations.to_string(),
                    quadratic_form_result.mean_relative_error.to_string(),
                    quadratic_form_result.std_dev_relative_error.to_string(),
                    quadratic_form_result.max_relative_error.to_string(),
                    sparsification_stats.sparsification_rate().to_string(),
                ]);

                // if you haven't written the dataset information for this input file yet, do it now.
                if first {
                    dataset_stats.record_result(vec![
                        dataset_name.to_string(),
                        sparsifier.num_nodes.to_string(),
                        sparsification_stats.get_num_orig_edges().to_string(),
                        ".".to_string(),
                    ]);
                    first = false;
                }

                if writeout {
                    // write out sparsifier laplacian with informative filename
                    let output_filename = dataset_name;
                    let output_filepath = crate::OUTPUT_LAPLACIAN_PATH;
                    let mut sketch_type = "_discrete";
                    if sparsifier.sketch_uniform {sketch_type = "_uniform";}
                    let params = sparsifier.epsilon.to_string() + &sketch_type;
                    let output_location = output_filepath.to_owned() + &output_filename.to_owned() + "_rust_static_" + &params.to_owned() + ".mtx";
                    println!("Writing to {}", output_location);
                    crate::utils::write_mtx(&output_location, &sparsifier.current_laplacian);
                }
            }
        }
    }
    experiment_result.finalize();
    dataset_stats.finalize();
}

// this experiment tests the effect of the jl scaling factor on solution quality. NOTE: this is DIFFERENT than jl_factor, which controls the jl dimension.
pub fn jl_scaling_factor_sensitivity(input_filenames: &[&str], dataset_names: &[&str], writeout: bool) {
    let sketch_types = [true, false]; //true for uniform, false for discrete (fix this later)
    let output_filename = format!("experiment_results/jl_scaling/experiment_results.csv");
    let dataset_stats_filename = format!("experiment_results/jl_scaling/dataset_stats.csv");
    let column_headers = vec![
        "dataset".to_string(),
        "epsilon".to_string(),
        "sketch_type".to_string(),
        "jl_scaling_factor".to_string(),
        "evim_time".to_string(),
        "jl_time".to_string(),
        "solve_time".to_string(),
        "diff_norm_time".to_string(),
        "reweight_time".to_string(),
        "cc_success".to_string(),
        "upper_bound_violations".to_string(),
        "lower_bound_violations".to_string(),
        "mean_rel_error".to_string(),
        "std_dev_rel_error".to_string(),
        "max_rel_error".to_string(),
        "sparsification_rate".to_string(),
    ];

    let mut experiment_result = ExperimentResult::new(output_filename, column_headers);

    let dataset_stats_column_headers = vec![
        "dataset".to_string(),
        "num_nodes".to_string(),
        "num_edges".to_string(),
        "notes".to_string(),
    ];

    let mut dataset_stats = ExperimentResult::new(dataset_stats_filename, dataset_stats_column_headers);
        
    for (input_filename, dataset_name) in input_filenames.iter().zip(dataset_names.iter()) {
        // track if it's the first experiment for this input file. if so, write its stats into the dataset sets csv file.
        let mut first = true;
        for jl_scaling_option in [1,2,3,4] {
            for sketch_type in sketch_types {
                let mut sketch_name = "discrete";
                if sketch_type {sketch_name = "uniform";}
                let mut parameters = SparsifierParameters::new_default(true);
                parameters.jl_factor = 4.0;
                parameters.sketch_seed = 0;
                parameters.sampling_seed = 0;
                parameters.epsilon = 0.5;
                parameters.sketch_uniform = sketch_type;
 
                let mut jl_dim = -1;
                
                // scoped sparsifier build to get jl_dim without keeping dummy sparsifier around.
                // figure out a better way to handle this later.
                if true {
                    let temp_sparsifier = Sparsifier::<i32>::from_matrix(input_filename, &parameters);
                    jl_dim = temp_sparsifier.jl_dim;
                    println!("jl dim set to {}, sqrt is {}", jl_dim, (jl_dim as f64).sqrt());
                }

                match jl_scaling_option{
                    1=>parameters.jl_scaling_factor = (3.0_f64).sqrt(),
                    2=>parameters.jl_scaling_factor = 1.0/((3.0_f64).sqrt()),
                    3=>parameters.jl_scaling_factor = 1.0/((jl_dim as f64).sqrt()),
                    4=>parameters.jl_scaling_factor = (jl_dim as f64).sqrt(),
                    _=>assert!(false, "Bad option for jl scaling factor in crate::experiments::jl_scaling_factor_sensitivity"),
                }


                println!("------------------------------------------------------------------------");
                println!("    Sparsifying {} with jl scaling factor {:.2} and {} sketch", dataset_name, parameters.jl_scaling_factor, sketch_name);
                println!("------------------------------------------------------------------------");

                let stream = InputStream::new(input_filename, dataset_name);
                let (sparsifier, sparsification_stats) = stream.run_stream(&parameters, false, false);

                let cc_success = verify_ccs(&stream, &sparsifier);
                let quadratic_form_result = probe_quadratic_form(&stream, &sparsifier, dataset_name, 100);
                println!("------------------------------------------------------------------------");
                println!("      Finished {} with jl scaling factor {:.2} and {} sketch", dataset_name, parameters.jl_scaling_factor, sketch_name);
                println!("------------------------------------------------------------------------");
                experiment_result.record_result(vec![
                    dataset_name.to_string(),
                    parameters.epsilon.to_string(),
                    sketch_name.to_string(),
                    sparsifier.jl_scaling_factor.to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::EvimComplete, BenchmarkPoint::Initialize).to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::JlSketchComplete, BenchmarkPoint::EvimComplete).to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::SolvesComplete, BenchmarkPoint::JlSketchComplete).to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::DiffNormsComplete, BenchmarkPoint::SolvesComplete).to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::ReweightingsComplete, BenchmarkPoint::DiffNormsComplete).to_string(),
                    cc_success.to_string(),
                    quadratic_form_result.upper_bound_violations.to_string(),
                    quadratic_form_result.lower_bound_violations.to_string(),
                    quadratic_form_result.mean_relative_error.to_string(),
                    quadratic_form_result.std_dev_relative_error.to_string(),
                    quadratic_form_result.max_relative_error.to_string(),
                    sparsification_stats.sparsification_rate().to_string(),
                ]);

                // if you haven't written the dataset information for this input file yet, do it now.
                if first {
                    dataset_stats.record_result(vec![
                        dataset_name.to_string(),
                        sparsifier.num_nodes.to_string(),
                        sparsification_stats.get_num_orig_edges().to_string(),
                        ".".to_string(),
                    ]);
                    first = false;
                }

                if writeout {
                    // write out sparsifier laplacian with informative filename
                    let output_filename = dataset_name;
                    let output_filepath = crate::OUTPUT_LAPLACIAN_PATH;
                    let mut sketch_type = "_discrete";
                    if sparsifier.sketch_uniform {sketch_type = "_uniform";}
                    let params = sparsifier.jl_scaling_factor.to_string() + &sketch_type;
                    let output_location = output_filepath.to_owned() + &output_filename.to_owned() + "_rust_static_" + &params.to_owned() + ".mtx";
                    println!("Writing to {}", output_location);
                    crate::utils::write_mtx(&output_location, &sparsifier.current_laplacian);
                }

            }
        }
    }
    experiment_result.finalize();
    dataset_stats.finalize();
}

// this experiment tests the impact of the number of jl sketch matrix columns on solution quality and system performance.
pub fn jl_dim_sensitivity(input_filenames: &[&str], dataset_names: &[&str], writeout: bool) {
    let sketch_types = [true, false]; //true for uniform, false for discrete (fix this later)
    let output_filename = format!("experiment_results/jl_factor/experiment_results.csv");
    let dataset_stats_filename = format!("experiment_results/jl_factor/dataset_stats.csv");
    let column_headers = vec![
        "dataset".to_string(),
        "epsilon".to_string(),
        "sketch_type".to_string(),
        "jl_factor".to_string(),
        "jl_scaling_factor".to_string(),
        "jl_dim".to_string(),
        "evim_time".to_string(),
        "jl_time".to_string(),
        "solve_time".to_string(),
        "diff_norm_time".to_string(),
        "reweight_time".to_string(),
        "cc_success".to_string(),
        "upper_bound_violations".to_string(),
        "lower_bound_violations".to_string(),
        "mean_rel_error".to_string(),
        "std_dev_rel_error".to_string(),
        "max_rel_error".to_string(),
        "sparsification_rate".to_string(),
    ];

    let mut experiment_result = ExperimentResult::new(output_filename, column_headers);

    let dataset_stats_column_headers = vec![
        "dataset".to_string(),
        "num_nodes".to_string(),
        "num_edges".to_string(),
        "notes".to_string(),
    ];

    let mut dataset_stats = ExperimentResult::new(dataset_stats_filename, dataset_stats_column_headers);
        
    for (input_filename, dataset_name) in input_filenames.iter().zip(dataset_names.iter()) {
        // track if it's the first experiment for this input file. if so, write its stats into the dataset sets csv file.
        let mut first = true;
        for jl_factor in [1.5, 4.0, 10.0] {
            for sketch_type in sketch_types {
                let mut sketch_name = "discrete";
                if sketch_type {sketch_name = "uniform";}
                let mut parameters = SparsifierParameters::new_default(true);
                parameters.jl_factor = jl_factor;
                parameters.sketch_seed = 0;
                parameters.sampling_seed = 0;
                parameters.epsilon = 0.5;
                parameters.sketch_uniform = sketch_type;


                println!("------------------------------------------------------------------------");
                println!("    Sparsifying {} with jl factor {:.2} and {} sketch", dataset_name, parameters.jl_factor, sketch_name);
                println!("------------------------------------------------------------------------");

                let stream = InputStream::new(input_filename, dataset_name);
                let (sparsifier, sparsification_stats) = stream.run_stream(&parameters, false, false);

                let cc_success = verify_ccs(&stream, &sparsifier);
                let quadratic_form_result = probe_quadratic_form(&stream, &sparsifier, dataset_name, 100);
                println!("------------------------------------------------------------------------");
                println!("      Finished {} with jl factor {:.2} and {} sketch", dataset_name, parameters.jl_factor, sketch_name);
                println!("------------------------------------------------------------------------");
                experiment_result.record_result(vec![
                    dataset_name.to_string(),
                    parameters.epsilon.to_string(),
                    sketch_name.to_string(),
                    sparsifier.jl_factor.to_string(),
                    sparsifier.jl_scaling_factor.to_string(),
                    sparsifier.jl_dim.to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::EvimComplete, BenchmarkPoint::Initialize).to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::JlSketchComplete, BenchmarkPoint::EvimComplete).to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::SolvesComplete, BenchmarkPoint::JlSketchComplete).to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::DiffNormsComplete, BenchmarkPoint::SolvesComplete).to_string(),
                    sparsifier.benchmarker.get_duration(BenchmarkPoint::ReweightingsComplete, BenchmarkPoint::DiffNormsComplete).to_string(),
                    cc_success.to_string(),
                    quadratic_form_result.upper_bound_violations.to_string(),
                    quadratic_form_result.lower_bound_violations.to_string(),
                    quadratic_form_result.mean_relative_error.to_string(),
                    quadratic_form_result.std_dev_relative_error.to_string(),
                    quadratic_form_result.max_relative_error.to_string(),
                    sparsification_stats.sparsification_rate().to_string(),
                ]);

                // if you haven't written the dataset information for this input file yet, do it now.
                if first {
                    dataset_stats.record_result(vec![
                        dataset_name.to_string(),
                        sparsifier.num_nodes.to_string(),
                        sparsification_stats.get_num_orig_edges().to_string(),
                        ".".to_string(),
                    ]);
                    first = false;
                }

                if writeout {
                    // write out sparsifier laplacian with informative filename
                    let output_filename = dataset_name;
                    let output_filepath = crate::OUTPUT_LAPLACIAN_PATH;
                    let mut sketch_type = "_discrete";
                    if sparsifier.sketch_uniform {sketch_type = "_uniform";}
                    let params = sparsifier.jl_scaling_factor.to_string() + &sketch_type;
                    let output_location = output_filepath.to_owned() + &output_filename.to_owned() + "_rust_static_" + &params.to_owned() + ".mtx";
                    println!("Writing to {}", output_location);
                    crate::utils::write_mtx(&output_location, &sparsifier.current_laplacian);
                }

            }
        }
    }
    experiment_result.finalize();
    dataset_stats.finalize();
}

pub fn space_use(input_filenames: &[&str], dataset_names: &[&str], writeout: bool) {
    println!("Running space use experiment with epsilon 0.25 and both uniform and discrete sketch types");
    let epsilon = 0.25;
    let sketch_types = [true, false]; //true for uniform, false for discrete (fix this later)
    let output_filename = format!("experiment_results/space/space_results.csv");
    let dataset_stats_filename = format!("experiment_results/space/dataset_stats.csv");
    let column_headers = vec![
        "dataset".to_string(),
        "epsilon".to_string(),
        "sketch_type".to_string(),
        "evim_time".to_string(),
        "jl_time".to_string(),
        "solve_time".to_string(),
        "diff_norm_time".to_string(),
        "reweight_time".to_string(),
        "cc_success".to_string(),
        "upper_bound_violations".to_string(),
        "lower_bound_violations".to_string(),
        "mean_rel_error".to_string(),
        "std_dev_rel_error".to_string(),
        "max_rel_error".to_string(),
        "sparsification_rate".to_string(),
    ];

    let mut experiment_result = ExperimentResult::new(output_filename, column_headers);

    let dataset_stats_column_headers = vec![
        "dataset".to_string(),
        "num_nodes".to_string(),
        "num_edges".to_string(),
        "notes".to_string(),
    ];

    let mut dataset_stats = ExperimentResult::new(dataset_stats_filename, dataset_stats_column_headers);
        
    for (input_filename, dataset_name) in input_filenames.iter().zip(dataset_names.iter()) {
        // track if it's the first experiment for this input file. if so, write its stats into the dataset sets csv file.
        let mut first = true;
        for sketch_type in sketch_types {
            let mut sketch_name = "discrete";
            if sketch_type {sketch_name = "uniform";}
            println!("------------------------------------------------------------------------");
            println!("    Sparsifying {} with epsilon {} and {} sketch", dataset_name, epsilon, sketch_name);
            println!("------------------------------------------------------------------------");
            let mut parameters = SparsifierParameters::new_default(true);
            parameters.jl_factor = 4.0;
            parameters.sketch_seed = 0;
            parameters.sampling_seed = 0;
            parameters.epsilon = epsilon;
            parameters.sketch_uniform = sketch_type;

            let stream = InputStream::new(input_filename, dataset_name);
            let (sparsifier, sparsification_stats) = stream.run_stream(&parameters, false, false);

            let cc_success = verify_ccs(&stream, &sparsifier);
            let quadratic_form_result = probe_quadratic_form(&stream, &sparsifier, dataset_name, 100);
            println!("------------------------------------------------------------------------");
            println!("      Finished {} with epsilon {} and {} sketch", dataset_name, epsilon, sketch_name);
            println!("------------------------------------------------------------------------");
            experiment_result.record_result(vec![
                dataset_name.to_string(),
                epsilon.to_string(),
                sketch_name.to_string(),
                sparsifier.benchmarker.get_duration(BenchmarkPoint::EvimComplete, BenchmarkPoint::Initialize).to_string(),
                sparsifier.benchmarker.get_duration(BenchmarkPoint::JlSketchComplete, BenchmarkPoint::EvimComplete).to_string(),
                sparsifier.benchmarker.get_duration(BenchmarkPoint::SolvesComplete, BenchmarkPoint::JlSketchComplete).to_string(),
                sparsifier.benchmarker.get_duration(BenchmarkPoint::DiffNormsComplete, BenchmarkPoint::SolvesComplete).to_string(),
                sparsifier.benchmarker.get_duration(BenchmarkPoint::ReweightingsComplete, BenchmarkPoint::DiffNormsComplete).to_string(),
                cc_success.to_string(),
                quadratic_form_result.upper_bound_violations.to_string(),
                quadratic_form_result.lower_bound_violations.to_string(),
                quadratic_form_result.mean_relative_error.to_string(),
                quadratic_form_result.std_dev_relative_error.to_string(),
                quadratic_form_result.max_relative_error.to_string(),
                sparsification_stats.sparsification_rate().to_string(),
            ]);

            // if you haven't written the dataset information for this input file yet, do it now.
            if first {
                dataset_stats.record_result(vec![
                    dataset_name.to_string(),
                    sparsifier.num_nodes.to_string(),
                    sparsification_stats.get_num_orig_edges().to_string(),
                    ".".to_string(),
                ]);
                first = false;
            }

            if writeout {
                // write out sparsifier laplacian with informative filename
                let output_filename = dataset_name;
                let output_filepath = crate::OUTPUT_LAPLACIAN_PATH;
                let mut sketch_type = "_discrete";
                if sparsifier.sketch_uniform {sketch_type = "_uniform";}
                let params = sparsifier.epsilon.to_string() + &sketch_type;
                let output_location = output_filepath.to_owned() + &output_filename.to_owned() + "_rust_static_" + &params.to_owned() + ".mtx";
                println!("Writing to {}", output_location);
                crate::utils::write_mtx(&output_location, &sparsifier.current_laplacian);
            }
        }
    }
    experiment_result.finalize();
    dataset_stats.finalize();
}

pub fn get_lap() -> Sparsifier<i32> {
    let input_filename = crate::INPUT_FILENAME_VIRUS;
    let dataset_name = "virus";
    let stream = InputStream::new(input_filename, dataset_name);
    let sparsifier = stream.produce_laplacian();
    sparsifier
}

// this simple test is used in conjunction with valgrind's massif tool to measure memory use during sparsification.
pub fn space_use_simple() {
    // let mut parameters = SparsifierParameters::<i32>::new_default(true);
    // parameters.jl_factor = 4.0;
    // parameters.sketch_seed = 0;
    // parameters.sampling_seed = 0;
    // parameters.epsilon = 0.25;
    // parameters.sketch_uniform = false;

    println!("virus laplacian from stream");
    let mut sparsifier = get_lap();

    println!("num nodes is {}, num edges is {}, nonzeros in laplacian is {}",
        sparsifier.num_nodes, sparsifier.num_edges(), sparsifier.current_laplacian.nnz());

    println!("sparsifying the dataset");

    sparsifier.sparsify(false);
}