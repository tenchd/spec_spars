use crate::stream::InputStream;
use crate::sparsifier::{Sparsifier,SparsifierParameters};
use crate::utils::{CustomIndex, CustomIndex::from_int, BenchmarkPoint};
use crate::tests::make_random_vector;
use ndarray::Array1;
use petgraph::algo::connected_components;
use petgraph::Graph;
use std::fs::File;
use csv::Writer;


struct ExperimentResult {
    output_filename: String,
    column_headers: Vec<String>,
    writer: csv::Writer<std::fs::File>,
}

impl ExperimentResult {
    fn new(output_filename: String, column_headers: Vec<String>) -> Self {
        let file = File::create(&output_filename).expect("Could not create file");
        let mut writer = Writer::from_writer(file);
        writer.write_record(&column_headers).expect("Could not write column headers");
        ExperimentResult {
            output_filename,
            column_headers,
            writer,
        }
    }

    fn record_result(&mut self, result: Vec<String>) {
        self.writer.write_record(&result).expect("Could not write record");
        self.writer.flush().expect("Could not flush writer");
    }

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
    return original_ccs ==  sparsified_ccs;
}

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

// first experiment: for mouse dataset, vary epsilon and sketch type. track the follwing for each run:
// dataset (for now all mouse)
// epsilon
// sketch type (discrete, uniform)
// runtime (broken down by task)
// whether observe CC failure, 
// quadratic form statistics (# failures, mean/std/max of relative error)
// sparsification rate
pub fn basic_exploration(input_filename: &str, dataset_name: &str) {
    let epsilons = [0.75, 0.5, 0.25];
    let sketch_types = [true, false]; //true for uniform, false for discrete (fix this later)
    let output_filename = format!("{}_experiment_results.csv", dataset_name);
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
            let (sparsifier, sparsification_stats) = stream.run_stream(&parameters, false);

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
        }
    }
    // each csv output line should be: dataset, epsilon, sketch type, runtime breakdown, CC success, # upper bound violations, # lower bound violations, mean rel error, std dev rel error, max rel error, sparsification rate


}