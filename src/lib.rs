#![allow(unused)]
//#![feature(test)]
#![feature(step_trait)]
#![feature(new_range_api)]
#![feature(import_trait_associated_functions)]

extern crate fasthash;
extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;
extern crate rand;
extern crate petgraph;

pub mod utils;
pub mod sparsifier;
pub mod stream;
pub mod tests;
pub mod experiments;

use utils::read_mtx;
use sparsifier::{Sparsifier,SparsifierParameters};
use stream::InputStream;
use crate::{ffi::FlattenedVec};
use ndarray::Array2;

//-----const variables used to standardize location of files used in correctness tests or experiments.-----
// filenames for original file inputs
pub const INPUT_FILENAME_VIRUS: &str = "/home/dtench/Programming/spec_spars/data/virus.mtx";
pub const INPUT_FILENAME_HUMAN1: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/human_gene1/human_gene1.mtx";
pub const INPUT_FILENAME_HUMAN2: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/human_gene2/human_gene2.mtx";
pub const INPUT_FILENAME_MOUSE: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/mouse_gene/mouse_gene.mtx";
pub const INPUT_FILENAME_K49: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/k49_norm_10NN/k49_norm_10NN.mtx";
pub const INPUT_FILENAME_BCSSTK30: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/bcsstk30/bcsstk30_nonpattern.mtx";
pub const INPUT_FILENAME_CAHEPPH: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/ca-HepPh/ca-HepPh_nonpattern.mtx";
pub const INPUT_FILENAME_COPAPERS: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/coPapersCiteseer/coPapersCiteseer_nonpattern.mtx";
pub const INPUT_FILENAME_GUPTA2: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/gupta2/gupta2_nonpattern.mtx";
pub const INPUT_FILENAME_GUPTA3: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/gupta3/gupta3_nonpattern.mtx";
pub const INPUT_FILENAME_LOCBRIGHT: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/loc-Brightkite/loc-Brightkite_nonpattern.mtx";
pub const INPUT_FILENAME_MYCIELSKIAN: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/mycielskian15/mycielskian15_nonpattern.mtx";
pub const INPUT_FILENAME_PATTERN1: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/pattern1/pattern1_nonpattern.mtx";
pub const INPUT_FILENAME_SMALL: &str = "/global/u1/d/dtench/rust_spars/spec_spars/data/small_input.mtx";
pub const INPUT_FILENAME_KRON13: &str = "/global/homes/d/dtench/m1982/david/dense_streams/kron13.mtx";
pub const INPUT_FILENAME_KRON14: &str = "/global/homes/d/dtench/m1982/david/dense_streams/kron14.mtx";
pub const INPUT_FILENAME_KRON15: &str = "/global/homes/d/dtench/m1982/david/dense_streams/kron15.mtx";
pub const INPUT_FILENAME_KRON17: &str = "/global/homes/d/dtench/m1982/david/dense_streams/kron17.mtx";
pub const INPUT_FILENAME_KTREE13: &str = "/global/homes/d/dtench/m1982/david/dense_streams/ktree13.mtx";
pub const INPUT_FILENAME_KTREE15: &str = "/global/homes/d/dtench/m1982/david/dense_streams/ktree15.mtx";
pub const INPUT_FILENAME_KTREE16: &str = "/global/homes/d/dtench/m1982/david/dense_streams/ktree16.mtx";
pub const INPUT_FILENAME_KTREE17: &str = "/global/homes/d/dtench/m1982/david/dense_streams/ktree17.mtx";


pub const DATASET_NAME_VIRUS: &str = "virus";
pub const DATASET_NAME_HUMAN1: &str = "human_gene1";
pub const DATASET_NAME_HUMAN2: &str = "human_gene2";
pub const DATASET_NAME_MOUSE: &str = "mouse_gene";
pub const DATASET_NAME_K49: &str = "k49_norm_10NN";
pub const DATASET_NAME_BCSSTK30: &str = "bcsstk30";
pub const DATASET_NAME_CAHEPPH: &str = "ca-HepPh";
pub const DATASET_NAME_COPAPERS: &str = "coPapersCiteseer";
pub const DATASET_NAME_GUPTA2: &str = "gupta2";
pub const DATASET_NAME_GUPTA3: &str = "gupta3";
pub const DATASET_NAME_LOCBRIGHT: &str = "loc-Brightkite";
pub const DATASET_NAME_MYCIELSKIAN: &str = "mycielskian15";
pub const DATASET_NAME_PATTERN1: &str = "pattern1";
pub const DATASET_NAME_SMALL: &str = "small_input";
pub const DATASET_NAME_KRON13: &str = "kron13";
pub const DATASET_NAME_KRON14: &str = "kron14";
pub const DATASET_NAME_KRON15: &str = "kron15";
pub const DATASET_NAME_KRON17: &str = "kron17";
pub const DATASET_NAME_KTREE13: &str = "ktree13";
pub const DATASET_NAME_KTREE15: &str = "ktree15";
pub const DATASET_NAME_KTREE16: &str = "ktree16";
pub const DATASET_NAME_KTREE17: &str = "ktree17";

const OUTPUT_LAPLACIAN_PATH: &str = "/global/homes/d/dtench/m1982/david/spec_spars_files/rust_sparse_output_unknown/";

#[cxx::bridge]
mod ffi {
    struct FlattenedVec {
        vec: Vec<f64>,
        num_cols: usize,
        num_rows: usize,
    }

    unsafe extern "C++" {
        include!("spec_spars/include/example.h");

        #[allow(dead_code)]
        fn go(shared_jl_cols: FlattenedVec) -> FlattenedVec;

        #[allow(dead_code)]
        fn test_roll(jl_cols: FlattenedVec) -> FlattenedVec;

        #[allow(dead_code)]
        fn sprs_test(col_ptrs: Vec<usize>, row_indices: Vec<usize>, values: Vec<f64>);

        #[allow(dead_code)]
        fn sprs_correctness_test(col_ptrs: Vec<i32>, row_indices: Vec<i32>, values: Vec<f64>);

        fn run_solve_lap(shared_jl_cols: FlattenedVec, rust_col_ptrs: Vec<i32>, rust_row_indices: Vec<i32>, rust_values: Vec<f64>, solver_output_filename: &str, num_nodes:i32, verbose: bool) -> FlattenedVec;
    
        #[allow(dead_code)]
        fn julia_test_solve(interop_jl_cols: FlattenedVec, lap_col_ptrs: Vec<i32>, lap_row_indices: Vec<i32>, lap_values: Vec<f64>, num_nodes:i32);

        #[allow(dead_code)]
        fn test_stager(interop_jl_cols: FlattenedVec, lap_col_ptrs: Vec<i32>, lap_row_indices: Vec<i32>, lap_values: Vec<f64>, input_filename: &str, julia_lap_filename: &str, julia_sketch_product_filename: &str, solver_output_filename: &str, num_nodes:i32, test_selector:i32, verbose: bool) -> bool;

        #[allow(dead_code)]
        fn test_diff_norm(rust_lap_filename: &str, julia_sketch_product_filename: &str, solver_output_filename: &str) -> FlattenedVec;
    }
}

impl FlattenedVec {
    //construct from Array2 type
    pub fn new(input_array: &Array2<f64>) -> ffi::FlattenedVec {
        let (num_rows, num_cols) = input_array.dim();
        let mut output = ffi::FlattenedVec{vec: vec![], num_rows: num_rows, num_cols: num_cols};

        for i in input_array.iter() { //what order is this done in? presumably row major order?
            output.vec.push(*i);
        }   

        output
    }

    pub fn to_array2(&self) -> Array2<f64>{
        let mut output = Array2::<f64>::zeros((self.num_rows, self.num_cols));
        for i in 0..self.vec.len() {
            let row: usize = ((i as f64)/(self.num_cols as f64)).floor() as usize;
            let col: usize = i % self.num_cols;
            let value: f64 = *self.vec.get(i).unwrap();
            //println!("index {i} has value {value} and is put in row {row} and column {col}");
            output[[row,col]] = value;
        }
        output
    }
}

pub fn lap_test(input_filename: &str, dataset_name: &str, epsilon: f64, verbose: bool, sketch_seed: u64, sampling_seed: u64, benchmark: bool) {
    
    // for now, users can't set these parameters
    let jl_factor: f64 = 4.0;
    let jl_scaling_factor: f64 = (3.0_f64).sqrt();
    let beta_constant = 4;
    let row_constant = 2;
    let sketch_uniform = true;
    let parameters = SparsifierParameters::new(epsilon, beta_constant, row_constant, verbose, jl_factor, jl_scaling_factor, sketch_seed, sampling_seed, benchmark, sketch_uniform);

    // not a test
    let test = false;

    // don't write sparsifier as a file
    let writeout = false;

    let stream = InputStream::new(input_filename, dataset_name);
    stream.run_stream(&parameters, test, writeout);
}

pub fn run_basic_experiment() {
    let input_filenames = [
        // crate::INPUT_FILENAME_VIRUS,
        // crate::INPUT_FILENAME_HUMAN1, 
        // crate::INPUT_FILENAME_HUMAN2, 
        // crate::INPUT_FILENAME_MOUSE, 
        // crate::INPUT_FILENAME_K49, 
        // crate::INPUT_FILENAME_BCSSTK30, 
        // crate::INPUT_FILENAME_CAHEPPH, 
        // crate::INPUT_FILENAME_COPAPERS,
        // crate::INPUT_FILENAME_GUPTA2,
        // crate::INPUT_FILENAME_GUPTA3,
        // crate::INPUT_FILENAME_LOCBRIGHT,
        // crate::INPUT_FILENAME_MYCIELSKIAN,
        // crate::INPUT_FILENAME_PATTERN1
        // crate::INPUT_FILENAME_KRON13,
        // crate::INPUT_FILENAME_KRON14,
        // crate::INPUT_FILENAME_KRON15,
        // crate::INPUT_FILENAME_KRON17
        crate::INPUT_FILENAME_KTREE13,
        // crate::INPUT_FILENAME_KTREE15,
        // crate::INPUT_FILENAME_KTREE16,
        // crate::INPUT_FILENAME_KTREE17
    ];
        
    let dataset_names = [
        // crate::DATASET_NAME_VIRUS,
        // crate::DATASET_NAME_HUMAN1, 
        // crate::DATASET_NAME_HUMAN2, 
        // crate::DATASET_NAME_MOUSE, 
        // crate::DATASET_NAME_K49, 
        // crate::DATASET_NAME_BCSSTK30, 
        // crate::DATASET_NAME_CAHEPPH, 
        // crate::DATASET_NAME_COPAPERS,
        // crate::DATASET_NAME_GUPTA2,
        // crate::DATASET_NAME_GUPTA3,
        // crate::DATASET_NAME_LOCBRIGHT,
        // crate::DATASET_NAME_MYCIELSKIAN,
        // crate::DATASET_NAME_PATTERN1
        // crate::DATASET_NAME_KRON13,
        // crate::DATASET_NAME_KRON14,
        // crate::DATASET_NAME_KRON15,
        // crate::DATASET_NAME_KRON17
        crate::DATASET_NAME_KTREE13,
        // crate::DATASET_NAME_KTREE15,
        // crate::DATASET_NAME_KTREE16,
        // crate::DATASET_NAME_KTREE17
    ];

    let writeout = true;
    crate::experiments::basic_exploration(&input_filenames, &dataset_names, writeout);
}

pub fn run_jl_scaling_experiment(){
        let input_filenames = [
        // crate::INPUT_FILENAME_VIRUS,
        // crate::INPUT_FILENAME_HUMAN1, 
        // crate::INPUT_FILENAME_HUMAN2, 
        // crate::INPUT_FILENAME_MOUSE, 
        // crate::INPUT_FILENAME_K49, 
        // crate::INPUT_FILENAME_BCSSTK30, 
        // crate::INPUT_FILENAME_CAHEPPH, 
        // crate::INPUT_FILENAME_COPAPERS,
        // crate::INPUT_FILENAME_GUPTA2,
        // crate::INPUT_FILENAME_GUPTA3,
        // crate::INPUT_FILENAME_LOCBRIGHT,
        // crate::INPUT_FILENAME_MYCIELSKIAN,
        // crate::INPUT_FILENAME_PATTERN1
        //crate::INPUT_FILENAME_KRON13,
        // crate::INPUT_FILENAME_KRON14,
        // crate::INPUT_FILENAME_KRON15,
        // crate::INPUT_FILENAME_KRON17
        crate::INPUT_FILENAME_KTREE13,
        // crate::INPUT_FILENAME_KTREE15,
        // crate::INPUT_FILENAME_KTREE16,
        // crate::INPUT_FILENAME_KTREE17
    ];
        
    let dataset_names = [
        // crate::DATASET_NAME_VIRUS,
        // crate::DATASET_NAME_HUMAN1, 
        // crate::DATASET_NAME_HUMAN2, 
        // crate::DATASET_NAME_MOUSE, 
        // crate::DATASET_NAME_K49, 
        // crate::DATASET_NAME_BCSSTK30, 
        // crate::DATASET_NAME_CAHEPPH, 
        // crate::DATASET_NAME_COPAPERS,
        // crate::DATASET_NAME_GUPTA2,
        // crate::DATASET_NAME_GUPTA3,
        // crate::DATASET_NAME_LOCBRIGHT,
        // crate::DATASET_NAME_MYCIELSKIAN,
        // crate::DATASET_NAME_PATTERN1
        //crate::DATASET_NAME_KRON13,
        // crate::DATASET_NAME_KRON14,
        // crate::DATASET_NAME_KRON15,
        // crate::DATASET_NAME_KRON17
        crate::DATASET_NAME_KTREE13,
        // crate::DATASET_NAME_KTREE15,
        // crate::DATASET_NAME_KTREE16,
        // crate::DATASET_NAME_KTREE17
    ];

    let writeout = false;
    crate::experiments::jl_scaling_factor_sensitivity(&input_filenames, &dataset_names, writeout);
}

pub fn run_jl_dim_experiment(){
        let input_filenames = [
        // crate::INPUT_FILENAME_VIRUS,
        // crate::INPUT_FILENAME_HUMAN1, 
        // crate::INPUT_FILENAME_HUMAN2, 
        // crate::INPUT_FILENAME_MOUSE, 
        // crate::INPUT_FILENAME_K49, 
        // crate::INPUT_FILENAME_BCSSTK30, 
        // crate::INPUT_FILENAME_CAHEPPH, 
        // crate::INPUT_FILENAME_COPAPERS,
        // crate::INPUT_FILENAME_GUPTA2,
        // crate::INPUT_FILENAME_GUPTA3,
        // crate::INPUT_FILENAME_LOCBRIGHT,
        // crate::INPUT_FILENAME_MYCIELSKIAN,
        // crate::INPUT_FILENAME_PATTERN1
        //crate::INPUT_FILENAME_KRON13,
        // crate::INPUT_FILENAME_KRON14,
        crate::INPUT_FILENAME_KRON15,
        // crate::INPUT_FILENAME_KRON17
        // crate::INPUT_FILENAME_KTREE13,
        // crate::INPUT_FILENAME_KTREE15,
        // crate::INPUT_FILENAME_KTREE16,
        // crate::INPUT_FILENAME_KTREE17
    ];
        
    let dataset_names = [
        // crate::DATASET_NAME_VIRUS,
        // crate::DATASET_NAME_HUMAN1, 
        // crate::DATASET_NAME_HUMAN2, 
        // crate::DATASET_NAME_MOUSE, 
        // crate::DATASET_NAME_K49, 
        // crate::DATASET_NAME_BCSSTK30, 
        // crate::DATASET_NAME_CAHEPPH, 
        // crate::DATASET_NAME_COPAPERS,
        // crate::DATASET_NAME_GUPTA2,
        // crate::DATASET_NAME_GUPTA3,
        // crate::DATASET_NAME_LOCBRIGHT,
        // crate::DATASET_NAME_MYCIELSKIAN,
        // crate::DATASET_NAME_PATTERN1
        // crate::DATASET_NAME_KRON13,
        // crate::DATASET_NAME_KRON14,
        crate::DATASET_NAME_KRON15,
        // crate::DATASET_NAME_KRON17
        // crate::DATASET_NAME_KTREE13,
        // crate::DATASET_NAME_KTREE15,
        // crate::DATASET_NAME_KTREE16,
        // crate::DATASET_NAME_KTREE17
    ];

    let writeout = false;
    crate::experiments::jl_dim_sensitivity(&input_filenames, &dataset_names, writeout);
}

pub fn run_space_use_experiment() {
    let input_filenames = [
        crate::INPUT_FILENAME_VIRUS,
        // crate::INPUT_FILENAME_HUMAN1, 
        // crate::INPUT_FILENAME_HUMAN2, 
        // crate::INPUT_FILENAME_MOUSE, 
        // crate::INPUT_FILENAME_K49, 
        // crate::INPUT_FILENAME_BCSSTK30, 
        // crate::INPUT_FILENAME_CAHEPPH, 
        // crate::INPUT_FILENAME_COPAPERS,
        // crate::INPUT_FILENAME_GUPTA2,
        // crate::INPUT_FILENAME_GUPTA3,
        // crate::INPUT_FILENAME_LOCBRIGHT,
        // crate::INPUT_FILENAME_MYCIELSKIAN,
        // crate::INPUT_FILENAME_PATTERN1
        // crate::INPUT_FILENAME_KRON13,
        // crate::INPUT_FILENAME_KRON14,
        // crate::INPUT_FILENAME_KRON15,
        // crate::INPUT_FILENAME_KRON17
        // crate::INPUT_FILENAME_KTREE13,
        // crate::INPUT_FILENAME_KTREE15,
        // crate::INPUT_FILENAME_KTREE16,
        // crate::INPUT_FILENAME_KTREE17
    ];
        
    let dataset_names = [
        crate::DATASET_NAME_VIRUS,
        // crate::DATASET_NAME_HUMAN1, 
        // crate::DATASET_NAME_HUMAN2, 
        // crate::DATASET_NAME_MOUSE, 
        // crate::DATASET_NAME_K49, 
        // crate::DATASET_NAME_BCSSTK30, 
        // crate::DATASET_NAME_CAHEPPH, 
        // crate::DATASET_NAME_COPAPERS,
        // crate::DATASET_NAME_GUPTA2,
        // crate::DATASET_NAME_GUPTA3,
        // crate::DATASET_NAME_LOCBRIGHT,
        // crate::DATASET_NAME_MYCIELSKIAN,
        // crate::DATASET_NAME_PATTERN1
        // crate::DATASET_NAME_KRON13,
        // crate::DATASET_NAME_KRON14,
        // crate::DATASET_NAME_KRON15,
        // crate::DATASET_NAME_KRON17
        // crate::DATASET_NAME_KTREE13,
        // crate::DATASET_NAME_KTREE15,
        // crate::DATASET_NAME_KTREE16,
        // crate::DATASET_NAME_KTREE17
    ];
    let writeout = false;
    crate::experiments::space_use(&input_filenames, &dataset_names, writeout);
}