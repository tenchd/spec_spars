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

use utils::read_mtx;
use sparsifier::{Sparsifier};
use stream::InputStream;
use crate::{ffi::FlattenedVec};
use ndarray::Array2;


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

        fn run_solve_lap(shared_jl_cols: FlattenedVec, rust_col_ptrs: Vec<i32>, rust_row_indices: Vec<i32>, rust_values: Vec<f64>, num_nodes:i32, verbose: bool) -> FlattenedVec;
    
        #[allow(dead_code)]
        fn julia_test_solve(interop_jl_cols: FlattenedVec, lap_col_ptrs: Vec<i32>, lap_row_indices: Vec<i32>, lap_values: Vec<f64>, num_nodes:i32);

        #[allow(dead_code)]
        fn test_stager(interop_jl_cols: FlattenedVec, lap_col_ptrs: Vec<i32>, lap_row_indices: Vec<i32>, lap_values: Vec<f64>, num_nodes:i32, test_selector:i32, verbose: bool) -> bool;
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

pub fn lap_test(input_filename: &str, dataset_name: &str, epsilon: f64, verbose: bool, seed: u64, benchmark: bool) {
    
    // for now, users can't set these parameters
    let jl_factor: f64 = 1.5;
    let beta_constant = 4;
    let row_constant = 2;

    // not a test
    let test = false;

    let stream = InputStream::new(input_filename, dataset_name);
    stream.run_stream(epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmark, test);
}




// fn main() {
//     let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
//     //let input_filename = "data/virus_input.mtx";
//     //let input_filename = "/global/u1/d/dtench/rust_spars/cxx-test/data/cage3.mtx";
//     //let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/human_gene2/human_gene2.mtx";
//     lap_test(input_filename);

//     // let mat = ndarray::array![[-7.0, 1.0, 2.0, 4.0], 
//     //                                                     [1.0, -8.0, 3.0, 5.0],
//     //                                                     [2.0, 3.0, -11.0, 6.0],
//     //                                                     [4.0, 5.0, 6.0, -15.0], ];   
//     // let sprsmat = sprs::CsMatBase::csc_from_dense(mat.view(), 0.1);
//     // utils::write_mtx("data/test.mtx", &sprsmat);
// }


    //jl_visualize();
    //l2_norm("/global/u1/d/dtench/rust_spars/cxx-test/sketch/virus_sketch.mtx");
    //tianyu_test(); 