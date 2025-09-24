#![allow(unused)]
//#![feature(test)]
extern crate fasthash;
extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;
extern crate rand;
//extern crate test;

mod utils;
mod jl_sketch;
mod sparsifier;
mod stream;
mod tests;

use utils::{read_mtx, read_vecs_from_file_flat, l2_norm, read_sketch_from_mtx, convert_mtx_to_csv};
use jl_sketch::{populate_matrix};
use sparsifier::{Sparsifier};
use stream::InputStream;
use crate::{ffi::FlattenedVec};
use ndarray::Array2;
use sprs::CsMatI;


#[cxx::bridge]
mod ffi {

    struct FlattenedVec {
        vec: Vec<f64>,
        num_cols: usize,
        num_rows: usize,
    }

    unsafe extern "C++" {
        include!("cxx-test/include/example.h");

        //fn f(elements: Vec<Shared>) -> Vec<Shared>;

        fn go(shared_jl_cols: FlattenedVec) -> FlattenedVec;

        fn test_roll(jl_cols: FlattenedVec) -> FlattenedVec;

        fn sprs_test(col_ptrs: Vec<usize>, row_indices: Vec<usize>, values: Vec<f64>);

        fn sprs_correctness_test(col_ptrs: Vec<i32>, row_indices: Vec<i32>, values: Vec<f64>);

        fn run_solve_lap(shared_jl_cols: FlattenedVec, rust_col_ptrs: Vec<i32>, rust_row_indices: Vec<i32>, rust_values: Vec<f64>, num_nodes:i32) -> FlattenedVec;
    
        fn julia_test_solve(interop_jl_cols: FlattenedVec);
    }
}

impl FlattenedVec {
    //construct from Array2 type
    pub fn new(input_array: &Array2<f64>) -> ffi::FlattenedVec {
        let (num_rows, num_cols) = input_array.dim();
        let mut output = ffi::FlattenedVec{vec: vec![], num_rows: num_rows, num_cols: num_cols};

        for i in input_array.iter() {
            output.vec.push(*i);
        }   

        output
    }

    pub fn to_array2(&self) -> Array2<f64>{
        let mut output = Array2::<f64>::zeros((self.num_rows, self.num_cols));
        for i in 0..self.vec.len() {
            let row: usize = ((i as f64)/(self.num_rows as f64)).floor() as usize;
            let col: usize = i % self.num_cols;
            let value: f64 = *self.vec.get(i).unwrap();
            output[[row,col]] = value;
        }
        output
    }
}

// fn jl_sketch_dataset(input_filename: &str, output_filename: &str, jl_factor: f64, seed: u64){
//     //be careful about jl sketch dimensions now that i'm adding a row and col to the laplacian. check this later
//     let input_csc = read_mtx(input_filename, false);

//     let output_csc = jl_sketch_sparse(&input_csc, jl_factor, seed);
//     let dense_output = output_csc.to_dense().reversed_axes();
//     write_csv(output_filename, &dense_output);

// }

// currently assumes that you don't need to manage diagonals of input matrix. fix this later
fn precondition_and_solve(input_filename: &str, sketch_filename: &str, jl_factor: f64) -> FlattenedVec {
    //let filename = "data/fake_jl_multi.csv".to_string();

    let input_csc = read_mtx(input_filename);
    println!("{}", input_csc.outer_dims());

    //make sure diagonals are nonzero; generalize this later
    // for i in input_csc.diag_iter(){
    //     assert!(*i.unwrap() != 0.0 as f64)
    // }

    let n: usize = input_csc.cols();
    let m: usize = input_csc.rows();
    assert_eq!(n,m);

    //let seed: u64 = 1;
    //let jl_factor: f64 = 1.5;
    let jl_dim = ((n as f64).log2() *jl_factor).ceil() as usize;
    println!("output matrix should be {}x{}", n, jl_dim);

    // not used at the moment; need to convert it to flattenedvec so i can pass it to c++. try the to_dense function maybe?
    //let mut sketch_sparse_format: CsMat<f64> = CsMat::zero((jl_dim,n)).transpose_into();
    //println!("{}", sketch_sparse_format.outer_dims());

    //jl_sketch_sparse_blocked(&input_csc, &mut sketch_sparse_format, jl_dim, seed, block_rows, block_cols, display);
    //write_mtx("real_jl_sketch", &sketch_sparse_format);
    
    let shared_jl_cols_flat = read_vecs_from_file_flat(sketch_filename);
    let m: i32 = shared_jl_cols_flat.num_cols.try_into().unwrap();
    let n: i32 = shared_jl_cols_flat.num_rows.try_into().unwrap();
    println!("output matrix actually is {}x{}", n, m);    

    // let input_col_ptrs = input_csc.indptr().as_slice().unwrap().to_vec();
    // let input_row_indices = input_csc.indices().to_vec();

    let temp_input_col_ptrs = input_csc.indptr().as_slice().unwrap().to_vec();
    let input_col_ptrs: Vec<i32> = temp_input_col_ptrs.into_iter().map(|x| x as i32).collect();
    let temp_input_row_indices = input_csc.indices().to_vec();
    let input_row_indices: Vec<i32> = temp_input_row_indices.into_iter().map(|x| x as i32).collect();

    let input_values = input_csc.data().to_vec();

    println!("input col_ptrs size in rust: {:?}. first value: {}", input_col_ptrs.len(), input_col_ptrs[0]);
    println!("input row_indices size in rust: {:?}. first value: {}", input_row_indices.len(), input_row_indices[0]);
    println!("input values size in rust: {:?}. first value: {}", input_values.len(), input_values[0]);
    println!("nodes in input csc: {}, {}", input_csc.cols(), input_csc.rows());
    //ffi::sprs_correctness_test(input_col_ptrs, input_row_indices, input_values);
    ffi::run_solve_lap(shared_jl_cols_flat, input_col_ptrs, input_row_indices, input_values, n)
}



fn solve_test() {


    let jl_factor: f64 = 1.5;

    //let stream = InputStream::new("data/cage3.mtx");
    //stream.run_stream(epsilon, beta_constant, row_constant, verbose);


    let sketch_filename = "data/fake_jl_multi.csv";
    let input_filename = "/global/u1/d/dtench/cholesky/Parallel-Randomized-Cholesky/physics/parabolic_fem/parabolic_fem-nnz-sorted.mtx";

    //let sketch_filename = "data/virus_jl_sketch.csv";
    //let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";


    let solution = precondition_and_solve(input_filename, sketch_filename, jl_factor);

    println!("solution has {} cols, {} rows, and initial value {}", solution.num_cols, solution.num_rows, solution.vec[0]);

}

fn lap_test(input_filename: &str) {
    let seed: u64 = 1;
    let jl_factor: f64 = 1.5;

    let epsilon = 0.5;
    let beta_constant = 4;
    let row_constant = 2;
    let verbose = false;


    let stream = InputStream::new(input_filename);
    stream.run_stream(epsilon, beta_constant, row_constant, verbose, jl_factor, seed);
}

fn jl_visualize() {
    let seed: u64 = 2;
    let jl_factor: f64 = 1.5;
    let num_rows = 10000;
    let num_cols = 10000;
    let jl_dim = ((num_rows as f64).log2() *jl_factor).ceil() as usize;
    //let nnz = 25;
    //let csc = true;
    // let small_example = make_random_evim_matrix(num_rows, num_cols, csc);
    // println!("Example EVIM matrix:");
    // for (col_num, col_vec) in small_example.outer_iterator().enumerate(){
    //     print!("Col {}:  ", col_num);
    //     for (row_num, value) in col_vec.iter() {
    //         print!("row {}, val {:.2}.  ", row_num, value);
    //     }
    //     println!("");
    // }
    let mut sketch_matrix: Array2<f64> = Array2::zeros((num_cols,jl_dim)); 
    populate_matrix(&mut sketch_matrix, seed, jl_dim);
    
    let rows = sketch_matrix.dim().0;
    let cols = sketch_matrix.dim().1;
    println!("Example sketch matrix:");
    println!("should be -1 or 1 uniformly at random divided by sqrt(jl_dim).");
    println!("jl_dim is {}, sqrt(jl_dim) is {}, 1/sqrt(jl_dim) is {}", jl_dim, (jl_dim as f64).sqrt(), 1.0/(jl_dim as f64).sqrt());
    let mut sum = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            //print!("{} ", sketch_matrix[[i,j]]);
            sum += sketch_matrix[[i,j]];
        }
        //println!("");
    }
    println!("sum of sketch matrix entries = {}", sum);
}

fn tianyu_test() {
    let sketch_filename = "../tianyu-stream/data/virus_sketch_tianyu.mtx";
    let lap_filename = "../tianyu-stream/data/virus_lap_tianyu.mtx";

    let sketch: ffi::FlattenedVec = read_sketch_from_mtx(sketch_filename);
    let lap: CsMatI<f64, i32> = read_mtx(lap_filename);
    let n = lap.cols();
    let m = lap.rows();
    assert_eq!(n,m);

    for i in 0..28 {
        print!("{} ", sketch.vec.get(i).unwrap());
    }
    println!();

    println!("output matrix is {}x{}", sketch.num_rows, sketch.num_cols);

    let lap_col_ptrs: Vec<i32> = lap.indptr().as_slice().unwrap().to_vec();
    let lap_row_indices: Vec<i32> = lap.indices().to_vec();
    let lap_values: Vec<f64> = lap.data().to_vec();

    println!("input col_ptrs size in rust: {:?}. first value: {}", lap_col_ptrs.len(), lap_col_ptrs[0]);
    println!("input row_indices size in rust: {:?}. first value: {}", lap_row_indices.len(), lap_row_indices[0]);
    println!("input values size in rust: {:?}. first value: {}", lap_values.len(), lap_values[0]);
    println!("nodes in input csc: {}, {}", lap.cols(), lap.rows());
    //ffi::sprs_correctness_test(input_col_ptrs, input_row_indices, input_values);
    ffi::run_solve_lap(sketch, lap_col_ptrs, lap_row_indices, lap_values, n.try_into().unwrap());
    
}

fn jl_sketch_compare_test() {
    let sketch_filename = "../tianyu-stream/data/virus_sketch_tianyu.mtx";

    let sketch: ffi::FlattenedVec = read_sketch_from_mtx(sketch_filename);

    println!("interop matrix is {}x{}", sketch.num_rows, sketch.num_cols);

    ffi::julia_test_solve(sketch);
    
}

fn main() {
    //let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
    //let input_filename = "/global/u1/d/dtench/rust_spars/cxx-test/data/cage3.mtx";
    //let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/human_gene2/human_gene2.mtx";
    //lap_test(input_filename);
    //jl_visualize();
    //l2_norm("/global/u1/d/dtench/rust_spars/cxx-test/sketch/virus_sketch.mtx");
    tianyu_test();
    //let sketch_filename = "../tianyu-stream/data/virus_sketch_tianyu.mtx";
    //let sketch_output = "../tianyu-stream/data/virus_sketch_tianyu.csv";
    //convert_mtx_to_csv(sketch_filename, sketch_output);

    //jl_sketch_compare_test();
}