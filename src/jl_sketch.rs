use std::ops::Add;
use std::sync::mpsc;
use std::thread;
use std::hash::{Hash,Hasher};
use std::ops::Mul;

use crate::fasthash::FastHasher;
use crate::{ffi, utils};

use fasthash::xx::Hasher64 as XXHasher;
use ndarray::{Array1,Array2,s};
use sprs::{CsMatI, CsMatViewI};
use num_traits::cast;
use utils::{CustomIndex, CustomValue};

//maps hash function output to {-1,1} evenly
fn transform(input: i64) -> i64 {
    let result = ((input >> 31) * 2 - 1 ) as i64;
    result
}

// function to add value 'val' to position 'row', 'col' in sparse matrix. inefficient; production code should not use.
#[allow(dead_code)]
pub fn add_to_position<IndexType: CustomIndex, ValueType: CustomValue>(matrix: &mut CsMatI<ValueType, IndexType>, row: usize, col: usize, val:ValueType) {
    let location = matrix.get_mut(row,col);
    match location {
        Some(p) => *p += val,
        None => matrix.insert(row, col, val),
    }
}

//this function is used to find the value of a hash function seeded with seed, when given two positional arguments. Used to repeatably compute values in the JL sketch matrix
// by passing the coordinates of the position as the two arguments.
pub fn hash_with_inputs(seed: u64, input1: u64, input2: u64) -> i64 {
   let mut checkhash = XXHasher::with_seed(seed);
    input1.hash(&mut checkhash);
    input2.hash(&mut checkhash);
    let result = checkhash.finish() as u32;
    //println!("{}",result);
    transform(result as i64)
}

// used to completely fill a jl sketch matrix with random values from a hash function.
pub fn populate_matrix(input: &mut Array2<f64>, seed: u64, jl_dim: usize) {
    let rows = input.dim().0;
    let cols = input.dim().1;
    let scaling_factor = (jl_dim as f64).sqrt();
    for i in 0..rows {
        for j in 0..cols {
            input[[i,j]] += (hash_with_inputs(seed, i as u64, j as u64) as f64) / scaling_factor;
        }
    }
}

// used to compute a single row of a jl sketch matrix. entries should match that of the corresponding row in the matrix returned
// by populate_matrix() above
pub fn populate_row<IndexType: CustomIndex, ValueType: CustomValue>(input: &mut Array1<ValueType>, row: IndexType, col_start: IndexType, col_end: IndexType, seed: u64, jl_dim: IndexType){
    let num_cols = (col_end - col_start).index();
    let scaling_factor = (jl_dim.index() as f64).sqrt();
    for col in 0..num_cols {
        let actual_col = col+col_start.index(); //have to hash actual column value which should be col+col_start
        input[[col]] = cast::<f64, ValueType>((hash_with_inputs(seed, row.index() as u64, actual_col as u64) as f64) / scaling_factor).unwrap(); 
    }
}

// computes the jl sketch multiplication by creating the entire factor matrix and performing the mult with the sprs crate multiplication function.
// not scalable; used as a reference implementation for correctness.
#[allow(dead_code)]
pub fn jl_sketch_sparse<IndexType: CustomIndex>(og_matrix: &CsMatI<f64, IndexType>, jl_factor: f64, seed: u64, display: bool) -> Array2<f64> {
    let og_rows = og_matrix.rows();
    let og_cols = og_matrix.cols();
    let jl_dim = ((og_rows as f64).log2() *jl_factor).ceil() as usize;
    let mut sketch_matrix: Array2<f64> = Array2::zeros((og_cols,jl_dim));
    if display {println!("EVIM has {} rows and {} cols, jl sketch matrix has {} rows and {} cols", og_rows, og_cols, og_cols, jl_dim);}
    populate_matrix(&mut sketch_matrix, seed, jl_dim);
    if display {println!("populated sketch matrix");}
    let result = og_matrix.mul(&sketch_matrix);
    if display {println!("performed multiplication");}
    result
 }

 // this function JL sketches a sparse encoding of the input matrix and outputs in a sparse format as well. 
// it doesn't do blocked operations though, so it's still not scalable because it represents the entire
// dense sketch matrix at all times.
#[allow(dead_code)]
pub fn jl_sketch_sparse_flat<IndexType: CustomIndex>(og_matrix: &CsMatI<f64, IndexType>, jl_factor: f64, seed: u64, display: bool) -> ffi::FlattenedVec {
    let result = jl_sketch_sparse(og_matrix, jl_factor, seed, display);
    ffi::FlattenedVec::new(&result)
 }

// computes the jl sketch product of a block of the overall EVIM matrix. a single thread computes this.
pub fn jl_sketch_colwise<IndexType: CustomIndex>(og_matrix: &CsMatViewI<f64, IndexType>, block_number: usize, jl_dim: usize, seed: u64, _display: bool) -> Array2<f64> {
    let og_rows = og_matrix.rows();
    let mut output_block: Array2<f64> = Array2::zeros([og_rows, jl_dim]);
    for (col_ind, col_vec) in og_matrix.outer_iterator().enumerate() {
        // adjust col ind because this is a subblock of the total input matrix
        let true_col_ind = block_number*og_rows + col_ind;
        assert!(col_vec.nnz() == 2);
        let mut jl_sketch_row: Array1<f64> = Array1::zeros(jl_dim);
        populate_row(&mut jl_sketch_row, true_col_ind, 0, jl_dim, seed, jl_dim);
        for (row_ind, &value) in col_vec.iter() {
            let product_row: Array1<f64> = jl_sketch_row.clone() * value;
            let mut row_view = output_block.slice_mut(s![row_ind, ..]);
            row_view += &product_row;
        }
    }
    output_block
}

// performs the jl sketch multiplication by breaking up the EVIM into blocks of columns, and assigning the multiplication for each column to a thread.
pub fn jl_sketch_colwise_batch<IndexType: CustomIndex>(og_matrix: &CsMatI<f64, IndexType>, result_matrix: &mut CsMatI<f64, IndexType>, jl_dim: usize, seed: u64, display: bool) {
    let og_rows = og_matrix.rows();
    
    let (tx, rx) = mpsc::channel();
    thread::scope(|s| {
        for (block_index, block) in og_matrix.outer_block_iter(og_rows).enumerate() {
            let clone = tx.clone();
            s.spawn(move || {
                let output_block = jl_sketch_colwise(&block, block_index, jl_dim, seed, display);
                let sprs_block = CsMatI::<f64, IndexType>::csc_from_dense(output_block.view(), 0.0);
                clone.send(sprs_block).unwrap();
                drop(clone);
            });

        }
        drop(tx);
        for received in rx {
            *result_matrix = result_matrix.add(&received);
        }
    });
}

// outer function for the optimized jl sketch multiplication function.
pub fn jl_sketch_colwise_flat<IndexType: CustomIndex>(og_matrix: &CsMatI<f64, IndexType>, jl_factor: f64, seed: u64, display: bool) -> ffi::FlattenedVec {
    let og_rows = og_matrix.rows();
    let jl_dim = ((og_rows as f64).log2() *jl_factor).ceil() as usize;
    let mut result_matrix: CsMatI<f64, IndexType> = CsMatI::zero((og_rows, jl_dim)).into_csc();
    jl_sketch_colwise_batch(&og_matrix, &mut result_matrix, jl_dim, seed,  display);
    println!("performed jl sketch multiplication");
    ffi::FlattenedVec::new(&result_matrix.to_dense())
}
