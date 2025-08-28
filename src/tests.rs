use sprs::{CsMat,CsMatI,TriMat,TriMatI,CsVec,CsVecI};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use crate::{Sparsifier,InputStream};
use std::ops::Add;
use approx::AbsDiffEq;


pub fn make_random_matrix(num_rows: usize, num_cols: usize, nnz: usize, csc: bool) -> CsMat<f64> {
    let mut trip: TriMat<f64> = TriMat::new((num_rows, num_cols));
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(-1.0, 1.0);
    for _ in 0..nnz {
        let row_pos = rng.gen_range(0..num_rows);
        let col_pos = rng.gen_range(0..num_rows);
        let value = uniform.sample(&mut rng);
        trip.add_triplet(row_pos, col_pos, value);
    }
    if csc {
        return trip.to_csc();
    }
    trip.to_csr()
}

pub fn make_random_evim_matrix(num_rows: usize, num_cols: usize, csc: bool) -> CsMat<f64> {
    let mut trip: TriMat<f64> = TriMat::new((num_rows, num_cols));
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(0.0, 1.0);
    for i in 0..num_cols {
        let endpoint1 = rng.gen_range(0..num_rows-1);
        let endpoint2 = rng.gen_range(endpoint1+1..num_rows);
        let value = uniform.sample(&mut rng);
        trip.add_triplet(endpoint1, i, value);
        trip.add_triplet(endpoint2, i, -1.0*value);
    }
    if csc {
        return trip.to_csc();
    }
    trip.to_csr()
}

// pub fn make_random_matrix_i32(num_rows: usize, num_cols: usize, nnz: usize, csc: bool) -> CsMatI<f64, i32> {
//     let mut trip: TriMatI<f64, i32> = TriMat::new((num_rows, num_cols));
//     let mut rng = rand::thread_rng();
//     let uniform = Uniform::new(-1.0, 1.0);
//     for _ in 0..nnz {
//         let row_pos = rng.gen_range(0..num_rows);
//         let col_pos = rng.gen_range(0..num_rows);
//         let value = uniform.sample(&mut rng);
//         let row_pos_32: i32 = row_pos.try_into().unwrap();
//         let col_pos_32: i32 = col_pos.try_into().unwrap();
//         trip.add_triplet(row_pos_32, col_pos_32, value);
//     }
//     if csc {
//         return trip.to_csc();
//     }
//     trip.to_csr()
// }

pub fn make_random_vec(num_values: usize) -> CsVec<f64> {

    let indices: Vec<usize> = (0..num_values).collect();
    let mut values: Vec<f64> = vec![0.0; num_values];
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(-1.0, 1.0);
    // add random values for each entry except the last.
    for i in 0..num_values {
        if i%50000 == 0 {
            println!("{}",i);
        }
        let value = uniform.sample(&mut rng);
        if let Some(position) = values.get_mut(i) {
            *position += value;
        }
        //fake_jl_col.get_mut(i) += value;
    }
    println!("done");

    let rand_vec = CsVec::new(num_values, indices, values);
    rand_vec
}

#[cfg(test)]
mod tests {
    use crate::jl_sketch::{jl_sketch_sparse_blocked, jl_sketch_sparse};

    use super::*;
    use test::Bencher;

    //test that takes in random entries, pushes triplet entries to laplacian, and never sparsifies. ensures that we always have a valid laplacian.
    #[test]
    fn lap_valid_random() {
        println!("TEST:----Running lap validity test: insert many random updates and periodically check laplacian for validity.-----");
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;
        let block_rows: usize = 100;
        let block_cols: usize = 15000;
        let display: bool = false;

        let num_nodes = 10000;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;

        let mut sparsifier = Sparsifier::new(num_nodes, epsilon, beta_constant, row_constant, verbose, jl_factor, seed);

        //generate random stream of updates
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);
        let num_updates = 10000;
        for i in 0..num_updates {
            let row_pos = rng.gen_range(0..num_nodes);
            let col_pos = rng.gen_range(0..num_nodes);
            let value = uniform.sample(&mut rng);
            sparsifier.insert(row_pos, col_pos, value);
            if (i%500 == 0) {
                sparsifier.check_diagonal();
            }
        }
        sparsifier.check_diagonal();
    }

    // TODO: test that takes in file, triggers a dummy sparsification where fake probabilities (all 0.5) are used, and verifies that the laplacian is altered appropriately:
    // about half of the entries are deleted, total diagonal sum is multiplied by about sqrt(2)/2, and laplacian is still valid.
    #[test]
    fn sampling_verify(){
        println!("TEST:-----Testing that, given edge probabilities all 0.5, laplacian is appropriately sparsified.-----");
        let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;
        let block_rows: usize = 100;
        let block_cols: usize = 15000;
        let display: bool = false;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;

        let add_node = false;

        let stream = InputStream::new(input_filename, add_node);

        let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose, jl_factor, seed);

        for (value, (row, col)) in stream.input_matrix.iter() {
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }

        let before_num_edges = sparsifier.new_entries.col_indices.len()/2; 
        let mut before_diag_sum: f64 = 0.0;
        for (index, value) in sparsifier.new_entries.diagonal.iter().enumerate() {
            before_diag_sum += value;
        }

        let end_early = false;
        let test = true;
        sparsifier.sparsify(end_early, test);

        let after_num_edges = sparsifier.num_edges();
        let mut after_diag_sum: f64 = 0.0;
        for (_position, value) in sparsifier.current_laplacian.diag().iter() {
            after_diag_sum += value;
        }

        let edge_ratio = (before_num_edges as f64)/(after_num_edges as f64);
        let weight_ratio = before_diag_sum/after_diag_sum;


        //println!("Before sparsification, {} edges and {} diag weight. After, {} edges and {} diag weight. ratios {} and {}", 
        //    before_num_edges, before_diag_sum, after_num_edges, after_diag_sum, edge_ratio, weight_ratio);

        let weight_ratio_target = 2.0/(2.0_f64).sqrt();

        assert!(edge_ratio.abs_diff_eq(&2.0, 0.05));
        assert!(weight_ratio.abs_diff_eq(&weight_ratio_target, 0.05));
    }

    // Test that takes in an input file, reads it into the sparsifier, converts the triplet representation to both evim and csc, 
    // and verifies that the matrices are equivalent.
    // also verifies that the laplacian is valid - each col sums to 0, matrix is symmetric.
    #[test]
    fn evim_csc_equiv(){
        println!("TEST:-----Testing equivalence of laplacian and edge-vertex incidence matrix on virus dataset.-----");
        let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;
        let block_rows: usize = 100;
        let block_cols: usize = 15000;
        let display: bool = false;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;

        let add_node = false;

        let stream = InputStream::new(input_filename, add_node);

        let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose, jl_factor, seed);

        for (value, (row, col)) in stream.input_matrix.iter() {
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }

        // create EVIM representation
        let evim: CsMatI<f64, i32> = sparsifier.new_entries.to_edge_vertex_incidence_matrix();
        let evim_nnz = evim.nnz();

        // create diagonal values for new entries
        sparsifier.new_entries.process_diagonal();
        // get the new entries in csc format
        let new_stuff = sparsifier.new_entries.clone().to_csc();
        // clear the new entries from the triplet representation
        sparsifier.new_entries.delete_state();
        // add the new entries to the laplacian
        sparsifier.current_laplacian = sparsifier.current_laplacian.add(&new_stuff);

        let lap_nnz = sparsifier.num_edges()*2;

        assert_eq!(evim_nnz, lap_nnz);

        for (edge_number, edge_vec) in evim.outer_iterator().enumerate() {
            //println!("{}", edge_number);
            let mut indices: Vec<i32> = vec![];
            let mut values: Vec<f64> = vec![];
            for (endpoint, value) in edge_vec.iter() {
                indices.push(endpoint.try_into().unwrap());
                values.push(*value);
            }
            assert!(indices.len() == 2);
            assert!(values.len() == 2);
            assert!(values[0] == -1.0*values[1]);
            let row: usize = indices[0].try_into().unwrap();
            let col: usize = indices[1].try_into().unwrap();
            let value = values[1];
            let lap_value = *sparsifier.current_laplacian.get(row, col).unwrap();
            assert!(lap_value == value);
        }
        println!("evim and lap equivalent");
        sparsifier.check_diagonal();
    }

    //verifies that blocked jl sketching matrix multiplication gives the same output as library mat mult implementations.
    #[test]
    fn jl_sketch_equiv(){
        println!("TEST:-----Testing that blocked jl sketching matrix multiplication gives the same output as library mat mult implementation.-----");
        let num_rows = 50;
        let num_cols = 50;
        let nnz = 1000;
        let csc = true;
            
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;
        let jl_dim = ((num_rows as f64).log2() *jl_factor).ceil() as usize;

        let block_rows: usize = 9;
        let block_cols: usize = 17;
        let display: bool = false;

        let input_matrix = make_random_matrix(num_rows, num_cols, nnz, csc);
        let sparse_nonblocked = jl_sketch_sparse(&input_matrix, jl_factor, seed);
        let mut sparse_blocked:CsMat<f64> = CsMat::zero((num_cols, jl_dim)).into_csc();
        jl_sketch_sparse_blocked(&input_matrix, &mut sparse_blocked, jl_dim, seed, block_rows, block_cols, display);

        assert!(sparse_blocked == sparse_nonblocked);

    }

    // TODO: test where 



    //For benchmarking how long a sparse matrix x dense vector multiplication takes.
    pub fn spmv_basic(num_rows: usize, nnz: usize, csc: bool, b: &mut Bencher) {
        // let mut mat_type = "";
        // if csc {
        //     mat_type = "CSC";
        // }
        // else {
        //     mat_type = "CSR";
        // }
        // println!("Testing SPMV time for a {} x {} matrix in {} form with {} nonzeros", num_rows, num_rows, mat_type, nnz);
        let mat = make_random_matrix(num_rows, num_rows, nnz, csc);
        let vector = make_random_vec(num_rows);
        //let result = &mat * &vector;
        b.iter(|| &mat * &vector);
        //assert!(result.nnz()>0);
    }

    

    //benchmark a small multiplication when the matrix is in csc form
    #[bench]
    #[ignore]
    fn spmv1c(b: &mut Bencher){
        spmv_basic(10,20,true, b);
    }       

    //benchmark a small multiplication when the matrix is in csr form
    #[bench]
    #[ignore]
    fn spmv1r(b: &mut Bencher){
        spmv_basic(10,20,false, b);
    }
    
    #[bench]
    #[ignore]
    fn spmv2c(b: &mut Bencher){
        spmv_basic(100,2000,true, b);
    }

    #[bench]
    #[ignore]
    fn spmv2r(b: &mut Bencher){
        spmv_basic(100,2000,false, b);
    }

    #[bench]
    #[ignore]
    fn spmv3c(b: &mut Bencher){
        spmv_basic(1000,200000,true, b);
    }

    #[bench]
    #[ignore]
    fn spmv3r(b: &mut Bencher){
        spmv_basic(1000,200000,false, b);
    }

    #[bench]
    #[ignore]
    fn spmv4c(b: &mut Bencher){
        spmv_basic(200000,4000000,true, b);
    }

    #[bench]
    #[ignore]
    fn spmv4r(b: &mut Bencher){
        spmv_basic(200000,4000000,false, b);
    }


}