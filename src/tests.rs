use sprs::{CsMat,CsMatI,TriMat,TriMatI,CsVec,CsVecI};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use crate::{Sparsifier,InputStream};
use std::ops::Add;


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
    use super::*;
    use test::Bencher;

    // test that takes in random entries, pushes triplet entries to laplacian, and never sparsifies. ensures that we always have a valid laplacian.

    // test that takes in file, triggers a dummy sparsification where fake probabilities (all 0.5) are used, and verifies that the laplacian is altered appropriately:
    // about half of the entries are deleted, total diagonal sum is multiplied by about sqrt(2)/2, and laplacian is still valid.

    // Test that takes in an input file, reads it into the sparsifier, converts the triplet representation to both evim and csc, 
    // and verifies that the matrices are equivalent.
    #[test]
    fn evim_csc_equiv(){
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

        // create diagonal values for new entries
        sparsifier.new_entries.process_diagonal();
        // get the new entries in csc format
        let new_stuff = sparsifier.new_entries.clone().to_csc();
        // clear the new entries from the triplet representation
        sparsifier.new_entries.delete_state();
        // add the new entries to the laplacian
        sparsifier.current_laplacian = sparsifier.current_laplacian.add(&new_stuff);

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
        sparsifier.check_diagonal();
    }






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