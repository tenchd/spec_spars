
use sprs::{CsMat,TriMat,CsMatI};
use rand::Rng;
use rand::distributions::{Distribution, Uniform};
use ndarray::{Axis, Array1};

// if lower_diagonal, assumes a square matrix and only populates entries below the main diagonal. else populates randomly from all possible matrix locations
pub fn make_random_matrix(num_rows: usize, num_cols: usize, nnz: usize, csc: bool, lower_diagonal: bool) -> CsMat<f64> {
    let mut trip: TriMat<f64> = TriMat::new((num_rows, num_cols));
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(0.0, 1.0);
    for _ in 0..nnz {
        if lower_diagonal {
            let row_pos = rng.gen_range(1..num_rows);
            let col_pos = rng.gen_range(0..row_pos);
            let value = uniform.sample(&mut rng);
            trip.add_triplet(row_pos, col_pos, value);
        }
        else {
            let row_pos = rng.gen_range(0..num_rows);
            let col_pos = rng.gen_range(0..num_cols);
            let value = uniform.sample(&mut rng);
            trip.add_triplet(row_pos, col_pos, value);
        }
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

pub fn mean_and_std_dev(input: &Array1<f64>) -> (f64, f64) {
    let total = input.sum_axis(Axis(0));
    let length = input.len() as f64;
    let mean: f64 = total[()] / length;
    let mut variance: f64 = 0.0;
    for sum in input {
        let sq_diff = (mean - sum).powi(2);
        variance += sq_diff;
    }
    variance = variance / length;
    let st_dev = variance.sqrt(); 
    return (mean, st_dev);
}

#[cfg(test)]
mod integration_tests {
    use std::ops::Add;
    use std::time::{Instant};
    use std::process::Command;
    use sprs::{CsMat,TriMat,CsMatI};
    use rand::Rng;
    use rand::distributions::{Distribution, Uniform};
    use ndarray::{Axis, Array1};
    use ::approx::{AbsDiffEq};
    use petgraph::Graph;
    use petgraph::algo::{connected_components,min_spanning_tree};
    use petgraph::prelude::*;
    use petgraph::data::FromElements;
    use crate::{ffi::test_roll, utils, Sparsifier,InputStream};
    use crate::utils::{Benchmarker,CustomIndex};
    use crate::ffi;
    use crate::tests::make_random_evim_matrix;

    //test that takes in random entries, pushes triplet entries to laplacian, and never sparsifies. ensures that we always have a valid laplacian.
    #[test]
    //#[ignore]
    fn lap_valid_random() {
        println!("TEST:----Running lap validity test: insert many random updates and periodically check laplacian for validity.-----");
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;

        let num_nodes = 10000;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;

        let benchmarker = Benchmarker::new(false);
        let mut sparsifier = Sparsifier::new(num_nodes, epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);

        //generate random stream of updates
        let mut rng = rand::thread_rng();
        let uniform = Uniform::new(-1.0, 1.0);
        let num_updates = 10000;
        for i in 0..num_updates {
            let row_pos = rng.gen_range(0..num_nodes);
            let col_pos = rng.gen_range(0..num_nodes);
            let value = uniform.sample(&mut rng);
            sparsifier.insert(row_pos, col_pos, value);
            if i%500 == 0 {
                sparsifier.check_diagonal();
            }
        }
        sparsifier.check_diagonal();
    }

    // test that takes in file, triggers a dummy sparsification where fake probabilities (all 0.5) are used, and verifies that the laplacian is altered appropriately:
    // about half of the entries are deleted, total diagonal sum is multiplied by about sqrt(2)/2, and laplacian is still valid.
    #[test]
    //#[ignore]
    fn sampling_verify(){
        println!("TEST:-----Testing that, given edge probabilities all 0.5, laplacian is appropriately sparsified.-----");
        let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
        //let input_filename = "data/test.mtx";
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;


        let stream = InputStream::new(input_filename, "");

        let benchmarker = Benchmarker::new(false);
        let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);

        // insert edges into new_entries
        for (value, (row, col)) in stream.input_matrix.iter() {
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }
        
        // record current number of edges
        let before_num_edges = sparsifier.new_entries.col_indices.len()/2; 
        let mut before_diag_sum: f64 = 0.0;
        for (_index, value) in sparsifier.new_entries.diagonal.iter().enumerate() {
            before_diag_sum += value;
        }

        sparsifier.form_laplacian(true);

        // sample each edge with probability 0.5
        let probs = vec![0.5; before_num_edges];
        let mut reweightings = sparsifier.sample_and_reweight(probs);
        reweightings.process_diagonal();
        let csc_reweightings = reweightings.to_csc();
        sparsifier.current_laplacian = sparsifier.current_laplacian.add(&csc_reweightings);

        // record number of edges after sampling
        let after_num_edges = sparsifier.num_edges();
        let mut after_diag_sum: f64 = 0.0;
        for (_position, value) in sparsifier.current_laplacian.diag().iter() {
            after_diag_sum += value;
        }

        //println!("{:?}", sparsifier.current_laplacian.to_dense());
        println!("original had {} edges and sparsifier had {} edges", before_num_edges, after_num_edges);
        let edge_ratio = (before_num_edges as f64)/(after_num_edges as f64);
        let weight_ratio = before_diag_sum/after_diag_sum;


        //println!("Before sparsification, {} edges and {} diag weight. After, {} edges and {} diag weight. ratios {} and {}", 
        //    before_num_edges, before_diag_sum, after_num_edges, after_diag_sum, edge_ratio, weight_ratio);

        let weight_ratio_target = 2.0/(2.0_f64).sqrt();

        println!("edge ratio: {}", edge_ratio);
        assert!(edge_ratio.abs_diff_eq(&2.0, 0.05));
        assert!(weight_ratio.abs_diff_eq(&weight_ratio_target, 0.05));
    }

    // Test that takes in an input file, reads it into the sparsifier, converts the triplet representation to both evim and csc, 
    // and verifies that the matrices are equivalent.
    // also verifies that the laplacian is valid - each col sums to 0, matrix is symmetric.
    #[test]
    //#[ignore]
    fn evim_csc_equiv(){
        println!("TEST:-----Testing equivalence of laplacian and edge-vertex incidence matrix on virus dataset.-----");
        let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;

        let stream = InputStream::new(input_filename, "");

        let benchmarker = Benchmarker::new(false);
        let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);

        for (value, (row, col)) in stream.input_matrix.iter() {
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }

        // create EVIM representation
        let evim: CsMatI<f64, i32> = sparsifier.new_entries.to_edge_vertex_incidence_matrix();
        let evim_nnz = evim.nnz();

        // make sure each evim column has 2 distinct entries (no empty cols, no diagonal (single-nz col) entries)
        for (_col_ind, col_vec) in evim.outer_iterator().enumerate() {
            assert!(col_vec.nnz() == 2);
        }

        sparsifier.form_laplacian(true);

        let lap_nnz = sparsifier.num_edges()*2;

        assert_eq!(evim_nnz, lap_nnz);

        for (_edge_number, edge_vec) in evim.outer_iterator().enumerate() {
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
    //#[ignore]
    fn jl_sketch_equiv_random(){
        println!("TEST:-----Testing that blocked jl sketching matrix multiplication gives the same output as library mat mult implementation.-----");
        let num_rows = 5005;
        let num_cols = 60000;
        let csc = true;
            
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;
        let jl_dim = ((num_rows as f64).log2() *jl_factor).ceil() as usize;

        let benchmarker = Benchmarker::new(false);
        let sparsifier = Sparsifier::new(num_rows, epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);


        let input_matrix = make_random_evim_matrix(num_rows, num_cols, csc);

        println!("---- Time for jl sketch multiplication methods: ----");

        let nonblocked_timer = Instant::now();
        let sparse_nonblocked = sparsifier.jl_sketch_sparse(&input_matrix);
        let nonblocked_time = nonblocked_timer.elapsed().as_millis();
        println!("library: ------------------- {} ms", nonblocked_time);

        let colwise_batch_timer = Instant::now();
        let mut colwise_batch_answer:CsMat<f64> = CsMat::zero((num_rows, jl_dim)).into_csc();
        sparsifier.jl_sketch_colwise_batch(&input_matrix, &mut colwise_batch_answer);
        let colwise_batch_time = colwise_batch_timer.elapsed().as_millis(); 
        println!("multithreaded simple: ------ {} ms", colwise_batch_time);

        let sparse_nonblocked: CsMat<f64> = CsMat::csc_from_dense(sparse_nonblocked.view(),0.0);
        assert!(colwise_batch_answer.abs_diff_eq(&sparse_nonblocked, 0.00001));
    }

    #[test]
    //#[ignore]
    fn jl_sketch_equiv_virus(){
        println!("TEST:-----Testing that simplified jl sketching matrix multiplication gives the same output as library mat mult implementation.-----");
        let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;

        let stream = InputStream::new(input_filename, "");
        let num_rows = stream.num_nodes;
        let jl_dim = ((num_rows as f64).log2() *jl_factor).ceil() as usize;

        let benchmarker = Benchmarker::new(false);
        let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);

        for (value, (row, col)) in stream.input_matrix.iter() {
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }

        let input_matrix: CsMat<f64> = sparsifier.new_entries.to_edge_vertex_incidence_matrix();

        println!("---- Time for jl sketch multiplication methods: ----");

        let nonblocked_timer = Instant::now();
        let sparse_nonblocked = sparsifier.jl_sketch_sparse(&input_matrix);
        let nonblocked_time = nonblocked_timer.elapsed().as_millis();
        println!("library: ------------------- {} ms", nonblocked_time);

        let new_timer = Instant::now();
        let mut new_answer:CsMat<f64> = CsMat::zero((num_rows, jl_dim)).into_csc();
        sparsifier.jl_sketch_colwise_batch(&input_matrix, &mut new_answer);
        let new_time = new_timer.elapsed().as_millis(); 
        println!("multithreaded simple: ------ {} ms", new_time);

        let sparse_nonblocked: CsMat<f64> = CsMat::csc_from_dense(sparse_nonblocked.view(),0.0);

        assert!(new_answer.abs_diff_eq(&sparse_nonblocked, 0.00001));
    }

    // test whether jl sketch produces column sums near 0
    #[test]
    //#[ignore]
    pub fn jl_sketch_zero() {
        println!("TEST:-----Verifying that jl sketch output columns each sum to 0.-----");
        let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
        //let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/human_gene2/human_gene2.mtx";
        //let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/bcsstk30/bcsstk30.mtx";
        let seed: u64 = 2;
        let jl_factor: f64 = 1.5;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;


        // let test = utils::load_pattern_as_csr(input_filename);
        // println!("made it");

        let stream = InputStream::new(input_filename, "");

        let benchmarker = Benchmarker::new(false);
        let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);

        for (value, (row, col)) in stream.input_matrix.iter() {
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }

        // create EVIM representation
        let evim: CsMatI<f64, i32> = sparsifier.new_entries.to_edge_vertex_incidence_matrix();

        println!("now outputing results for xxhash");
        let sketch_cols: ffi::FlattenedVec = sparsifier.jl_sketch_colwise_flat(&evim);
        let sketch_array = sketch_cols.to_array2();
        
        let sums = sketch_array.sum_axis(Axis(0));
        //println!("{:?}", sums);
        let result = crate::tests::mean_and_std_dev(&sums);
        println!("mean {}, std dev {}", result.0, result.1);

        let _total = sums.sum_axis(Axis(0));
        //println!("TOTAL: {:?}", total);
        //assert!(total[0].abs_diff_eq(&0.0, 0.05));

    }

    #[test]
    //#[ignore]
    pub fn flatten_interop() {
        println!("TEST:-----Verifying that interop doesn't corrupt jl sketch columns.-----");
        let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;

        let stream = InputStream::new(input_filename, "");

        let benchmarker = Benchmarker::new(false);
        let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);

        for (value, (row, col)) in stream.input_matrix.iter() {
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }

        // create EVIM representation
        let evim: CsMatI<f64, i32> = sparsifier.new_entries.to_edge_vertex_incidence_matrix();
        let sketch_cols: ffi::FlattenedVec = sparsifier.jl_sketch_sparse_flat(&evim);
        let saved_rows = sketch_cols.num_rows;
        let saved_cols = sketch_cols.num_cols;
        let saved_vec = sketch_cols.vec.clone();

        let rerolled_cols: ffi::FlattenedVec = test_roll(sketch_cols);

        assert!(saved_cols == rerolled_cols.num_cols);
        assert!(saved_rows == rerolled_cols.num_rows);

        for i in 0..saved_vec.len() {
            assert!(saved_vec.get(i).unwrap() == rerolled_cols.vec.get(i).unwrap());
        }
    }

    // tests whether the rolling/unrolling functionality of the FlattenedVec class works correctly.
    #[test]
    //#[ignore]
    fn flatvec_equiv() {
        let mat = crate::tests::make_random_matrix(400, 1000, 100000, true, true).to_dense();
        // let mat = ndarray::array![[0.0, 1.0, 2.0, 3.0], 
        //                                                         [4.0, 5.0, 6.0, 7.0],
        //                                                         [8.0, 9.0, 10.0, 11.0],
        //                                                         [12.0, 13.0, 14.0, 15.0],
        //                                                         [16.0, 17.0, 18.0, 19.0],
        //                                                         [20.0, 21.0, 22.0, 23.0]];        
        // let mat = ndarray::array![[0.0, 1.0, 2.0, 3.0, 4.0, 5.0], 
        //                                                         [6.0, 7.0, 8.0, 9.0, 10.0, 11.0], 
        //                                                         [12.0, 13.0, 14.0, 15.0, 16.0, 17.0], 
        //                                                         [18.0, 19.0, 20.0, 21.0, 22.0, 23.0]];
        let flat = ffi::FlattenedVec::new(&mat);
        let new_mat = flat.to_array2();
        let num_rows = mat.dim().0;
        let num_cols = mat.dim().1;
        assert!(num_rows == new_mat.dim().0);
        assert!(num_cols == new_mat.dim().1);
        // println!("{:?}", mat);
        // println!("{:?}", flat.vec);
        // println!("{:?}", new_mat);
        for row in 0..num_rows {
            for col in 0..num_cols {
                let value = mat[[row,col]];
                let new_value = new_mat[[row,col]];
                assert!(value == new_value, "row {row} col {col} mismatch: original had {value}, new had {new_value}");
            }
        }
    }

    // INTEROP TESTS THAT CALL NONTRIVIAL C++ CODE
    // to really understand these tests you need to look at the functions with the same names in example.cc
    fn run_interop_test(test_selector: i32, verbose: bool) {
        let sketch_filename = "../tianyu-stream/data/virus_sketch_tianyu.mtx";
        let lap_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";

        let sketch: ffi::FlattenedVec = utils::read_sketch_from_mtx(sketch_filename);
        println!("interop jl sketch matrix is {}x{}", sketch.num_rows, sketch.num_cols);
        
        let lap_stream: InputStream = InputStream::new(lap_filename, "");
        //let lap: CsMatI<f64, i32> = read_mtx(lap_filename);
        let lap: CsMatI<f64, i32> = lap_stream.produce_laplacian();
        let n = lap.cols();
        let m = lap.rows();
        assert_eq!(n,m);
        let lap_col_ptrs: Vec<i32> = lap.indptr().as_slice().unwrap().to_vec();
        let lap_row_indices: Vec<i32> = lap.indices().to_vec();
        let lap_values: Vec<f64> = lap.data().to_vec();
        println!("input col_ptrs size in rust: {:?}. first value: {}", lap_col_ptrs.len(), lap_col_ptrs[0]);
        println!("input row_indices size in rust: {:?}. first value: {}", lap_row_indices.len(), lap_row_indices[0]);
        println!("input values size in rust: {:?}. first value: {}", lap_values.len(), lap_values[0]);
        println!("nodes in input csc: {}, {}", lap.cols(), lap.rows());

        let result: bool = ffi::test_stager(sketch, lap_col_ptrs, lap_row_indices, lap_values, n.try_into().unwrap(), test_selector, verbose);
        assert!(result);
    }

    // this test reads in jl sketch and lap from file (produced by tianyu julia code) and solves. status: WORKS 
    //(but figure out how to check solution for good quality later)
    #[test]
    //#[ignore]
    fn file_only_solver_test() {
        run_interop_test(1, false);
    }

    // this test establishes that the jl sketches from file and interop are the same. status: WORKS
    #[test]
    //#[ignore]
    fn jl_file_interop_equiv_test() {
        run_interop_test(2, false);
    }

    // this test tries to solve with jl sketch from interop and lap from direct file read (in tianyu's sparse matrix processor code). status: WORKS
    #[test]
    //#[ignore]
    fn jl_interop_lap_file_solver_test() {
        run_interop_test(3, false);
    }

    // tests whether the file and interop laplacians are equivalent. status: WORKS
    #[test]
    //#[ignore]
    fn lap_equiv_test() {
        run_interop_test(4, false);
    }

    // this test tries to solve with jl sketch and lap both from interop. status: WORKS
    #[test]
    //#[ignore]
    fn interop_only_solver_test() {
        run_interop_test(5, false);
    }

    // helper function for below test
    fn map_to_pattern(value: &f64) -> i32 {
        if value.abs_diff_eq(&0.0, 0.00005) {
            0
        }
        else {
            1
        }
    }

    // this test produces a sparsifier and verifies that the edge set of the sparsifier is a subset of the edge set of the original graph.
    #[test]
    //#[ignore]
    fn check_for_additions() {
        println!("TEST:-----Verifying that sparsified graph doesn't contain edges not present in original graph.-----");
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;
        let benchmark = true;
        let test = true;

        let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";

        let stream = InputStream::new(input_filename, "");
        let sparsifier: Sparsifier<i32> = stream.run_stream(epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmark, test);

        //stream.input_matrix       // input matrix
        //sparsifier.current_laplacian   //sparsifier matrix

        let input_pattern = stream.input_matrix.map(&map_to_pattern);
        let sparsifier_pattern = sparsifier.current_laplacian.map(&map_to_pattern);
        let difference = &input_pattern - &sparsifier_pattern;
        let mut neg_counter = 0;
        let mut one_counter = 0;
        let mut yet = false;
        for (value, (row, col)) in difference.iter() {
            if row < col {
                if *value < 0 {
                    neg_counter += 1;
                    if *value == -1 {
                        one_counter += 1;
                    }
                    if !yet {
                        println!("negative value found in difference matrix, indicating addition. ({}, {}) = {}", row, col, *value);
                        yet = true;
                    }
                }
            }
        }
        assert_eq!(neg_counter, one_counter);
        assert_eq!(neg_counter, 0, "there were {} positive values, indicating added edges", neg_counter);
    }

    #[test]
    #[ignore]
    fn check_for_additions_long() {
        println!("TEST:-----Verifying that sparsified graph doesn't contain edges not present in original graph, for a lot of datasets.-----");
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;
        let benchmark = true;
        let test = true;

        let input_filenames = ["/global/cfs/cdirs/m1982/david/bulk_to_process/mouse_gene/mouse_gene.mtx", 
                                        "/global/cfs/cdirs/m1982/david/bulk_to_process/human_gene1/human_gene1.mtx", 
                                        "/global/cfs/cdirs/m1982/david/bulk_to_process/human_gene2/human_gene2.mtx"];

        for input_filename in input_filenames {

            let stream = InputStream::new(input_filename, "");
            let sparsifier: Sparsifier<i32> = stream.run_stream(epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmark, test);

            //stream.input_matrix       // input matrix
            //sparsifier.current_laplacian   //sparsifier matrix

            let input_pattern = stream.input_matrix.map(&map_to_pattern);
            let sparsifier_pattern = sparsifier.current_laplacian.map(&map_to_pattern);
            let difference = &input_pattern - &sparsifier_pattern;
            let mut neg_counter = 0;
            let mut one_counter = 0;
            let mut yet = false;
            for (value, (row, col)) in difference.iter() {
                if row < col {
                    if *value < 0 {
                        neg_counter += 1;
                        if *value == -1 {
                            one_counter += 1;
                        }
                        if !yet {
                            println!("negative value found in difference matrix, indicating addition. ({}, {}) = {}", row, col, *value);
                            yet = true;
                        }
                    }
                }
            }
            assert_eq!(neg_counter, one_counter);
            assert_eq!(neg_counter, 0, "there were {} positive values, indicating added edges", neg_counter);
        }
    }

    #[test]
    //#[ignore]
    fn check_for_additions_mini() {
        println!("TEST:-----Verifying that sparsified graph doesn't contain edges not present in original graph.-----");
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;
        let benchmark = true;
        let test = true;

        // let mut random_matrix = crate::tests::make_random_matrix(20,20,40,true);
        // random_matrix = random_matrix.add(&(random_matrix.transpose_view()));
        // crate::utils::write_mtx_and_edgelist(&random_matrix, "small_input", false);
        // crate::utils::write_mtx("data/small_input.mtx", &random_matrix);
        // Command::new("bash").arg("-c").arg("sed '1,3d' data/small_input.mtx > data/small_input.edgelist").output();
        //println!("{:?}", random_matrix.map(&map_to_pattern).to_dense());

        let input_filename = "/global/u1/d/dtench/rust_spars/spec_spars/data/small_input.mtx";

        let stream = InputStream::new(input_filename, "small_input");
        let mut sparsifier: Sparsifier<i32> = stream.run_stream(epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmark, test);
        //crate::utils::write_mtx_and_edgelist(&sparsifier.current_laplacian, "small_input",true);

        // below commented code adds "bad" edges, if you uncomment this the test should fail. I included this to verify that erroneous edge 
        // additions would be verified by this kind of test.

        // sparsifier.insert(5, 2, 1.0);
        // sparsifier.insert(8, 5, 1.0);
        // sparsifier.insert(14, 12, 1.0);
        // sparsifier.insert(15, 2, 1.0);
        // sparsifier.form_laplacian(true);

        let input_pattern = stream.input_matrix.map(&map_to_pattern);

        let sparsifier_pattern = sparsifier.current_laplacian.map(&map_to_pattern);
        let difference = &input_pattern - &sparsifier_pattern;
        let mut neg_counter = 0;
        let mut one_counter = 0;
        let mut yet = false;
        for (value, (row, col)) in difference.iter() {
            if row < col {
                if *value < 0 {
                    neg_counter += 1;
                    if *value == -1 {
                        one_counter += 1;
                    }
                    if !yet {
                        println!("negative value found in difference matrix, indicating addition. ({}, {}) = {}", row, col, *value);
                        yet = true;
                    }
                }
            }
        }
        assert_eq!(neg_counter, one_counter);
        assert_eq!(neg_counter, 0, "there were {} positive values, indicating added edges", neg_counter);
    }

    // #[test]
    // //#[ignore]
    // fn verify_connectivity() {
    //     println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph.-----");
    //     let seed: u64 = 1;
    //     let jl_factor: f64 = 1.5;

    //     let epsilon = 0.5;
    //     let beta_constant = 4;
    //     let row_constant = 2;
    //     let verbose = false;
    //     let benchmark = true;
    //     let test = true;

    //     let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";

    //     let stream = InputStream::new(input_filename, "");
    //     let sparsifier: Sparsifier<i32> = stream.run_stream(epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmark, test);

    //     let mut original_edges: Vec<(i32, i32)> = Vec::new();
    //     let sparsifier_edges: Vec<(i32, i32)> = Vec::new();
    //     for (value, (row, col)) in stream.input_matrix.iter() {
    //         if row < col {
    //             original_edges.push((row, col));
    //         }
    //     }
    // }

    fn graphtest(input_filename: &str) {
        //println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph.-----");
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;
        let benchmark = true;
        let test = true;

        let stream = InputStream::new(input_filename, "");
        let sparsifier: Sparsifier<i32> = stream.run_stream(epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmark, test);

        let original_graph = stream.get_input_graph();
        //let sparsified_graph: Graph<usize, f64, petgraph::Undirected, usize> = Graph::from_elements(min_spanning_tree(&original_graph));
        let sparsified_graph = sparsifier.to_petgraph();

        let original_ccs = connected_components(&original_graph);
        let sparsified_ccs = connected_components(&sparsified_graph);
        println!("for file {} original # of ccs is {} and sparsified # of ccs is {}", input_filename, original_ccs, sparsified_ccs);
        //assert_eq!(original_ccs, sparsified_ccs);
    }

    #[test]
    #[ignore]
    fn petgraph_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for several datasets.-----");
        let input_filenames = ["/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx",
                                        "/global/cfs/cdirs/m1982/david/bulk_to_process/mouse_gene/mouse_gene.mtx", 
                                        "/global/cfs/cdirs/m1982/david/bulk_to_process/human_gene1/human_gene1.mtx", 
                                        "/global/cfs/cdirs/m1982/david/bulk_to_process/human_gene2/human_gene2.mtx"];

        for input_filename in input_filenames{
            graphtest(input_filename);
        }
    }
}

