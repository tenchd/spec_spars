
use sprs::{CsMat,TriMat,CsMatI,CsVecI};
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

pub fn make_random_vector(length: usize) -> CsVecI::<f64, i32> {
    let mut rng = rand::thread_rng();
    let uniform = Uniform::new(-1.0, 1.0);
    let mut data: Vec<f64> = vec![0.0; length];
    let indices = (0..length as i32).collect();
    for i in 0..length {
        data[i] = uniform.sample(&mut rng);
    }
    CsVecI::<f64, i32>::new(length, indices, data)
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
    use sprs::{CsMat,TriMat,CsMatI, CsVecI};
    use rand::Rng;
    use rand::distributions::{Distribution, Uniform};
    use ndarray::{Axis, Array1, Array2};
    use ::approx::{AbsDiffEq};
    use petgraph::Graph;
    use petgraph::algo::{connected_components,min_spanning_tree};
    use petgraph::prelude::*;
    use petgraph::data::FromElements;
    use crate::{ffi::test_roll, utils, Sparsifier,InputStream};
    use crate::utils::{Benchmarker,CustomIndex};
    use crate::ffi;
    use crate::tests::{make_random_evim_matrix, make_random_vector};
    use crate::sparsifier::{Triplet, SparsifierParameters};

    //-----static variables used to standardize location of files used in correctness tests.-----
    // filenames for original file inputs
    static INPUT_FILENAME_VIRUS: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/virus/virus.mtx";
    static INPUT_FILENAME_HUMAN1: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/human_gene1/human_gene1.mtx";
    static INPUT_FILENAME_HUMAN2: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/human_gene2/human_gene2.mtx";
    static INPUT_FILENAME_MOUSE: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/mouse_gene/mouse_gene.mtx";
    static INPUT_FILENAME_K49: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/k49_norm_10NN/k49_norm_10NN.mtx";
    static INPUT_FILENAME_BCSSTK30: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/bcsstk30/bcsstk30_nonpattern.mtx";
    static INPUT_FILENAME_CAHEPPH: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/ca-HepPh/ca-HepPh_nonpattern.mtx";
    static INPUT_FILENAME_COPAPERS: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/coPapersCiteseer/coPapersCiteseer_nonpattern.mtx";
    static INPUT_FILENAME_GUPTA2: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/gupta2/gupta2_nonpattern.mtx";
    static INPUT_FILENAME_GUPTA3: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/gupta3/gupta3_nonpattern.mtx";
    static INPUT_FILENAME_LOCBRIGHT: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/loc-Brightkite/loc-Brightkite_nonpattern.mtx";
    static INPUT_FILENAME_MYCIELSKIAN: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/mycielskian15/mycielskian15_nonpattern.mtx";
    static INPUT_FILENAME_PATTERN1: &str = "/global/cfs/cdirs/m1982/david/bulk_to_process/pattern1/pattern1_nonpattern.mtx";
    static INPUT_FILENAME_SMALL: &str = "/global/u1/d/dtench/rust_spars/spec_spars/data/small_input.mtx";

    // filename for solver output file; empty string means it writes no output
    static SOLVER_OUTPUT_FILENAME: &str= "";

    // filename for laplacian file written out by rust
    static RUST_LAP_FILENAME: &str = "/global/cfs/cdirs/m1982/david/spec_spars_files/misc/rust_laplacian.mtx";

    // filename for laplacian written out by julia file
    static JULIA_LAP_FILENAME: &str = "/global/cfs/cdirs/m1982/david/spec_spars_files/julia_output/julia_lap.mtx";
    // evim from julia run
    static JULIA_EVIM_FILENAME: &str = "/global/cfs/cdirs/m1982/david/spec_spars_files/julia_output/julia_evim.mtx";
    // sketch factor matrix from julia 
    static JULIA_SKETCH_FACTOR_FILENAME: &str = "/global/cfs/cdirs/m1982/david/spec_spars_files/julia_output/julia_sketch_factor.csv";
    // sketch product matrix from julia, in both csv and mtx formats
    static JULIA_SKETCH_PRODUCT_CSV_FILENAME: &str = "/global/cfs/cdirs/m1982/david/spec_spars_files/julia_output/julia_sketch_product.csv";
    static JULIA_SKETCH_PRODUCT_MTX_FILENAME: &str = "/global/cfs/cdirs/m1982/david/spec_spars_files/julia_output/julia_sketch_product.mtx";
    // solution matrix from julia
    static JULIA_SOLUTION_FILENAME: &str = "/global/cfs/cdirs/m1982/david/spec_spars_files/julia_output/julia_solution.mtx";
    // diff norms written out from julia run
    static JULIA_DIFF_NORMS_FILENAME: &str = "/global/cfs/cdirs/m1982/david/spec_spars_files/julia_output/julia_diff_norms.csv";
    // probabilities written out from julia run
    static JULIA_PROBS_FILENAME: &str = "/global/cfs/cdirs/m1982/david/spec_spars_files/julia_output/julia_probs.csv";
    // 
    static JULIA_OUTCOMES_FILENAME: &str = "/global/cfs/cdirs/m1982/david/spec_spars_files/julia_output/decisions.csv";

    // this eventually needs to be moved to somewhere else. but we need the rust lap file for later tests, and running it here ensures later tests will have it.
    #[test]
    fn write_lap(){
        let mut parameters: SparsifierParameters::<i32> = SparsifierParameters::new_default(false);
        parameters.jl_factor = 4.0;
        parameters.verbose = true;

        let stream = InputStream::new(INPUT_FILENAME_VIRUS, "");
        let lap = stream.produce_laplacian().current_laplacian;
        crate::utils::write_mtx(RUST_LAP_FILENAME, &lap);
    }

    //test that takes in random entries, pushes triplet entries to laplacian, and never sparsifies. 
    // ensures that we always have a valid laplacian, i.e., that the row and column sums are 0.
    #[test]
    //#[ignore]
    fn lap_valid_random() {
        println!("TEST:----Running lap validity test: insert many random updates and periodically check laplacian for validity.-----");
        for seed in 0..5{
            let num_nodes = 10000;
            let mut parameters = SparsifierParameters::new_default(false);
            parameters.sketch_seed = seed;
            parameters.sampling_seed = seed;

            let mut sparsifier = Sparsifier::new(num_nodes, &parameters);

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
    }

    // make sure laplacian produced by rust is identical to that produced by julia. in particular, columns should be in the same order.
    #[test]
    //#[ignore]
    fn lap_equiv_julia_rust(){
        println!("TEST:----Running lap equivalence test: compare rust laplacian with known good example from julia reference implementation.-----");
        let lap_stream: InputStream = InputStream::new(INPUT_FILENAME_VIRUS, "");
        let rust_lap: CsMatI<f64, i32> = lap_stream.produce_laplacian().current_laplacian;
        let julia_lap: CsMatI<f64, i32> = crate::utils::read_mtx(JULIA_LAP_FILENAME);
        let difference_lap = &rust_lap + &julia_lap; //this is a subtraction because the rust values are negative.

        let mut iteration_counter = 0;
        for (diff_value, (diff_row, diff_col)) in difference_lap.iter() {
            if diff_row != diff_col {
                assert!(diff_value.abs_diff_eq(&0.0, 0.00001), "iteration {} finds position {},{} with difference {}", iteration_counter, diff_row, diff_col, diff_value);
            }
            iteration_counter += 1;
        }
    }

    // test that takes in file, triggers a dummy sparsification where fake probabilities (all 0.5) are used, and verifies that the laplacian is altered appropriately:
    // about half of the entries are deleted, total diagonal sum is multiplied by about sqrt(2)/2, and laplacian is still valid.
    #[test]
    //#[ignore]
    fn sampling_verify(){
        println!("TEST:-----Testing that, given edge probabilities all 0.5, laplacian is appropriately sparsified.-----");
        
        for seed in 0..5 {
            let stream = InputStream::new(INPUT_FILENAME_VIRUS, "");
            let mut parameters: SparsifierParameters::<i32> = SparsifierParameters::new_default(false);
            parameters.sketch_seed = seed;
            parameters.sampling_seed = seed;
            let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), &parameters);

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
            sparsifier.apply_reweightings(reweightings, false);

            // record number of edges after sampling
            let after_num_edges = sparsifier.num_edges();
            let mut after_diag_sum: f64 = 0.0;
            for (_position, value) in sparsifier.current_laplacian.diag().iter() {
                after_diag_sum += value;
            }

            println!("original had {} edges and sparsifier had {} edges", before_num_edges, after_num_edges);
            let edge_ratio = (before_num_edges as f64)/(after_num_edges as f64);
            let weight_ratio = before_diag_sum/after_diag_sum;
            let weight_ratio_target = 2.0/(2.0_f64).sqrt();
            println!("edge ratio: {}", edge_ratio);
            assert!(edge_ratio.abs_diff_eq(&2.0, 0.05));
            assert!(weight_ratio.abs_diff_eq(&weight_ratio_target, 0.05));
        }
    }

        // currently this test doesn't fail if the row<col logic in reweight is flipped incorrectly.
    // need to design the test so it fails under those conditions.
    #[test]
    //#[ignore]
    fn sampling_test() {
        let mut parameters: SparsifierParameters::<i32> = SparsifierParameters::new_default(false);
        parameters.jl_factor = 4.0;
        parameters.verbose = true;
        let mut sparsifier: Sparsifier<i32> = Sparsifier::new(100, &parameters);

        // add edges from each node to the next one, making a path
        for i in 0..99 {
            let row = i+1;
            let col = i;
            sparsifier.insert(row, col, 1.0);
        }

        // add "shortcut" edges from each 0 mod 10 node to the next 9 mod 10 node
        for i in 0..10 {
            let col = i*10;
            let row = col+9;
            sparsifier.insert(row, col, 1.0);
        }

        sparsifier.form_laplacian(true);
        let original_graph = sparsifier.to_petgraph();

        // set probabilities so that every 2nd edge in the path is deleted, and no shortcut edges are deleted.
        // this results in 40 ccs if the deletions work properly.
        let mut probs = vec![1.0; sparsifier.num_edges()];
        for i in 0..10 {
            let ten_starting_index = i*11;
            let first_delete_index = ten_starting_index + 2;
            for j in 0..5 {
                let current_delete_index = first_delete_index + j*2;
                if current_delete_index == sparsifier.num_edges() {
                    break;
                }
                probs[current_delete_index] = 0.0;
            }
        }

        // sample according to the probabilites and make sure we end up with the right number of connected components in the sparsified graph.
        let mut reweightings = sparsifier.sample_and_reweight(probs);
        reweightings.process_diagonal();
        let csc_reweightings = reweightings.to_csc();
        sparsifier.current_laplacian = sparsifier.current_laplacian.add(&csc_reweightings);
        let sparsified_graph = sparsifier.to_petgraph();

        let original_ccs = connected_components(&original_graph);
        let sparsified_ccs = connected_components(&sparsified_graph);
        println!("original # of ccs is {} and sparsified # of ccs is {}", original_ccs, sparsified_ccs);
        assert_eq!(original_ccs, 1);
        assert_eq!(sparsified_ccs, 40);
    }

    // Test that takes in an input file, reads it into the sparsifier, converts the triplet representation to both evim and csc, 
    // and verifies that the matrices are equivalent.
    // also verifies that the laplacian is valid - each col sums to 0, matrix is symmetric.
    #[test]
    //#[ignore]
    fn evim_csc_equiv(){
        println!("TEST:-----Testing equivalence of laplacian and edge-vertex incidence matrix on virus dataset.-----");        
        let stream = InputStream::new(INPUT_FILENAME_VIRUS, "");
        let parameters: SparsifierParameters::<i32> = SparsifierParameters::new_default(false);
        let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), &parameters);

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

        // make sure rust evim and rust laplacian are equivalent, i.e., that for each edge, 
        // the two nonzeros in the corresponding evim column have the same value as the negative of the 
        // sqrt of the corresponding off-diagonal entry in the laplacian.
        println!("rust evim has dimensions {} x {}", evim.rows(), evim.cols());
        for (_edge_number, edge_vec) in evim.outer_iterator().enumerate() {
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
            //NOTE: changed this to compare the sqrt of lap value with evim value, because that's how evim is defined.
            let lap_value = (-1.0*sparsifier.current_laplacian.get(row, col).unwrap()).sqrt() * -1.0;
            assert!(lap_value == value, "rust evim edge {},{} with value {} mismatch with lap entry {}", row, col, value, lap_value);
        }
        println!("rust evim and lap equivalent");
        sparsifier.check_diagonal();

        // now make sure that rust evim matches the julia evim (read in from file)
        let julia_evim = utils::read_mtx_csr::<i32>(JULIA_EVIM_FILENAME).transpose_into();
        let julia_nnz = julia_evim.nnz();
        assert_eq!(julia_nnz, lap_nnz);
        println!("julia evim has dimensions {} x {}", julia_evim.rows(), julia_evim.cols());
        assert_eq!(evim.rows(), julia_evim.rows());
        assert_eq!(evim.cols(), julia_evim.cols());
        for (_edge_number, edge_vec) in julia_evim.outer_iterator().enumerate() {
            let mut indices: Vec<i32> = vec![];
            let mut values: Vec<f64> = vec![];
            for (endpoint, value) in edge_vec.iter() {
                indices.push(endpoint.try_into().unwrap());
                values.push(*value);
            }
            assert!(indices.len() == 2, "col {} has {} nonzeros. col: {:?}", _edge_number, indices.len(), edge_vec);
            assert!(values.len() == 2);
            assert!(values[0] == -1.0*values[1]);
            let row: usize = indices[0].try_into().unwrap();
            let col: usize = indices[1].try_into().unwrap();
            let value = values[1];
            //NOTE: changed this to compare the sqrt of lap value with evim value, because that's how evim is defined.
            let lap_value = (-1.0*sparsifier.current_laplacian.get(row, col).unwrap()).sqrt() * -1.0;
            assert!(lap_value.abs_diff_eq(&value, 0.00001), "julia evim edge {},{} with value {} mismatch with lap entry {}", row, col, value, lap_value);
        }
        println!("julia evim and lap equivalent");
    }

    // verifies that blocked jl sketching matrix multiplication gives the same output as library mat mult implementations.
    #[test]
    //#[ignore]
    fn jl_sketch_equiv_random(){
        println!("TEST:-----Testing that blocked jl sketching matrix multiplication gives the same output as library mat mult implementation.-----");
        let num_rows = 5005;
        let num_cols = 60000;
        let csc = true;
        let iterations = 5;

        let mut library_times = Array1::zeros(iterations);
        let mut multithread_times = Array1::zeros(iterations);
        
        for seed in 0..iterations {
            let mut parameters = SparsifierParameters::new_default(false);
            parameters.sketch_seed = seed as u64;
            parameters.sampling_seed = seed as u64;
            let sparsifier = Sparsifier::new(num_rows, &parameters);
            let jl_dim = sparsifier.jl_dim;

            let input_matrix = make_random_evim_matrix(num_rows, num_cols, csc);

            let nonblocked_timer = Instant::now();
            let sparse_nonblocked = sparsifier.jl_sketch_sparse(&input_matrix);
            let nonblocked_time = nonblocked_timer.elapsed().as_millis();
            library_times[seed] = nonblocked_time;

            let colwise_batch_timer = Instant::now();
            let mut colwise_batch_answer:CsMat<f64> = CsMat::zero((num_rows, jl_dim)).into_csc();
            sparsifier.jl_sketch_colwise_batch(&input_matrix, &mut colwise_batch_answer);
            let colwise_batch_time = colwise_batch_timer.elapsed().as_millis(); 
            multithread_times[seed] = colwise_batch_time;

            let sparse_nonblocked: CsMat<f64> = CsMat::csc_from_dense(sparse_nonblocked.view(),0.0);
            assert!(colwise_batch_answer.abs_diff_eq(&sparse_nonblocked, 0.00001));
        }

        println!("---- Time for jl sketch multiplication methods: ----");
        println!("library: ------------------- {} ms", library_times.mean().unwrap());
        println!("multithreaded simple: ------ {} ms", multithread_times.mean().unwrap());    
    }

    // verifies that blocked jl sketching matrix multiplication gives the same output as library mat mult implementations, for a handful of real-world datasets.
    #[test]
    //#[ignore]
    fn jl_sketch_equiv_virus(){
        println!("TEST:-----Testing that simplified jl sketching matrix multiplication gives the same output as library mat mult implementation.-----");
        let stream = InputStream::new(INPUT_FILENAME_VIRUS, "");
        let num_rows = stream.num_nodes;
        let parameters = SparsifierParameters::new_default(false);
        let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), &parameters);
        let jl_dim = sparsifier.jl_dim;

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

        let stream = InputStream::new(INPUT_FILENAME_VIRUS, "");
        let parameters: SparsifierParameters::<i32> = SparsifierParameters::new_default(false);
        let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), &parameters);

        for (value, (row, col)) in stream.input_matrix.iter() {
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }

        // create EVIM representation
        let evim: CsMatI<f64, i32> = sparsifier.new_entries.to_edge_vertex_incidence_matrix();

        //println!("now outputing results for xxhash");
        let sketch_cols: ffi::FlattenedVec = sparsifier.jl_sketch_colwise_flat(&evim);
        let sketch_array = sketch_cols.to_array2();
        
        let sums = sketch_array.sum_axis(Axis(0));
        let result = crate::tests::mean_and_std_dev(&sums);
        println!("mean {}, std dev {}", result.0, result.1);

        let total = sums.sum_axis(Axis(0));
        println!("TOTAL: {:?}", total);
        for i in 0..sums.len() {
            assert!(sums[i].abs_diff_eq(&0.0, 0.05));
        }

    }

    // sends a sketch matrix via interop from rust to C++, unpacks it, then repacks it and sends it back. verifies that it recieves the same thing it sent.
    #[test]
    //#[ignore]
    pub fn flatten_interop() {
        println!("TEST:-----Verifying that flattening doesn't corrupt jl sketch columns.-----");
        let stream = InputStream::new(INPUT_FILENAME_VIRUS, "");
        let parameters= SparsifierParameters::new_default(false);
        let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), &parameters);

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
        let flat = ffi::FlattenedVec::new(&mat);
        let new_mat = flat.to_array2();
        let num_rows = mat.dim().0;
        let num_cols = mat.dim().1;
        assert!(num_rows == new_mat.dim().0);
        assert!(num_cols == new_mat.dim().1);
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
        let sketch: ffi::FlattenedVec = utils::read_sketch_from_mtx(JULIA_SKETCH_PRODUCT_MTX_FILENAME);
        println!("interop jl sketch matrix is {}x{}", sketch.num_rows, sketch.num_cols);
        let lap_stream: InputStream = InputStream::new(INPUT_FILENAME_VIRUS, "");
        let lap: CsMatI<f64, i32> = lap_stream.produce_laplacian().current_laplacian;
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

        // need to pass original input mtx, julia lap, julia sketch, output. 
        let result: bool = ffi::test_stager(sketch, lap_col_ptrs, lap_row_indices, lap_values, INPUT_FILENAME_VIRUS, JULIA_LAP_FILENAME, JULIA_SKETCH_PRODUCT_CSV_FILENAME, SOLVER_OUTPUT_FILENAME, n.try_into().unwrap(), test_selector, verbose);
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
        let parameters = SparsifierParameters::new_default(true);
        let test = true;
        let stream = InputStream::new(INPUT_FILENAME_VIRUS, "");
        let sparsifier: Sparsifier<i32> = stream.run_stream(&parameters, test);

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
        let parameters = SparsifierParameters::new_default(true);
        let test = true;

        let input_filenames = [INPUT_FILENAME_MOUSE, INPUT_FILENAME_HUMAN1, INPUT_FILENAME_HUMAN2];

        for input_filename in input_filenames {

            let stream = InputStream::new(input_filename, "");
            let sparsifier: Sparsifier<i32> = stream.run_stream(&parameters, test);

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
        let parameters = SparsifierParameters::new_default(true);
        let test = true;

        let stream = InputStream::new(INPUT_FILENAME_SMALL, "small_input");
        let mut sparsifier: Sparsifier<i32> = stream.run_stream(&parameters, test);
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

    fn compare_diff_norms_to_julia(rust_diff_norms: &Vec<f64>, threshold: f64) {
        let file_diff_norms = crate::utils::read_csv_as_vec(JULIA_DIFF_NORMS_FILENAME).unwrap();
        assert!(rust_diff_norms.len() == file_diff_norms.len(), 
                "diff norm length mismatch. julia has length {}, rust has length {}", file_diff_norms.len(), rust_diff_norms.len());
        let diff_norm_array = Array1::from_vec(rust_diff_norms.clone());
        let julia_diff_norm_array = Array1::from_vec(file_diff_norms.clone());
        let difference = &diff_norm_array - &julia_diff_norm_array;
        let diff_mean = difference.mean().unwrap();
        let diff_std = difference.std(0.0);
        //let diff_max = difference.max().unwrap();
        assert!(diff_mean.abs() < threshold, "mean difference between rust and julia diff norms is {}, which is higher than expected", diff_mean);
        assert!(diff_std < threshold, "std dev of difference between rust and julia diff norms is {}, which is higher than expected", diff_std);
        //assert!(diff_max.abs() < threshold*5.0, "max difference between rust and julia diff norms is {}, which is higher than expected", diff_max);
        println!("the mean difference between rust and julia diff norms is {} and the std dev of the difference is {}.",
             diff_mean, diff_std);
        println!("mean rust diff norm is {}", diff_norm_array.mean().unwrap());
    }

    fn compare_probs_to_julia(rust_probs: &Vec<f64>, threshold: f64) {
        let file_probs = crate::utils::read_csv_as_vec(JULIA_PROBS_FILENAME).unwrap();
        assert!(rust_probs.len() == file_probs.len(), 
                "probs length mismatch. julia has length {}, rust has length {}", file_probs.len(), rust_probs.len());
        let probs_array = Array1::from_vec(rust_probs.clone());
        let julia_probs_array = Array1::from_vec(file_probs.clone());
        let difference = &probs_array - &julia_probs_array;
        let prob_diff_mean = difference.mean().unwrap();
        let prob_diff_std = difference.std(0.0);
        assert!(prob_diff_mean.abs() < threshold, "mean difference between rust and julia probs is {}, which is higher than expected", prob_diff_mean);
        assert!(prob_diff_std < threshold, "std dev of difference between rust and julia probs is {}, which is higher than expected", prob_diff_std);
        println!("the mean difference between rust and julia probs is {} and the std dev of the difference is {}.",
             prob_diff_mean, prob_diff_std);
        println!("mean rust prob is {}", probs_array.mean().unwrap());
    }

    // loads the solution from the julia implementation from file, computes the diff norms and probabilities in rust, 
    // and verifies that they match the julia diff norms/probs from file.
    #[test]
    fn verify_diff_norm_and_probs_via_julia_solution() {
        println!("TEST:-----Verifying that diff norm and probability calculations match that of the julia implementation.-----");
        let mut parameters: SparsifierParameters::<i32> = SparsifierParameters::new_default(false);
        parameters.jl_factor = 4.0;
        let file_solution_dense = crate::utils::read_mtx::<i32>(JULIA_SOLUTION_FILENAME).to_dense();
        let file_solution_flat = ffi::FlattenedVec::new(&file_solution_dense);
        let sparsifier = Sparsifier::from_matrix(RUST_LAP_FILENAME, &parameters);
        sparsifier.check_diagonal();
        let num_edges = sparsifier.num_edges();

        // computing and verifying diff norms
        let new_diff_norms = sparsifier.compute_diff_norms(num_edges, &file_solution_flat);
        // tight threshold because only deviation should be from floating point error
        let threshold = 0.0001;
        compare_diff_norms_to_julia(&new_diff_norms, threshold);

        // computing and verifying probabilities
        let new_probs = sparsifier.compute_probs(num_edges, &new_diff_norms);
        compare_probs_to_julia(&new_probs, threshold);
    }

    // this test loads the rust laplacian (for the virus dataset) from file, passes it to c++ via interop, then in c++ loads the julia sketch product from file, 
    // does the solve, sends the result back to rust via interop, and computes the diff norms and probs. verifies against julia norms and probs from file. status: WORKS, but need to figure out how to check solution quality later. also, the diff norms and probs are very close to julia's, but not as close as when we just read the solution from file, so maybe there's some numerical instability in the solve or the interop that's causing some small differences.
    #[test]
    //#[ignore]
    fn verify_diff_norm_and_probs_via_julia_sketch_product(){
        let mut parameters: SparsifierParameters<i32> = SparsifierParameters::new_default(false);
        parameters.jl_factor = 4.0;

        // loading laplacian, reading julia sketch product from file, and getting the solution via c++ solver.
        let mut sparsifier = Sparsifier::from_matrix(RUST_LAP_FILENAME, &parameters);
        sparsifier.check_diagonal();
        let num_edges = sparsifier.num_edges();
        let solution: ffi::FlattenedVec = ffi::test_diff_norm(RUST_LAP_FILENAME, JULIA_SKETCH_PRODUCT_CSV_FILENAME, SOLVER_OUTPUT_FILENAME);

        // computing diff norms in rust and comparing with julia diff norms from file.
        let new_diff_norms = sparsifier.compute_diff_norms(num_edges, &solution);
        let threshold = 0.01;
        compare_diff_norms_to_julia(&new_diff_norms, threshold);

        // computing probs in rust and comparing with julia diff norms from file.
        let probs = sparsifier.compute_probs(sparsifier.num_edges(), &new_diff_norms);
        compare_probs_to_julia(&probs, threshold);
    }

    // this test runs the virus dataset entirely from rust, until it gets the probabilities, and verifies these match the julia probabilities.
    #[test]
    //#[ignore]
    fn diff_norm_and_probs_via_rust(){
        let mut parameters: SparsifierParameters::<i32> = SparsifierParameters::new_default(false);
        parameters.jl_factor = 4.0;
        let stream = InputStream::new(INPUT_FILENAME_VIRUS, "");
        let mut sparsifier = Sparsifier::new(stream.num_nodes.try_into().unwrap(), &parameters);

        for (value, (row, col)) in stream.input_matrix.iter() {
            //assert!(*value >= 0.0);
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }

        let evim = &sparsifier.new_entries.to_edge_vertex_incidence_matrix();
        let flat_sketch_cols = sparsifier.jl_sketch_colwise_flat(&evim);
        sparsifier.form_laplacian(true);
        let num_edges = sparsifier.num_edges();        
        let probs = sparsifier.get_probs(num_edges, flat_sketch_cols);
        // higher threshold because different randomness in jl sketch compared to julia version
        let threshold = 0.05;
        compare_probs_to_julia(&probs, threshold);
        let mut reweightings= sparsifier.sample_and_reweight(probs);
        sparsifier.apply_reweightings(reweightings, true);
    }

    // quadatric form tests pass for discrete sketch but not uniform.
    // try more datasets?
    fn graphtest(input_filename: &str) {
        let iterations = 1;
        for seed in 0..iterations {
            println!("_-_-_-_-_ running connectivity test for file {} with seed {} _-_-_-_-_", input_filename, seed);
            let mut parameters = SparsifierParameters::new_default(true);
            parameters.jl_factor = 4.0;
            parameters.sketch_seed = seed;
            parameters.sampling_seed = seed;
            parameters.sketch_uniform = true;
            let test = true;
            let stream = InputStream::new(input_filename, "");
            let sparsifier: Sparsifier<i32> = stream.run_stream(&parameters, test);

            let original_graph = stream.get_input_graph();
            let sparsified_graph = sparsifier.to_petgraph();
            let original_ccs = connected_components(&original_graph);
            let sparsified_ccs = connected_components(&sparsified_graph);
            println!("for file {} original # of ccs is {} and sparsified # of ccs is {}", input_filename, original_ccs, sparsified_ccs);
            assert_eq!(original_ccs, sparsified_ccs);

            let original_state = stream.produce_laplacian();
            let probe_iterations = 100;
            let mut errors = Array1::zeros(probe_iterations);
            for i in 0..probe_iterations {
                let probe_vector = make_random_vector(sparsifier.num_nodes as usize);
                let original_product = original_state.compute_quadratic_form(&probe_vector);
                let sparsified_product = sparsifier.compute_quadratic_form(&probe_vector);
                let upper_bound = original_product * (1.0 + sparsifier.epsilon);
                let lower_bound = original_product * (1.0 - sparsifier.epsilon);
                let relative_error = (sparsified_product - original_product).abs() / original_product.abs();
                //println!("original quadratic form is {}, sparsified quadratic form is {}, upper bound is {}, lower bound is {}",
                //    original_product, sparsified_product, upper_bound, lower_bound);
                assert!(sparsified_product <= upper_bound, "i = {}. sparsifier quadratic form {} is higher than upper bound {}. rel error {}", 
                    i, sparsified_product, upper_bound, relative_error);
                assert!(sparsified_product >= lower_bound, "i = {}. sparsifier quadratic form {} is lower than lower bound {}. rel error {}", 
                    i, sparsified_product, lower_bound, relative_error);
                errors[i] = relative_error;
            }
            let max_error = errors.iter().copied().max_by(|x, y| x.partial_cmp(y).unwrap()).unwrap();
            println!("for file {}, 
                mean rel error ------ {}
                std dev rel error --- {}
                max rel error ------- {}",
                input_filename, errors.mean().unwrap(), errors.std(0.0), max_error);
        }
    }

    #[test]
    #[ignore]
    fn virus_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the virus dataset.-----");
        graphtest(INPUT_FILENAME_VIRUS);
    }

    #[test]
    #[ignore]
    fn mouse_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the mouse dataset.-----");
        graphtest(INPUT_FILENAME_MOUSE);
    }

    #[test]
    #[ignore]
    fn human1_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the human1 dataset.-----");
        graphtest(INPUT_FILENAME_HUMAN1);
    }

    #[test]
    #[ignore]
    fn human2_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the human2 dataset.-----");
        graphtest(INPUT_FILENAME_HUMAN2);
    }

    #[test]
    #[ignore]
    fn k49_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the k49 dataset.-----");
        graphtest(INPUT_FILENAME_K49);
    }

    #[test]
    #[ignore]
    fn bcsstk30_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the bcsstk30 dataset.-----");
        graphtest(INPUT_FILENAME_BCSSTK30);
    }

    #[test]
    #[ignore]
    fn cahepph_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the cahepph dataset.-----");
        graphtest(INPUT_FILENAME_CAHEPPH);
    }

    #[test]
    #[ignore]
    fn copapers_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the copapers dataset.-----");
        graphtest(INPUT_FILENAME_COPAPERS);
    }

    #[test]
    #[ignore]
    fn gupta2_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the gupta2 dataset.-----");
        graphtest(INPUT_FILENAME_GUPTA2);
    }

    #[test]
    #[ignore]
    fn gupta3_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the gupta3 dataset.-----");
        graphtest(INPUT_FILENAME_GUPTA3);
    }

    #[test]
    #[ignore]
    fn locbright_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the locbright dataset.-----");
        graphtest(INPUT_FILENAME_LOCBRIGHT);
    }

    #[test]
    #[ignore]
    fn mycielskian_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the mycielskian dataset.-----");
        graphtest(INPUT_FILENAME_MYCIELSKIAN);
    }

    #[test]
    #[ignore]
    fn pattern1_full_test(){
        println!("TEST:-----Verifying that sparsified graph retains the connectivity of the original graph, for the pattern1 dataset.-----");
        graphtest(INPUT_FILENAME_PATTERN1);
    }

    fn sketch_sparsification_rate_test(input_filename: &str) {
        let mut parameters = SparsifierParameters::new_default(true);
        parameters.jl_factor = 4.0;
        let test = true;
        let stream = InputStream::new(input_filename, "");
        let sparsifier: Sparsifier<i32> = stream.run_stream(&parameters, test);

        let original_graph = stream.get_input_graph();
        let sparsified_graph = sparsifier.to_petgraph();
        let original_edges = original_graph.edge_count();
        let sparsified_edges = sparsified_graph.edge_count();
        let uniform_sketch_ratio = (sparsified_edges as f64 / original_edges as f64);

        let original_ccs = connected_components(&original_graph);
        let sparsified_ccs = connected_components(&sparsified_graph);
        assert_eq!(original_ccs, sparsified_ccs);

        let mut parameters = SparsifierParameters::new_default(true);
        parameters.jl_factor = 4.0;
        parameters.sketch_uniform = false;
        let stream = InputStream::new(input_filename, "");
        let sparsifier: Sparsifier<i32> = stream.run_stream(&parameters, test);
        let sparsified_graph = sparsifier.to_petgraph();
        let sparsified_edges = sparsified_graph.edge_count();
        let discrete_sketch_ratio = (sparsified_edges as f64 / original_edges as f64);
        let sparsified_ccs = connected_components(&sparsified_graph);
        assert_eq!(original_ccs, sparsified_ccs);

        println!("file {}, \n
            uniform sketch sparsification rate is {}\n
            discrete sketch sparsification rate is {}", 
            input_filename, uniform_sketch_ratio, discrete_sketch_ratio);
    }

    #[test]
    #[ignore]
    fn virus_sparsification_rate_test(){
        println!("TEST:-----Measuring sparsification rate of uniform and discrete sketch approaches, for the virus dataset.-----");
        sketch_sparsification_rate_test(INPUT_FILENAME_VIRUS);
    }

    #[test]
    //#[ignore]
    fn mouse_sparsification_rate_test(){
        println!("TEST:-----Measuring sparsification rate of uniform and discrete sketch approaches, for the mouse dataset.-----");
        sketch_sparsification_rate_test(INPUT_FILENAME_MOUSE);
    }

    #[test]
    #[ignore]
    fn human1_sparsification_rate_test(){
        println!("TEST:-----Measuring sparsification rate of uniform and discrete sketch approaches, for the human1 dataset.-----");
        sketch_sparsification_rate_test(INPUT_FILENAME_HUMAN1);
    }

    #[test]
    #[ignore]
    fn human2_sparsification_rate_test(){
        println!("TEST:-----Measuring sparsification rate of uniform and discrete sketch approaches, for the human2 dataset.-----");
        sketch_sparsification_rate_test(INPUT_FILENAME_HUMAN2);
    }

    // new test: vary parameters (epsilon, jl factor, others?) and ensure ccs stay the same. probably for smaller dataset.
    // write test establishing which sketch type is acceptable for various datasets. (seems like the uniform approach allows more aggressive sparsification).
}

