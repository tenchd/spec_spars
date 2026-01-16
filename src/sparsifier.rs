use std::ops::Add;
use sprs::{CsMatI, CsMatBase, TriMatBase, TriMatI, CsMatViewI};
use std::sync::mpsc;
use std::thread;
use std::hash::{Hash,Hasher};
use std::ops::Mul;

use rand::Rng;
use approx::AbsDiffEq;
use fasthash::FastHasher;
use fasthash::murmur2::Hasher64_x64 as Murmur2Hasher;
use ndarray::{Array1,Array2,s};
use num_traits::cast;

use crate::ffi::{self, FlattenedVec};
use crate::utils::{BenchmarkPoint, Benchmarker, CustomIndex, CustomIndex::from_int, CustomValue};

#[derive(Clone)]
pub struct Triplet<IndexType: CustomIndex>{
    pub num_nodes: IndexType, 
    pub col_indices: Vec<IndexType>,
    pub row_indices: Vec<IndexType>,
    pub diagonal: Vec<f64>,
    pub values: Vec<f64>
}
impl<IndexType: CustomIndex> Triplet <IndexType> {
    // default constructor that just returns three empty vectors.
    pub fn new(num_nodes: IndexType) -> Triplet <IndexType> {
        let col_indices: Vec<IndexType> = vec![];
        let row_indices: Vec<IndexType> = vec![];
        let diagonal: Vec<f64> = vec![0.0; num_nodes.index().try_into().unwrap()];
        let values: Vec<f64> = vec![];
        Triplet { 
            num_nodes: num_nodes, 
            col_indices: col_indices, 
            row_indices: row_indices, 
            diagonal: diagonal,
            values: values
        }
    }

    // inserts an edge with a weight/value into the triplet format. keeps track of diagonal too.
    pub fn insert(&mut self, v1: IndexType, v2: IndexType, value: f64) {
        // make sure node IDs are valid (0 indexed)
        assert!(v1 < self.num_nodes);
        assert!(v2 < self.num_nodes);

        //input files may have diagonals; we should skip them
        // this is a kludge for reading in mtx files. later this logic should be moved elsewhere.
        if v1 != v2 {
            // insert -value into v1,v2
            self.row_indices.push(v1);
            self.col_indices.push(v2);
            self.values.push(value*-1.0);

            // insert -value into v2,v1
            self.row_indices.push(v2);
            self.col_indices.push(v1);
            self.values.push(value*-1.0);

            // add 1 to diagonal entries v1,v1 and v2,v2
            self.diagonal[v1.index()] += value;
            self.diagonal[v2.index()] += value;
        }
    }

    pub fn process_diagonal(&mut self) {
        // add diagonal entries to triplet format
        for (index, value) in self.diagonal.iter().enumerate() {
            //let new_index = index as i32;
            self.row_indices.push(from_int(index));
            self.col_indices.push(from_int(index));
            self.values.push(*value);
        }
    }

    pub fn to_csc(self) -> CsMatI::<f64, IndexType> {
        let trip_form: TriMatI<f64, IndexType>  = TriMatI::<f64, IndexType>::from_triplets((self.num_nodes.index(), self.num_nodes.index()), self.row_indices, self.col_indices, self.values);
        let csc_form: CsMatI<f64, IndexType> = trip_form.to_csc();
        csc_form
    }

    // converts triplet into signed edge-vertex incidence matrix to be JL sketched
    pub fn to_edge_vertex_incidence_matrix(&self) -> CsMatI<f64, IndexType> {
        //we grab every other entry in the triplet vectors because each edge (i,j) is inserted as (i,j) and then also immediately as (j,i)
        let mut evim_triplets = Triplet::new(self.num_nodes);
        assert!(self.col_indices.len()%2 == 0);
        let end: usize = self.col_indices.len()/2;
        for i in 0..end {
            let index = 2*i as usize;
            evim_triplets.col_indices.push(from_int(i));
            evim_triplets.row_indices.push(self.col_indices[index]);
            evim_triplets.values.push(-1.0 * self.values[index]);

            evim_triplets.col_indices.push(from_int(i));
            evim_triplets.row_indices.push(self.row_indices[index]);
            evim_triplets.values.push(self.values[index]);
        }
        let evim_trip_form: TriMatBase<Vec<IndexType>, Vec<f64>>  = TriMatI::<f64, IndexType>::from_triplets((evim_triplets.num_nodes.index(), end), evim_triplets.row_indices, evim_triplets.col_indices, evim_triplets.values);
        let evim_csc_form: CsMatBase<f64, IndexType, Vec<IndexType>, Vec<IndexType>, Vec<f64>, _> = evim_trip_form.to_csc();
        evim_csc_form
    }

    pub fn delete_state(&mut self) {
        self.col_indices = vec![];
        self.row_indices = vec![];
        self.diagonal = vec![0.0; self.num_nodes.index()];
        self.values = vec![];
    }

    #[allow(dead_code)]
    pub fn display(&self) {
        println!("triplet values:");
        for value in &self.col_indices {
            print!("{}, ", value);
        }
        println!("");
        for value in &self.row_indices {
            print!("{}, ", value);
        }
        println!("");
        for value in &self.values {
            print!("{}, ", value);
        }
        println!("");
        for value in &self.diagonal {
            print!("{}, ", value);
        }
        println!("");
    }
}

pub struct Sparsifier<IndexType: CustomIndex>{
    pub num_nodes: IndexType,     // number of nodes in input graph. we need to know this at construction time.
    pub new_entries: Triplet<IndexType>,  //stores values that haven't been sparsified yet
    pub current_laplacian: CsMatI<f64, IndexType>,   //stores all values that survive sparsification
    pub threshold: IndexType,     //sparsify if size(new_entries) + size(current_laplacian) > threshold. computed from num_nodes, epsilon, and constants
                              // set to be row_constant*beta*nodesize in line 3(b) of alg pseudocode
    pub epsilon: f64,     //epsilon controls space (aggressivenes of sampling) and approximation factor guarantee.
    pub beta_constant: IndexType,    // set to be 200 in line 1 of alg pseudocode, probably can be far smaller
    pub row_constant: IndexType,    // set to be 20 in line 3(b) of alg pseudocode, probably can be far smaller
    pub beta: IndexType,     // parameter defined in line 1 of alg pseudocode
    pub verbose: bool,    //when true, prints a bunch of debugging info
    pub jl_factor: f64, //constant factor for jl sketch matrix
    pub jl_dim: IndexType, 
    pub seed: u64,   //random seed for hashed jl sketch matrix
    pub benchmarker: Benchmarker,  //when true, measure time it takes to do operations
}

impl<IndexType: CustomIndex> Sparsifier<IndexType> {
    pub fn new(num_nodes: IndexType, epsilon: f64, beta_constant: IndexType, row_constant: IndexType, verbose: bool, jl_factor: f64, seed: u64, benchmarker: Benchmarker) -> Sparsifier<IndexType> {
        // as per line 1
        let beta = from_int((epsilon.powf(-2.0) * (beta_constant.index() as f64) * (num_nodes.index() as f64).log(2.0)).round() as usize);
        // as per 3(b) condition
        let threshold = num_nodes * beta * row_constant;
        // initialize empty new elements triplet vectors
        let new_entries: Triplet<IndexType> = Triplet::new(num_nodes);
        // initialize empty sparse matrix for the laplacian
        // it needs to have num_nodes+1 rows and cols actually
        let current_laplacian: CsMatI<f64, IndexType> = CsMatI::<f64, IndexType>::zero((num_nodes.index(), num_nodes.index()));     
        let jl_dim = from_int(((num_nodes.index() as f64).log2() *jl_factor).ceil() as usize);

        Sparsifier{
            num_nodes: num_nodes,
            new_entries: new_entries,
            current_laplacian: current_laplacian,
            threshold: threshold,
            epsilon: epsilon,
            beta_constant: beta_constant,
            row_constant: row_constant,
            beta: beta,
            verbose: verbose,
            jl_factor: jl_factor,
            jl_dim: jl_dim,
            seed: seed,
            benchmarker: benchmarker,
        }
    }

    // returns # of edges in the entire sparsifier (including the new edges in triplet form and old edges in sparse matrix form)
    // note that currently it overcounts for laplacian since it also counts diagonals. maybe change this later?
    #[allow(dead_code)]
    pub fn size_of(&self) -> IndexType {
        IndexType::from_int(self.new_entries.col_indices.len())  // total entries in triplet form
        + 
        IndexType::from_int(self.current_laplacian.nnz())  // total nonzeros in laplacian
        
    }

    // returns number of edges represented in current laplacian, not counting diagonal entries.
    pub fn num_edges(&self) -> usize {
        let diag = self.current_laplacian.diag();
        let diag_nnz = diag.nnz();
        let num_nnz = (self.current_laplacian.nnz() - diag_nnz) / 2;
        // janky check for integer rounding. i hang my head in shame
        assert_eq!(num_nnz*2, (self.current_laplacian.nnz()-diag_nnz));
        num_nnz
    }

    // inserts an edge into the sparsifier. if this makes the size of the sparsifier cross the threshold, trigger sparsification.
    pub fn insert(&mut self, v1: IndexType, v2: IndexType, value: f64) {
        // insert -1 into v1,v2 and v2,v1. add 1 to v1,v1 and v2,v2
        // problem: duplicate values in diagonals for triplets. 
        // am i assuming that each edge appears at most once in the stream? if that's violated, i could have duplicate entries in the triplets
        
        // make sure node IDs are valid (0 indexed)
        assert!(v1 < self.num_nodes);
        assert!(v2 < self.num_nodes);

        // ignore diagonal entries, upper triangular entries, and entries with 0 value
        if v1 > v2 && value != 0.0 {
            self.new_entries.insert(v1, v2, value);
            //println!("inserting ({}, {}) with value {}", v1, v2, value);
        }

        //TODO: if it's too big, trigger sparsification step
    }

        //maps hash function output to {-1,1} evenly
    fn transform(&self, input: i64) -> i64 {
        let result = ((input >> 31) * 2 - 1 ) as i64;
        result
    }

    pub fn hash_with_inputs(&self, input1: u64, input2: u64) -> i64 {
    let mut checkhash = Murmur2Hasher::with_seed(self.seed);
        input1.hash(&mut checkhash);
        input2.hash(&mut checkhash);
        let result = checkhash.finish() as u32;
        //println!("{}",result);
        self.transform(result as i64)
    }

    // used to completely fill a jl sketch matrix with random values from a hash function. only used for correctness testing
    pub fn populate_matrix(&self, input: &mut Array2<f64>) {
        let rows = input.dim().0;
        let cols = input.dim().1;
        let scaling_factor = (self.jl_dim.index() as f64).sqrt();
        for i in 0..rows {
            for j in 0..cols {
                input[[i,j]] += (self.hash_with_inputs(i as u64, j as u64) as f64) / scaling_factor;
            }
        }
    }

    // used to compute a single row of a jl sketch matrix. entries should match that of the corresponding row in the matrix returned
    // by populate_matrix() above
    pub fn populate_row<ValueType: CustomValue>(&self, input: &mut Array1<ValueType>, row: IndexType, col_start: IndexType, col_end: IndexType){
        let num_cols = (col_end - col_start).index();
        let scaling_factor = (self.jl_dim.index() as f64).sqrt();
        for col in 0..num_cols {
            let actual_col = col+col_start.index(); //have to hash actual column value which should be col+col_start
            input[[col]] = cast::<f64, ValueType>((self.hash_with_inputs(row.index() as u64, actual_col as u64) as f64) / scaling_factor).unwrap(); 
        }
    }

    // computes the jl sketch multiplication by creating the entire factor matrix and performing the mult with the sprs crate multiplication function.
    // not scalable; used as a reference implementation for correctness.
    #[allow(dead_code)]
    pub fn jl_sketch_sparse(&self, og_matrix: &CsMatI<f64, IndexType>) -> Array2<f64> {
        let og_rows = og_matrix.rows();
        let og_cols = og_matrix.cols();
        let mut sketch_matrix: Array2<f64> = Array2::zeros((og_cols,self.jl_dim.index()));
        if self.verbose {println!("EVIM has {} rows and {} cols, jl sketch matrix has {} rows and {} cols", og_rows, og_cols, og_cols, self.jl_dim);}
        self.populate_matrix(&mut sketch_matrix);
        if self.verbose {println!("populated sketch matrix");}
        let result = og_matrix.mul(&sketch_matrix);
        if self.verbose {println!("performed multiplication");}
        result
    }

    // this function JL sketches a sparse encoding of the input matrix and outputs in a sparse format as well. 
    // it doesn't do blocked operations though, so it's still not scalable because it represents the entire
    // dense sketch matrix at all times.
    #[allow(dead_code)]
    pub fn jl_sketch_sparse_flat(&self, og_matrix: &CsMatI<f64, IndexType>) -> ffi::FlattenedVec {
        let result = self.jl_sketch_sparse(og_matrix);
        ffi::FlattenedVec::new(&result)
    }

    // computes the jl sketch product of a block of the overall EVIM matrix. a single thread computes this.
    pub fn jl_sketch_colwise(&self, og_matrix: &CsMatViewI<f64, IndexType>, block_number: usize) -> Array2<f64> {
        let og_rows = og_matrix.rows();
        let mut output_block: Array2<f64> = Array2::zeros([og_rows, self.jl_dim.index()]);
        for (col_ind, col_vec) in og_matrix.outer_iterator().enumerate() {
            // adjust col ind because this is a subblock of the total input matrix
            let true_col_ind = block_number*og_rows + col_ind;
            assert!(col_vec.nnz() == 2);
            let mut jl_sketch_row: Array1<f64> = Array1::zeros(self.jl_dim.index());
            self.populate_row(&mut jl_sketch_row, from_int(true_col_ind), from_int(0), self.jl_dim);
            for (row_ind, &value) in col_vec.iter() {
                let product_row: Array1<f64> = jl_sketch_row.clone() * value;
                let mut row_view = output_block.slice_mut(s![row_ind, ..]);
                row_view += &product_row;
            }
        }
        output_block
    }

    // performs the jl sketch multiplication by breaking up the EVIM into blocks of columns, and assigning the multiplication for each column to a thread.
    pub fn jl_sketch_colwise_batch(&self, og_matrix: &CsMatI<f64, IndexType>, result_matrix: &mut CsMatI<f64, IndexType>) {
        let og_rows = og_matrix.rows();
        
        let (tx, rx) = mpsc::channel();
        thread::scope(|s| {
            for (block_index, block) in og_matrix.outer_block_iter(og_rows).enumerate() {
                let clone = tx.clone();
                s.spawn(move || {
                    let output_block = self.jl_sketch_colwise(&block, block_index);
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
    pub fn jl_sketch_colwise_flat(&self, og_matrix: &CsMatI<f64, IndexType>) -> ffi::FlattenedVec {
        let og_rows = og_matrix.rows();
        let jl_dim = ((og_rows as f64).log2() *self.jl_factor).ceil() as usize;
        let mut result_matrix: CsMatI<f64, IndexType> = CsMatI::zero((og_rows, jl_dim)).into_csc();
        self.jl_sketch_colwise_batch(&og_matrix, &mut result_matrix);
        println!("performed jl sketch multiplication");
        ffi::FlattenedVec::new(&result_matrix.to_dense())
    }


    // takes the solution matrix and computes an approximate effective resistance for each edge in the laplacian.
    fn compute_diff_norms(&self, length: usize, solution: &ffi::FlattenedVec) -> Vec<f64>{
        let solution_cols = solution.num_cols;
        let mut diff_norms = vec![0.0; length];
        let solution_array = solution.to_array2();
        let mut probs: Vec<f64> = vec![1.0; length];
        //loop through lower diagonal entries
        let mut nonzero_counter = 0;
        for (value, (row, col)) in self.current_laplacian.iter() {
            if row < col {
                //for each edge u,v, compute l2 norm of dot products with columns of solution matrix
                for i in 0..solution_cols {
                    diff_norms[nonzero_counter] += (solution_array[[row.index(), i as usize]] - solution_array[[col.index(), i as usize]]).powi(2);
                    assert!(diff_norms[nonzero_counter] >= 0.0);
                }
                diff_norms[nonzero_counter] = diff_norms[nonzero_counter].sqrt();
                //compute probs from diff norm: multiply by value to get lev score, then multiply by beta, then bound at 1
                probs[nonzero_counter] *=  (self.beta.index() as f64 * -1.0 * value * diff_norms[nonzero_counter]/(solution_cols as f64)).min(1.0);
                assert!(probs[nonzero_counter] >= 0.0, "negative prob. nonzero_counter = {}, prob = {}, diff norm = {}, value = {}", nonzero_counter, probs[nonzero_counter], diff_norms[nonzero_counter], value);
                assert!(probs[nonzero_counter] <= 1.0, "prob greater than 1. nonzero_counter = {}, prob = {}, diff norm = {}, value = {}", nonzero_counter, probs[nonzero_counter], diff_norms[nonzero_counter], value);
                nonzero_counter += 1;
            }
        }
        return probs;
    }

    // returns probabilities for all off-diagonal nonzero entries in laplacian. placeholder for now
    pub fn get_probs(&mut self, length: usize, sketch_cols: FlattenedVec) -> Vec<f64> {

        let col_ptrs: Vec<IndexType> = self.current_laplacian.indptr().as_slice().unwrap().to_vec();
        let row_indices: Vec<IndexType> = self.current_laplacian.indices().to_vec();
        let values: Vec<f64> = self.current_laplacian.data().to_vec();

        let solution = ffi::run_solve_lap(sketch_cols, crate::utils::convert_indices_to_i32(&col_ptrs), crate::utils::convert_indices_to_i32(&row_indices), values, self.num_nodes.as_i32(), self.verbose);
        if self.benchmarker.is_active(){
            self.benchmarker.set_time(BenchmarkPoint::SolvesComplete);
        }

        let probs = self.compute_diff_norms(length, &solution);
        if self.benchmarker.is_active(){
            self.benchmarker.set_time(BenchmarkPoint::DiffNormsComplete);
        }
        return probs;
    }

    pub fn flip_coins(length: usize) -> Vec<f64> {
        let mut rng = rand::thread_rng();
        //let coins = vec![rng.gen_range(0.0..1.0); length];
        let mut coins = vec![0.0; length];
        for i in 0..length {
            coins[i] += rng.gen_range(0.0..1.0);
        }
        coins
    }
    //TODO: move sampling and reweighting logic into this function
    pub fn sample_and_reweight(&mut self, probs: Vec<f64>) -> Triplet<IndexType> {
        let coins = Self::flip_coins(probs.len());

        //encodes whether each edge survives sampling or not. True means it is sampled, False means it's not sampled
        let outcomes: Vec<bool> =  probs.clone().into_iter().zip(coins.into_iter()).map(|(p, c)| c < p).collect();

        //println!("{:?}", self.current_laplacian.to_dense());
        // probably move below code into sample and reweight funtion for testing purposes
        let mut reweightings: Triplet<IndexType> = Triplet::new(self.num_nodes);

        let mut counter = 0;
        //let mut deletion_counter = 0;
//        let mut first_deletion = true;
        for (value, (row, col)) in self.current_laplacian.iter() {
            if row > col {
                // actual value is the negative of what's in the off-diagonal. flip the sign so the following code is easier to read.
                // actually this turned out to be more confusing. it's correct currently, but I should go back and carefully undo this.
                let true_value = *value* -1.0;
                let is_sampled = outcomes[counter];
                let prob = probs[counter];
                if is_sampled {
                    // should be bigger than true value because 0 < prob < 1
                    let target_value = true_value/prob.sqrt();
                    let additive_change = true_value - target_value;
                    assert!(additive_change <= 0.0, "counter = {}, prob = {}, true_value = {}, target value = {}, additive change = {}", counter, prob, true_value, target_value, additive_change);
                    //println!("{},{} stays, target value is {} so applying addition {} to existing value {}", row, col, target_value, additive_change, true_value);
                    reweightings.insert(row, col, additive_change*-1.0);
                }
                else {
                    // else the edge wasn't sampled so delete it with an "insertion" with opposite value, cancelling it out.
                    //println!("{},{} is deleted, so apply addition {} to existing value {}", row, col, -1.0*true_value, true_value);
                    let additive_change = true_value*-1.0;
                    reweightings.insert(row, col, additive_change);
                    //assert!(additive_change == *value);
                    // if deletion_counter == 0 {
                    //     println!("deletion. row = {}, col = {}, value = {}, true value = {}, is_sampled = {}, prob = {}, additive change = {}", row, col, *value, true_value, is_sampled, prob, additive_change);
                    // }
                    //deletion_counter += 1;
                }


                counter +=1;
            }
        }
        reweightings
    }

    pub fn form_laplacian(&mut self, check: bool) {

        // apply diagonals to new triplet entries
        self.new_entries.process_diagonal();
        // get the new entries in csc format
        // improve this later; currently it clones the triplet object which uses extra memory
        let new_stuff = self.new_entries.clone().to_csc();
        // clear the new entries from the triplet representation
        self.new_entries.delete_state();
        // add the new entries to the laplacian
        self.current_laplacian = self.current_laplacian.add(&new_stuff);

        if check {println!("checking diagonal after populating laplacian");
        self.check_diagonal();}
    }

    pub fn sparsify(&mut self, check: bool) {
        // compute evim format of new triplet entries (no diagonal)
        if self.benchmarker.is_active(){
            self.benchmarker.start();
            self.benchmarker.set_time(BenchmarkPoint::Initialize);
        }
        let evim = &self.new_entries.to_edge_vertex_incidence_matrix();
        // println!("signed edge-vertex incidence matrix has {} rows and {} cols", evim.rows(), evim.cols());
        if self.benchmarker.is_active(){
            self.benchmarker.set_time(BenchmarkPoint::EvimComplete);
        }
        // then compute JL sketch of it
        let sketch_cols: ffi::FlattenedVec = self.jl_sketch_colwise_flat(&evim);
        //let sketch_cols: ffi::FlattenedVec = jl_sketch_sparse_flat(evim, self.jl_factor, self.seed, display);
        if self.benchmarker.is_active(){
            self.benchmarker.set_time(BenchmarkPoint::JlSketchComplete);
        }
        
        self.form_laplacian(check);

        let num_nnz = self.num_edges();
        // get probabilities for each edge
        let mut probs = vec![];
        
        probs = self.get_probs(num_nnz, sketch_cols);
        
        let mut reweightings: Triplet<IndexType> = self.sample_and_reweight(probs);
        
        if self.benchmarker.is_active(){
            self.benchmarker.set_time(BenchmarkPoint::ReweightingsComplete);
        }
        // make sure diagonal is correct after reweightings
        reweightings.process_diagonal();
        let csc_reweightings = reweightings.to_csc();

        self.current_laplacian = self.current_laplacian.add(&csc_reweightings);

        //println!("total number of deletions should be: {}", deletion_counter);

        if check {println!("checking diagonal after sampling");
        self.check_diagonal();}

        if self.benchmarker.is_active(){
            self.benchmarker.set_time(BenchmarkPoint::End);
            self.benchmarker.display_durations();
        }

    }


    #[allow(dead_code)]
    pub fn sparse_display(&self) {
        println!("laplacian: ");
        for (value, (row, col)) in self.current_laplacian.iter() {
            print!("({}, {}) has value {} ", row, col, value);
        }
        println!("");
    }

    // see if there's a library function to make summing cols more efficient later.
    // also find a way to write efficient code to sum rows. for now, maybe just do em as you do cols?
    pub fn check_diagonal(&self) {
        for (col_index, col_vector) in self.current_laplacian.outer_iterator().enumerate() {
            let mut sum: f64 = 0.0;
            for (_, value) in col_vector.iter() {
                sum += (value).as_f64();
            }
            // check each column is sum 0. floating point error means you'll be a little off
            assert!(sum.abs_diff_eq(&0.0, 1e-10), "column where we messed up: {}. column sum: {}.", col_index, sum);
        }
        assert!(sprs::is_symmetric(&self.current_laplacian));
        //println!("laplacian matrix: each column sums to 0. matrix is symmetric. format check PASSED.");
    }
}