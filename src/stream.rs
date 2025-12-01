use sprs::{CsMatI, DenseVector};
use crate::{read_mtx,Sparsifier};
use crate::utils::{Benchmarker, load_pattern_as_csr};
use std::process::Command;


pub struct InputStream {
    pub input_matrix: CsMatI<f64, i32>,
//    pub input_iterator: 
    pub num_nodes: usize,
//    pub num_edges: usize,
}

impl InputStream {
    // deal with diagonals?
    // if the graph is symmetric, de-symmetrize it ideally
    // how does the mtx reader handle symmetry?
    pub fn new(filename: &str) -> InputStream {
        let mut input = read_mtx(filename);
        //let mut input = load_pattern_as_csr(filename).expect("file read error");
        
        // zeroed diagonal entries remain explicitly represented using this format.
        // fix this later.
        let num_nodes = input.outer_dims();
        assert_eq!(input.outer_dims(), input.inner_dims());
        let mut diag_zeros = 0;
        for result in input.diag_iter_mut() {
            match result {
                Some(x) => *x = 0.0,
                none => diag_zeros += 1,
            }
        }

        println!("mtx file has {} nodes. there are {} zero entries on the diagonal originally.", num_nodes, diag_zeros);

        // for value in input.iter() {
        //     println!("{:?}", value);
        // }
        InputStream{
            input_matrix: input,
            num_nodes: num_nodes,
        }
    }

    pub fn run_stream(&self, epsilon: f64, beta_constant: i32, row_constant: i32, verbose: bool, jl_factor: f64, seed: u64, benchmark: bool, test: bool) -> Sparsifier {
        let mut benchmarker = Benchmarker::new(benchmark);
        let mut sparsifier = Sparsifier::new(self.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);

        for (value, (row, col)) in self.input_matrix.iter() {
            //assert!(*value >= 0.0);
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }

        // to check equivalence with original matrix, call sparsify with argument true and uncomment the check loop below
        sparsifier.sparsify(false, false);

        println!("checking diagonal final time");
        sparsifier.check_diagonal();

        let output_prefix = "data/virus_sparse";
        let output_suffix_mtx = ".mtx";
        let output_suffix_edgelist = ".edgelist";

        if (!test) {
            println!("writing to file.");
            let output_mtx = output_prefix.to_owned() + output_suffix_mtx;
            crate::utils::write_mtx(&output_mtx, &sparsifier.current_laplacian);
            println!("converting to edgelist format.");
            Command::new("bash").arg("-c").arg("sed '1,3d' data/virus_sparse.mtx > data/virus_sparse.edgelist").output();
        }

        sparsifier
    }

    // used for testing purposes
    pub fn produce_laplacian(&self) -> CsMatI<f64, i32>{
        let seed: u64 = 1;
        let jl_factor: f64 = 1.5;

        let epsilon = 0.5;
        let beta_constant = 4;
        let row_constant = 2;
        let verbose = false;
        let mut benchmarker = Benchmarker::new(false);
        let mut sparsifier = Sparsifier::new(self.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose, jl_factor, seed, benchmarker);

        for (value, (row, col)) in self.input_matrix.iter() {
            //assert!(*value >= 0.0);
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }

        sparsifier.form_laplacian();

        // now the sparsifier laplacian can be passed to c++ for interop testing.
        sparsifier.current_laplacian
    }
}