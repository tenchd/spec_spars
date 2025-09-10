use sprs::{CsMatI};
use crate::{read_mtx,Sparsifier};
use crate::utils::{load_pattern_as_csr};


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
                None => diag_zeros += 1,
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

    pub fn run_stream(&self, epsilon: f64, beta_constant: i32, row_constant: i32, verbose: bool, jl_factor: f64, seed: u64) {
        let mut sparsifier = Sparsifier::new(self.num_nodes.try_into().unwrap(), epsilon, beta_constant, row_constant, verbose, jl_factor, seed);

        for (value, (row, col)) in self.input_matrix.iter() {
            sparsifier.insert(row.try_into().unwrap(), col.try_into().unwrap(), *value);
        }

        // to check equivalence with original matrix, call sparsify with argument true and uncomment the check loop below
        sparsifier.sparsify(false, false);

        // for (value, (row, col)) in self.input_matrix.iter() {
        //     let check_value = sparsifier.current_laplacian.get(row as usize, col as usize).unwrap();
        //     if (row != col){
        //         assert!(*value == *check_value * -1.0, "not matching values row {} col {} original value {} lap value {}", row, col, *value, *check_value);
        //     }
        // }


        println!("checking diagonal final time");
        sparsifier.check_diagonal();

    }
}