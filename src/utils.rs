// this file stores boring functions for the Rust side of the sparsifier implementation. 
// boring means not related to ffi, not really part of the core sparsifier logic, etc.
use sprs::{CsMatI,TriMatI,indexing::SpIndex};
use sprs::io::{read_matrix_market, IoError};
use std::ops::Add;
use std::process::Command;
use std::time::{Instant, Duration};
use crate::ffi;
use std::path::Path;
use std::fs::File;
use std::io::{prelude::*, BufReader};
use csv::Writer;
use ndarray::{Array2, Axis};
use std::error::Error;

use crate::tests::make_random_matrix;


//Following traits are used for generic types for sparse matrix indices (implemented) and values (not yet implemented)
pub trait CustomValue: num_traits::float::Float + std::ops::AddAssign + Default + std::fmt::Debug + Clone + num_traits::Zero + std::fmt::Display + num_traits::cast::ToPrimitive{
    #[allow(dead_code)]
    fn from_float<U: num_traits::float::Float>(value: U) -> Self;
    fn as_f64(&self) -> f64;
}
impl<T> CustomValue for T where T: num_traits::float::Float + std::ops::AddAssign + Default + std::fmt::Debug + Clone + num_traits::Zero + std::fmt::Display + num_traits::cast::ToPrimitive {
    fn from_float<U: num_traits::float::Float>(value: U) -> Self {
        T::from(value).unwrap()
    }
    fn as_f64(&self) -> f64 {
        self.to_f64().unwrap()
    }
}

pub trait CustomIndex: SpIndex + std::range::Step + std::fmt::Display {
    fn from_int<U: num_traits::int::PrimInt>(value: U) -> Self;
    fn as_i32(&self) -> i32;
}
impl<T> CustomIndex for T where T: SpIndex + std::range::Step + std::fmt::Display {
    fn from_int<U: num_traits::int::PrimInt>(value: U) -> Self {
        T::from(value).unwrap()
    }
    fn as_i32(&self) -> i32 {
        self.to_i32().unwrap()
    }
}

pub fn convert_indices_to_i32<IndexType: CustomIndex>(input_vec: &Vec<IndexType>) -> Vec<i32>{
    input_vec.into_iter().map(|v| v.as_i32()).collect()
}

// fn convert_index_type_to_int<InitialIndexType: CustomIndex, DesiredIndexType: PrimInt>(vector: Vec<InitialIndexType>) -> Vec<DesiredIndexType> {
//     vector.into_iter().map(|v| v.index().try_into()).collect()
// }

// fn convert_value_type_to_float<InitialValueType: CustomValue, DesiredValueType: Float>(vector: Vec<InitialValueType>) -> Vec<DesiredValueType> {
//     vector.into_iter().map(|v| v.as_f64().try_into()).collect()
// }

// following code provides simple code for timing different parts of sparsification pipeline
pub enum BenchmarkPoint {
    Start,
    Initialize,
    EvimComplete,
    JlSketchComplete,
    SolvesComplete,
    DiffNormsComplete,
    ReweightingsComplete,
    End,
}

pub struct Benchmarker {
    pub active: bool,
    pub timer: Instant,
    //pub points: BenchmarkPoint,
    pub times: Vec<Option<Duration>>,
}

impl Benchmarker {
    pub fn new(benchmark: bool) -> Benchmarker {
        Benchmarker { 
            active: benchmark, 
            timer: Instant::now(), 
            //points: points,
            times: vec![None; 8],
        }
    }

    pub fn is_active(&self) -> bool {
        self.active
    }

    fn resolve(&self, point: BenchmarkPoint) -> usize {
        match point {
            BenchmarkPoint::Start => 0,
            BenchmarkPoint::Initialize => 1,
            BenchmarkPoint::EvimComplete => 2,
            BenchmarkPoint::JlSketchComplete=> 3,
            BenchmarkPoint::SolvesComplete=> 4,
            BenchmarkPoint::DiffNormsComplete=> 5,
            BenchmarkPoint::ReweightingsComplete=> 6,
            BenchmarkPoint::End => 7,
        }
    }

    pub fn start(&mut self) {
        assert!(self.is_active());
        self.times[0] = Some(self.timer.elapsed());
    }

    pub fn get_time(&self) -> Duration {
        assert!(self.is_active());
        //assert!(self.t_start.is_some());
        self.timer.elapsed() - self.times[0].expect("timer is not initialized")
    }

    pub fn set_time(&mut self, point: BenchmarkPoint) {
        let index = self.resolve(point);
        self.times[index] = Some(self.get_time());
    }

    fn compute_duration(&self, point1: BenchmarkPoint, point2: BenchmarkPoint) -> Duration {
        self.times[self.resolve(point1)].unwrap() - self.times[self.resolve(point2)].unwrap()
    }

    pub fn display_durations(&self){
        if self.active{
            println!("");
            println!("--------------- Runtime information: -----------------------");
            println!("time to compute EVIM: - - - - - - - - - - - -  {:.2} seconds", 
                self.compute_duration(BenchmarkPoint::EvimComplete, BenchmarkPoint::Initialize).as_secs_f64());
            println!("time to JL sketch: --------------------------- {:.2} seconds", 
                self.compute_duration(BenchmarkPoint::JlSketchComplete, BenchmarkPoint::EvimComplete).as_secs_f64());
            println!("time to solve: - - - - - - - - - - - - - - - - {:.2} seconds", 
                self.compute_duration(BenchmarkPoint::SolvesComplete, BenchmarkPoint::JlSketchComplete).as_secs_f64());
            println!("time to compute diff norms: ------------------ {:.2} seconds", 
                self.compute_duration(BenchmarkPoint::DiffNormsComplete, BenchmarkPoint::SolvesComplete).as_secs_f64());
            println!("time to reweight: - - - - - - - - - - - - - -  {:.2} seconds", 
                self.compute_duration(BenchmarkPoint::ReweightingsComplete, BenchmarkPoint::DiffNormsComplete).as_secs_f64());
            println!("------------------------------------------------------------");
            println!("");
        }
    }

}

pub fn read_mtx<IndexType: CustomIndex>(filename: &str) -> CsMatI<f64, IndexType>{
    println!("reading file {}", filename);
    let trip = read_matrix_market::<f64, IndexType, &str>(filename).unwrap();    
    let col_format = trip.to_csc::<IndexType>();
    return col_format;
}

#[allow(dead_code)]
pub fn read_sketch_from_mtx(filename: &str) -> ffi::FlattenedVec {
    let csc: CsMatI<f64, i32> = read_mtx(filename);
    let dense = csc.to_dense();
    let output = ffi::FlattenedVec::new(&dense);
    output
}

//BROKEN: function to read pattern .mtx files. needs to be fixed
#[allow(dead_code)]
pub fn load_pattern_as_csr<P>(path: P) -> Result<CsMatI<f64, i32>, std::io::Error>
where
    P: AsRef<Path>,
{
    // Read the file.  `read_matrix_market` returns a `TriMatI` **or**
    //     a crate‑specific `IoError`.  We map that error into a plain
    //     `std::io::Error` so the function can keep the `std::io::Result`
    //     signature that most callers expect.
    let triplet: TriMatI<f64, i32> =
        read_matrix_market(path).map_err(|e: IoError| std::io::Error::new(std::io::ErrorKind::Other, e))?;

    //  Convert the triplet (coordinate) representation to CSR.
    Ok(triplet.to_csr())
}

//make this generic later maybe
pub fn write_mtx<IndexType: CustomIndex>(filename: &str, matrix: &CsMatI<f64, IndexType>) {
        sprs::io::write_matrix_market(filename, matrix).ok();
}

fn flip_sign(value: &f64) -> f64 {
    value*-1.0
}

pub fn write_mtx_and_edgelist<IndexType: CustomIndex>(matrix: &CsMatI<f64, IndexType>, output_name: &str, flip: bool) {
    let output_prefix = "data/";
    let output_suffix_sparse = "_sparse";
    let output_suffix_mtx = ".mtx";
    let output_suffix_edgelist = ".edgelist";

    let output_mtx_full = output_prefix.to_owned() + &output_name + output_suffix_sparse + output_suffix_mtx;
    let output_edgelist_full = output_prefix.to_owned() + &output_name + output_suffix_sparse + output_suffix_edgelist;
    println!("writing to file {}", output_mtx_full);
    crate::utils::write_mtx(&output_mtx_full, matrix);

    let mut diagonal_trip = TriMatI::<f64, IndexType>::new((matrix.rows(), matrix.cols()));
    for (position, value) in matrix.diag().iter(){
        diagonal_trip.add_triplet(position, position, *value);
    }
    let mut diagonal_csc = diagonal_trip.to_csc();
    let final_trip = TriMatI::<f64, IndexType>::new((matrix.rows(), matrix.cols()));
    //let mut final_matrix = CsMatI::<f64, IndexType>::new((matrix.rows(), matrix.cols()), vec![], vec![], vec![]);
    let mut final_matrix = final_trip.to_csc();

    if flip {
        let matrix_flipped = matrix.map(&flip_sign);
        final_matrix = matrix_flipped.add(&diagonal_csc);
    }
    else {
        diagonal_csc = diagonal_csc.map(flip_sign);
        final_matrix = matrix.add(&diagonal_csc);
    }

    // println!("{:?}", matrix.to_dense());
    // println!("------------------------------------");
    // println!("{:?}", final_matrix.to_dense());

    crate::utils::write_mtx("temp.mtx", &final_matrix);
    let conversion_command = "sed '1,3d' temp.mtx > ".to_owned() + &output_edgelist_full;
    println!("converting mtx file to edgelist with the following command:");
    println!("{}", conversion_command);
    //Command::new("bash").arg("-c").arg("sed '1,3d' data/virus_sparse.mtx > data/virus_sparse.edgelist").output();
    Command::new("bash").arg("-c").arg(conversion_command).output();
    Command::new("bash").arg("-c").arg("rm temp.mtx").output();
}

#[allow(dead_code)]
pub fn l2_norm<IndexType: CustomIndex> (filename: &str) {
    let matrix: CsMatI<f64, IndexType> = read_mtx(filename);
    let mut sum = 0.0;
    for (value, (_row, _col)) in matrix.iter() {
        sum += value.powi(2);
    }
    let norm = sum.sqrt();
    println!("l2 norm of matrix: {}", norm);
}

#[allow(dead_code)]
// reads a jl_sketch vec of vecs from a file and returns it in flattened form.
pub fn read_vecs_from_file_flat(filename: &str) -> ffi::FlattenedVec {
    let file = File::open(filename).unwrap();
    let reader = BufReader::new(file);

    let mut jl_vec: Vec<f64> = vec![];
    let mut line_length: usize = 0; 
    let mut first: bool = true;
    let mut line_counter: usize = 0;

    for line in reader.lines() {
        line_counter += 1;
        let mut col: Vec<f64> = line.expect("uh oh").split(",")
                                        .map(|x| x.trim().parse::<f64>().unwrap())
                                        .collect();
        if first {
            line_length = col.len().try_into().unwrap();
            first = false;
        }
        let current_line_length= col.len().try_into().unwrap();
        assert_eq!(line_length, current_line_length);
        jl_vec.append(&mut col);
    }
    //println!("line length = {}, num_lines = {}", line_length, line_counter);

    //println!("dimensions of matrix: {} rows, {} cols", line_length, line_counter);
    let jl_cols_flat = ffi::FlattenedVec{vec: jl_vec, num_rows: line_counter, num_cols: line_length};
    jl_cols_flat
}

// writes dense array to csv for testing purposes. note: vibe coded.
pub fn write_f64_ndarray_to_csv<P>(array: &Array2<f64>, path: P) -> Result<(), Box<dyn Error>>
where
    P: AsRef<Path>,
{
    // Open (or create) the CSV file.
    let mut wtr = Writer::from_path(path)?;

    // Iterate over the outer axis (rows).
    for row in array.axis_iter(Axis(0)) {
        // `write_record` accepts any iterator of items that can be turned into `&[u8]`.
        // Mapping each `f64` to a `String` satisfies that bound.
        wtr.write_record(row.iter().map(|v| v.to_string()))?;
    }

    wtr.flush()?; // Ensure everything is flushed to disk.
    Ok(())
}

#[test]
// verifies that the above functions that handle reading/writing dense matrices into csv files write out and read back in the matrix correctly.
pub fn test_array_file(){
    let rows = 103;
    let cols = 91;
    let nnz = 9000;
    let csc = false;
    let repetitions = 10;
    for i in 0..repetitions {
        let test_matrix_sparse = make_random_matrix(rows, cols, nnz, csc, false);
        let test_matrix = test_matrix_sparse.to_dense();
        let temp_filename = "test_data/temp.csv";

        write_f64_ndarray_to_csv(&test_matrix, temp_filename);
        let recovered_matrix_flat = read_vecs_from_file_flat(temp_filename);
        let recovered_matrix = recovered_matrix_flat.to_array2();
        // println!("{:?}", test_matrix);
        // println!("-----------------------------------------------------------------------");
        // println!("-----------------------------------------------------------------------");
        // println!("-----------------------------------------------------------------------");
        // println!("{:?}", recovered_matrix);
        assert!(test_matrix.abs_diff_eq(&recovered_matrix, 0.0001));
    }
}

pub fn read_csv_as_vec<P: AsRef<Path>>(path: P) -> Result<Vec<f64>, Box<dyn Error>> {
    // Open the file and wrap it in a buffered reader.
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut line = String::new();

    // Read the first (and only) line.
    let bytes = reader.read_line(&mut line)?;
    if bytes == 0 {
        return Err("CSV file is empty".into());
    }

    // Ensure there is no second non‑empty line.
    let mut extra = String::new();
    if reader.read_line(&mut extra)? != 0 && !extra.trim().is_empty() {
        return Err("CSV file contains more than one line".into());
    }

    // Convert the comma‑separated fields into f64 values.
    let values = line
        .trim_end()               // Remove trailing newline / CRLF.
        .split(',')               // Split on commas.
        .map(str::trim)           // Trim whitespace around each field.
        .map(|s| s.parse::<f64>()) // Parse as f64.
        .collect::<Result<Vec<f64>, _>>()?; // Propagate parse errors.

    Ok(values)
}