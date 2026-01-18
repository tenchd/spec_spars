// this file stores boring functions for the Rust side of the sparsifier implementation. 
// boring means not related to ffi, not really part of the core sparsifier logic, etc.
use sprs::{CsMatI,TriMatI,indexing::SpIndex};
use sprs::io::{read_matrix_market, IoError};
use std::time::{Instant, Duration};
use crate::ffi;
use std::path::Path;


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

pub fn read_mtx(filename: &str) -> CsMatI<f64, i32>{
    println!("reading file {}", filename);
    let trip = read_matrix_market::<f64, i32, &str>(filename).unwrap();    
    let col_format = trip.to_csc::<i32>();
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
pub fn write_mtx(filename: &str, matrix: &CsMatI<f64, i32>) {
        sprs::io::write_matrix_market(filename, matrix).ok();
}

#[allow(dead_code)]
pub fn l2_norm(filename: &str) {
    let matrix = read_mtx(filename);
    let mut sum = 0.0;
    for (value, (_row, _col)) in matrix.iter() {
        sum += value.powi(2);
    }
    let norm = sum.sqrt();
    println!("l2 norm of matrix: {}", norm);
}
