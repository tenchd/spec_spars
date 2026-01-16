use spec_spars::lap_test;

fn main() {
    let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/virus/virus.mtx";
    //let output_file_prefix = "virus_sparse";
    //let input_filename = "/global/u1/d/dtench/m1982/david/bulk_to_process/human_gene2/human_gene2.mtx";
    lap_test(input_filename);
}