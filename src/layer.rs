#[derive(Debug)]
pub struct Layer {
    pub name: String,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub i_size: usize,
    pub o_size: usize
}

impl Layer {
    pub fn new(name: String, i_size: usize, o_size: usize) -> Self {
        Self{name, w: Vec::new(), b: Vec::new(), i_size, o_size}
    }
}