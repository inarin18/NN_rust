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

pub fn create_layers(layer_sizes: Vec<usize>) -> Vec<Layer> {
    let mut layers: Vec<Layer> = Vec::new();
    for i in 0..layer_sizes.len() - 1 {
        let layer = Layer::new(format!("layer_{}", i), layer_sizes[i], layer_sizes[i+1]);
        layers.push(layer);
    }
    layers
}