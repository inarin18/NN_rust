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
        Self{name, w: vec![0.0; i_size * o_size], b: vec![0.0; o_size], i_size, o_size}
    }

    // useless? 
    pub fn build(&mut self) {
        self.w = vec![0.0; self.i_size * self.o_size];
        self.b = vec![0.0; self.o_size];
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

pub fn print_layers(layers: &Vec<Layer>) {
    for layer in layers {
        println!("-----------------------------");
        println!(" Layer Name = {}", layer.name);
        println!("     Input Size      : {}", layer.i_size);
        println!("     Output Size     : {}", layer.o_size);
        println!("     Weights Length  : {}", layer.w.len());
        println!("     Biases Length   : {}", layer.b.len());
    }
    println!("-----------------------------");
}