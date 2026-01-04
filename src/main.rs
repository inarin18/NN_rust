mod layer;

use layer::Layer;

fn main() {
    println!("Start Program");
    
    let height: usize = 28;
    let width: usize = 28;
    
    println!("(HEIGHT, WIDTH) = ({}, {})", height, width);
    
    let i_size: usize = height * width;
    let o_size: usize = 1024;
    let layer_0: Layer = Layer::new("layer_0".to_string(), i_size, o_size);
    println!("layer_0: {:?}", layer_0);

    // フィールドを読み取る
    println!("Layer name: {}", layer_0.name);
    println!("Input size: {}, Output size: {}", layer_0.i_size, layer_0.o_size);
    println!("Weights length: {}, Biases length: {}", layer_0.w.len(), layer_0.b.len());
}