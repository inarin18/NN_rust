mod layer;

use layer::Layer;
use layer::create_layers;

fn main() {

    let seed: u64 = 42;
    println!("Start Program");
    
    // MNIST の画像サイズ
    let height: usize = 28;
    let width: usize = 28;
    println!("(HEIGHT, WIDTH) = ({}, {})", height, width);

    // レイヤーをまとめる配列
    let mut layers: Vec<Layer> = create_layers(vec![height * width, 1024, 1024, 10]);
    
    // & を付けることで layers の所有権を借用できる
    for layer in &layers {
        println!("layer: {:?}", layer);
    }

    // // フィールドを読み取る
    // println!("Layer name: {}", layer_0.name);
    // println!("Input size: {}, Output size: {}", layer_0.i_size, layer_0.o_size);
    // println!("Weights length: {}, Biases length: {}", layer_0.w.len(), layer_0.b.len());
}