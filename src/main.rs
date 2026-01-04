mod layer;
mod model;

use layer::Layer;
use layer::create_layers;
use layer::print_layers;

use model::Model;
use model::create_model;

fn main() {

    let seed: u64 = 42;
    println!("Start Program");
    
    // MNIST の画像サイズ
    let height: usize = 28;
    let width: usize = 28;
    println!("(HEIGHT, WIDTH) = ({}, {})", height, width);

    // レイヤーをまとめる配列
    let layers: Vec<Layer> = create_layers(vec![height * width, 1024, 1024, 10]);
    
    // & を付けることで layers の所有権を借用できる
    print_layers(&layers);

    // モデルを作成してビルド
    let mut model: Model = create_model(layers);
    model.build();
    println!("model built");
}