mod layers;
mod model;
mod activation;
mod data;

use rand::Rng;

use layers::base_layer::AbstractLayerTrait;
use layers::utils::create_layers;
use layers::utils::print_layers;

use model::Model;
use model::create_model;

use data::MnistData;

fn main() {

    println!("Start Program");
    
    // MNIST の画像サイズ
    let height: usize = 28;
    let width: usize = 28;
    println!("(HEIGHT, WIDTH) = ({}, {})", height, width);

    // レイヤーをまとめる配列
    let use_softmax: bool = true;
    let layers: Vec<Box<dyn AbstractLayerTrait>> = create_layers(
        vec![height * width, 1024, 1024, 10], 
        "relu".to_string(), 
        use_softmax
    );
    
    // & を付けることで layers の所有権を借用できる
    print_layers(&layers);

    // モデルを作成してビルド
    let mut model: Model = create_model(layers);
    model.build();
    println!("model built");

    // ランダムな入力を生成
    let mut rng = rand::thread_rng();
    let input: Vec<f32> = (0..height * width)
        .map(|_| rng.gen_range(0..=255) as f32)
        .collect::<Vec<f32>>()
        .iter()
        .map(|x| x / 255.0)
        .collect();
    println!("input: {:?}", &input[..10]);

    // モデルを forward する
    let output: Vec<f32> = model.forward(&input.clone());
    println!("output: {:?}", &output);

    // データを読み込む
    let data: MnistData = MnistData::load_from_binary("data/mnist.bin").unwrap();
    println!("data: {:?}", data.get_image(0));
    println!("data: {:?}", data.get_label(0));
}