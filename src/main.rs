mod layers;
mod model;
mod activation;
mod data;
mod losses;
mod optimizers;
mod trainer;

use layers::base_layer::AbstractLayerTrait;
use layers::utils::create_layers;
use layers::utils::print_layers;

use model::Model;
use model::create_model;

use data::DataSet;

use losses::base_loss::AbstractLossFunctionTrait;
use losses::cross_entropy_loss::CrossEntropyLoss;

use optimizers::base_optimizer::AbstractOptimizerTrait;
use optimizers::sgd::{Sgd, SgdParams};

use trainer::Trainer;

fn main() {

    println!("┌──────────────────────────────┐");
    println!("│ Start Program                │");
    println!("└──────────────────────────────┘");
    
    // MNIST の画像サイズ
    let height: usize = 28;
    let width: usize = 28;

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

    // データを読み込む
    let mnist_data: DataSet = DataSet::load_from_binary("data/mnist.bin").unwrap();
    println!("data: {:?}", mnist_data.get_image(0).expect("Failed to get image").len());
    println!("data: {:?}", mnist_data.get_label(0).expect("Failed to get label"));

    // アスキーアートで画像を表示
    mnist_data.display_image(0);

    // 損失関数を作成
    let mut loss_function: CrossEntropyLoss = CrossEntropyLoss::new("cross_entropy_loss".to_string());

    let index: usize = 0;
    let input: Vec<f32> = mnist_data.get_image(index).expect("Failed to get image").to_vec();
    let label: u8 = mnist_data.get_label(index).expect("Failed to get label");
    let mut true_labels = vec![0.0; 10];
    true_labels[label as usize] = 1.0;
    
    // モデルを forward する
    let output: Vec<f32> = model.forward(&input);

    // 損失関数を forward する
    let loss: f32 = loss_function.forward(&true_labels, &output);
    println!("loss: {:?}", loss);

    // 損失関数を backward する
    let loss_grad: Vec<f32> = loss_function.backward(&true_labels, &output);
    println!("loss_grad: {:?}", loss_grad);

    // モデルを backward する（勾配を計算して保存しただけ）
    model.backward(&loss_grad);

    // オプティマイザーを作成
    let mut optimizer: Sgd = Sgd::new("sgd".to_string());
    optimizer.build(
        SgdParams::new()
        .learning_rate(0.01)
        .verbose(true)
    );

    let weights_before: Vec<f32> = model.layers[1].w().clone();

    // オプティマイザーを update することでモデルのパラメータを更新
    optimizer.update(&mut model);

    // 更新できたか比較して確認
    let weights_after: Vec<f32> = model.layers[1].w().clone();
    let weight_diff: f32 = weights_before.iter()
        .zip(weights_after.iter())
        .map(|(b, a)| a - b)
        .sum::<f32>()
        .abs();
    println!("weight difference: {:?}", weight_diff);

    // Trainer を作成
    let (train_dataset, test_dataset) = mnist_data.split_dataset(0.01);
    let mut trainer: Trainer<Sgd, CrossEntropyLoss> = Trainer::new(
        model,
        optimizer,
        loss_function,
        train_dataset,
        test_dataset,
        10,
        true
    );
    trainer.run();
}