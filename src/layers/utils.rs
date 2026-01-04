
use crate::layers::fc_layer::FcLayer;
use crate::layers::softmax_layer::SoftmaxLayer;
use crate::layers::base_layer::AbstractLayerTrait;

pub fn create_layers(layer_sizes: Vec<usize>, use_softmax: bool) -> Vec<Box<dyn AbstractLayerTrait>> {
    let mut layers: Vec<Box<dyn AbstractLayerTrait>> = Vec::new();
    for i in 0..layer_sizes.len() - 1 {
        let layer = Box::new(FcLayer::new(format!("layer_{}", i), layer_sizes[i], layer_sizes[i+1]));
        layers.push(layer);
    }
    if use_softmax {
        let softmax_layer = Box::new(SoftmaxLayer::new(format!("softmax_{}", layer_sizes.len() - 1), layer_sizes[layer_sizes.len() - 1], layer_sizes[layer_sizes.len() - 1]));
        layers.push(softmax_layer);
    }
    layers
}

pub fn print_layers(layers: &[Box<dyn AbstractLayerTrait>]) {
    for layer in layers {
        println!("-----------------------------");
        println!(" Layer Name = {}", layer.name());
        println!("     Input Size      : {}", layer.i_size());
        println!("     Output Size     : {}", layer.o_size());

        if layer.activation_type() != "softmax" {
            println!("     Weights Length  : {}", layer.w().len());
            println!("     Biases Length   : {}", layer.b().len());
        }
    }
    println!("-----------------------------");
}