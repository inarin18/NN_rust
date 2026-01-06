use crate::layers::base_layer::AbstractLayerTrait;

pub struct Model {
    pub layers: Vec<Box<dyn AbstractLayerTrait>>,
}

impl Model {
    pub fn new(layers: Vec<Box<dyn AbstractLayerTrait>>) -> Self {
        Self { layers }
    }

    pub fn build(&mut self) {
        for layer in &mut self.layers {
            layer.build();
        }
    }

    pub fn forward(&mut self, x: &Vec<f32>) -> Vec<f32> {
        let mut y: Vec<f32> = x.clone();
        for layer in &mut self.layers {
            y = layer.forward(&y);
        }
        y
    }

    pub fn backward(&mut self, loss_grad: &[f32]) {
        let mut grad = loss_grad.to_vec();
        
        // レイヤーを逆順に処理
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad);
        }
    }
}

// instance method but not need to use?
pub fn create_model(layers: Vec<Box<dyn AbstractLayerTrait>>) -> Model {
    Model::new(layers)
}