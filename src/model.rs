use crate::layer::Layer;

pub struct Model {
    pub layers: Vec<Layer>,
}

impl Model {
    pub fn new(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn build(&mut self) {
        for layer in &mut self.layers {
            layer.build();
        }
    }

    // pub fn forward(&self, x: &Vec<f32>) -> Vec<f32> {
    //     let mut y = x.clone();
    //     for layer in &self.layers {
    //         y = layer.forward(y);
    //     }
    //     y
    // }
}

// instance method but not need to use?
pub fn create_model(layers: Vec<Layer>) -> Model {
    Model::new(layers)
}