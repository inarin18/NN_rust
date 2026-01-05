use crate::activation::{relu, sigmoid, identity};
use crate::layers::base_layer::{AbstractLayer, AbstractLayerTrait};


#[derive(Debug)]
pub struct FcLayer {
    base: AbstractLayer,
    pub activation_fn: fn(f32) -> f32,
}

impl FcLayer {
    pub fn new(name: String, i_size: usize, o_size: usize, activation_type: String) -> Self {
        Self { 
            base: AbstractLayer::new(name, i_size, o_size, activation_type),
            activation_fn: identity,
        }
    }
}

impl AbstractLayerTrait for FcLayer {
    fn build(&mut self) {
        self.base.w = vec![0.0; self.base.i_size * self.base.o_size];
        self.base.b = vec![0.0; self.base.o_size];
        match self.base.activation_type.as_str() {
            "relu" => self.activation_fn = relu,
            "sigmoid" => self.activation_fn = sigmoid,
            _ => self.activation_fn = identity,
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        // pass
        x.to_vec()
    }

    fn name(&self) -> &str {
        &self.base.name
    }

    fn i_size(&self) -> usize {
        self.base.i_size
    }

    fn o_size(&self) -> usize {
        self.base.o_size
    }

    fn w(&self) -> &Vec<f32> {
        &self.base.w
    }

    fn b(&self) -> &Vec<f32> {
        &self.base.b
    }

    fn activation_type(&self) -> &str {
        &self.base.activation_type
    }
}