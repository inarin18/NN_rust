use crate::activation::{relu, sigmoid, identity};
use crate::layers::base_layer::AbstractLayerTrait;


#[derive(Debug)]
pub struct FcLayer {
    pub name: String,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub i_size: usize,
    pub o_size: usize,
    pub activation_type: String,
    pub activation_fn: fn(f32) -> f32,
}

impl FcLayer {
    pub fn new(name: String, i_size: usize, o_size: usize) -> Self {
        Self { 
            name, 
            w: vec![0.0; i_size * o_size], 
            b: vec![0.0; o_size], 
            i_size, 
            o_size, 
            activation_type: "identity".to_string(), 
            activation_fn: identity,
        }
    }
}

impl AbstractLayerTrait for FcLayer {
    fn build(&mut self) {
        self.w = vec![0.0; self.i_size * self.o_size];
        self.b = vec![0.0; self.o_size];
        match self.activation_type.as_str() {
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
        &self.name
    }

    fn i_size(&self) -> usize {
        self.i_size
    }

    fn o_size(&self) -> usize {
        self.o_size
    }

    fn w(&self) -> &Vec<f32> {
        &self.w
    }

    fn b(&self) -> &Vec<f32> {
        &self.b
    }

    fn activation_type(&self) -> &str {
        &self.activation_type
    }
}