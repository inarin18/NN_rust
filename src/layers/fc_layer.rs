use crate::activation::{relu, sigmoid, identity};
use crate::layers::base_layer::{AbstractLayer, AbstractLayerTrait};
use rand_distr::{Distribution, Normal};


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

        // parameters for weight initialization
        let mu: f64 = 0.0;
        let sigma: f64 = 0.01;

        // initialize weights and biases
        let mut rng = rand::thread_rng();
        let normal = Normal::new(mu, sigma).unwrap();
        for i in 0..self.base.i_size * self.base.o_size {
            self.base.w[i] = normal.sample(&mut rng) as f32;
        }
        for i in 0..self.base.o_size {
            self.base.b[i] = normal.sample(&mut rng) as f32;
        }

        // set activation function
        match self.base.activation_type.as_str() {
            "relu" => self.activation_fn = relu,
            "sigmoid" => self.activation_fn = sigmoid,
            _ => self.activation_fn = identity,
        }
    }

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.base.o_size);
        
        // calculate output for each neuron
        for j in 0..self.base.o_size {
            let weight_row_start = j * self.base.i_size;
            let weight_row = &self.base.w[weight_row_start..weight_row_start + self.base.i_size];
            
            // calculate dot product of input and weight
            let dot_product: f32 = x.iter()
                .zip(weight_row.iter())
                .map(|(x_i, w_ij)| x_i * w_ij)
                .sum();

            let activation: f32 = (self.activation_fn)(dot_product + self.base.b[j]);
            out.push(activation);
        }

        out
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