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

    fn forward(&mut self, x: &[f32]) -> Vec<f32> {

        self.base.last_input = x.to_vec();
        
        let mut pre_activation = Vec::with_capacity(self.base.o_size);
        let mut post_activation = Vec::with_capacity(self.base.o_size);
        
        // calculate output for each neuron
        for j in 0..self.base.o_size {
            let weight_row_start = j * self.base.i_size;
            let weight_row = &self.base.w[weight_row_start..weight_row_start + self.base.i_size];
            
            // calculate dot product of input and weight
            let dot_product: f32 = x.iter()
                .zip(weight_row.iter())
                .map(|(x_i, w_ij)| x_i * w_ij)
                .sum();

            let z = dot_product + self.base.b[j];
            pre_activation.push(z);

            let activation: f32 = (self.activation_fn)(z);
            post_activation.push(activation);
        }

        self.base.last_output = post_activation.clone();
        post_activation
    }

    fn backward(&mut self, grad_output: &[f32]) -> Vec<f32> {

        // 活性化関数の勾配を計算
        let grad_activation: Vec<f32> = self.base.last_output.iter()
            .zip(grad_output.iter())
            .map(|(z, grad)| {
                match self.base.activation_type.as_str() {
                    "relu" => if *z > 0.0 { *grad } else { 0.0 },
                    "sigmoid" => {
                        let s = sigmoid(*z);
                        *grad * s * (1.0 - s)
                    },
                    _ => *grad,  // identity
                }
            })
            .collect();
        
        // 重みとバイアスの勾配用の配列を初期化
        self.base.grad_w = vec![0.0; self.base.i_size * self.base.o_size];
        self.base.grad_b = vec![0.0; self.base.o_size];
        
        for j in 0..self.base.o_size {
            // バイアスの勾配
            self.base.grad_b[j] = grad_activation[j];
            
            // 重みの勾配
            let weight_row_start = j * self.base.i_size;
            for i in 0..self.base.i_size {
                self.base.grad_w[weight_row_start + i] = 
                    grad_activation[j] * self.base.last_input[i];
            }
        }
        
        // 入力に対する勾配を計算（前のレイヤーに伝播）
        let mut grad_input = vec![0.0; self.base.i_size];
        for j in 0..self.base.o_size {
            let weight_row_start = j * self.base.i_size;
            for i in 0..self.base.i_size {
                grad_input[i] += grad_activation[j] * self.base.w[weight_row_start + i];
            }
        }
        
        grad_input
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
    
    fn grad_w(&self) -> &Vec<f32> {
        &self.base.grad_w
    }

    fn grad_b(&self) -> &Vec<f32> {
        &self.base.grad_b
    }

    fn update_weights(&mut self, delta_w: &[f32]) {
        for i in 0..self.base.w.len() {
            self.base.w[i] += delta_w[i];
        }
    }

    fn update_biases(&mut self, delta_b: &[f32]) {
        for i in 0..self.base.b.len() {
            self.base.b[i] += delta_b[i];
        }
    }

    fn activation_type(&self) -> &str {
        &self.base.activation_type
    }
}