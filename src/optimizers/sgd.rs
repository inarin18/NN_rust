use crate::optimizers::base_optimizer::{AbstractOptimizer, AbstractOptimizerTrait};
use crate::model::Model;
pub struct Sgd {
    base: AbstractOptimizer,
}

impl Sgd {
    pub fn new(name: String) -> Self {
        Self { base: AbstractOptimizer::new(name) }
    }
}

pub struct SgdParams {
    pub learning_rate: Option<f32>,
    pub verbose: Option<bool>,
}

impl SgdParams {
    pub fn new() -> Self {
        Self { learning_rate: None, verbose: None }
    }

    pub fn learning_rate(mut self, learning_rate: f32) -> Self {
        self.learning_rate = Some(learning_rate);
        self
    }

    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = Some(verbose);
        self
    }
}

impl AbstractOptimizerTrait for Sgd {
    type Params = SgdParams;

    fn update(&mut self, model: &mut Model) {
        for layer in &mut model.layers {
            let mut delta_w = vec![0.0; layer.w().len()];
            let mut delta_b = vec![0.0; layer.b().len()];
            for i in 0..layer.w().len() {
                delta_w[i] = layer.grad_w()[i] * self.base.learning_rate.unwrap();
            }
            for i in 0..layer.b().len() {
                delta_b[i] = layer.grad_b()[i] * self.base.learning_rate.unwrap();
            }
            layer.update_weights(&delta_w);
            layer.update_biases(&delta_b);
        }
    }

    fn name(&self) -> &str {
        &self.base.name
    }

    fn build(&mut self, params: Self::Params) {
        self.base.learning_rate = params.learning_rate;
        self.base.verbose = params.verbose;
    }
}