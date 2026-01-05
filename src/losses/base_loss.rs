pub trait AbstractLossFunctionTrait {
    fn forward(&self, y_true: &[f32], y_pred: &[f32]) -> f32;
    fn backward(&self, y_true: &[f32], y_pred: &[f32]) -> Vec<f32>;
    fn name(&self) -> &str;
    fn build(&mut self);
}

#[derive(Debug)]
pub struct AbstractLossFunction {
    pub name: String,
    pub loss: f32,
    pub gradient: Vec<f32>,
}

impl AbstractLossFunction {
    pub fn new(name: String) -> Self {
        Self { name, loss: 0.0, gradient: Vec::new() }
    }
}