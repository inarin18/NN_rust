use crate::model::Model;

pub trait AbstractOptimizerTrait {
    fn update(&mut self, model: &mut Model);
    fn name(&self) -> &str;
    fn build(&mut self);
}

#[derive(Debug)]
pub struct AbstractOptimizer {
    pub name: String,
    pub learning_rate: Option<f32>,
    pub verbose: Option<bool>,
    pub model: Option<Model>,
}

impl AbstractOptimizer {
    pub fn new(name: String) -> Self {
        Self { name, learning_rate: None, verbose: None, model: None }
    }
}

