use crate::model::Model;
pub struct OptimizerParams {
}

impl OptimizerParams {
    pub fn new() -> Self {
        Self {}
    }
}

pub trait AbstractOptimizerTrait {
    type Params;
    
    fn update(&mut self, model: &mut Model);
    fn name(&self) -> &str;
    fn build(&mut self, params: Self::Params);
    
}

#[derive(Debug)]
pub struct AbstractOptimizer {
    pub name: String,
    pub learning_rate: Option<f32>,
    pub verbose: Option<bool>,
}

impl AbstractOptimizer {
    pub fn new(name: String) -> Self {
        Self { name, learning_rate: None, verbose: None }
    }
}

