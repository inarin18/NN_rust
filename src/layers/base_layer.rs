pub trait AbstractLayerTrait {
    fn forward(&self, x: &[f32]) -> Vec<f32>;
    fn name(&self) -> &str;
    fn i_size(&self) -> usize;
    fn o_size(&self) -> usize;
    fn w(&self) -> &Vec<f32>;
    fn b(&self) -> &Vec<f32>;
    fn activation_type(&self) -> &str;
    fn build(&mut self);
}

#[derive(Debug)]
pub struct AbstractLayer {
    pub name: String,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub i_size: usize,
    pub o_size: usize,
    pub activation_type: String,
}

impl AbstractLayer {
    pub fn new(name: String, i_size: usize, o_size: usize, activation_type: String) -> Self {
        Self {
            name,
            w: vec![0.0; i_size * o_size],
            b: vec![0.0; o_size],
            i_size,
            o_size,
            activation_type,
        }
    }
}