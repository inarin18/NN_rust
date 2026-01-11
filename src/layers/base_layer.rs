pub trait AbstractLayerTrait {
    fn forward(&mut self, x: &[f32]) -> Vec<f32>;
    fn backward(&mut self, grad_output: &[f32]) -> Vec<f32>;
    fn name(&self) -> &str;
    fn i_size(&self) -> usize;
    fn o_size(&self) -> usize;
    fn w(&self) -> &Vec<f32>;
    fn b(&self) -> &Vec<f32>;
    fn grad_w(&self) -> &Vec<f32>;
    fn grad_b(&self) -> &Vec<f32>;
    fn grad_w_mut(&mut self) -> &mut Vec<f32>;
    fn grad_b_mut(&mut self) -> &mut Vec<f32>;
    fn update_weights(&mut self, delta_w: &[f32]);
    fn update_biases(&mut self, delta_b: &[f32]);
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
    pub last_input: Vec<f32>,
    pub last_output: Vec<f32>,
    pub grad_w: Vec<f32>,
    pub grad_b: Vec<f32>,
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
            last_input: vec![0.0; i_size],
            last_output: vec![0.0; o_size],
            grad_w: vec![0.0; i_size * o_size],
            grad_b: vec![0.0; o_size],
        }
    }
}
