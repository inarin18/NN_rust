use crate::layers::base_layer::AbstractLayerTrait;

#[derive(Debug)]
pub struct SoftmaxLayer {
    pub name: String,
    pub w: Vec<f32>,
    pub b: Vec<f32>,
    pub i_size: usize,
    pub o_size: usize,
    pub activation_type: String,
}

impl SoftmaxLayer {
    pub fn new(name: String, i_size: usize, o_size: usize) -> Self {
        Self {
            name, 
            w: vec![0.0; i_size * o_size],
            b: vec![0.0; o_size],
            i_size, 
            o_size, 
            activation_type: "softmax".to_string(), 
        }
    }
}

impl AbstractLayerTrait for SoftmaxLayer {
    fn build(&mut self) {}

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