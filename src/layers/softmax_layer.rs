use crate::layers::base_layer::{AbstractLayer, AbstractLayerTrait};
use crate::activation::softmax;

#[derive(Debug)]
pub struct SoftmaxLayer {
    base: AbstractLayer,
}

impl SoftmaxLayer {
    pub fn new(name: String, i_size: usize, o_size: usize) -> Self {
        Self {
            base: AbstractLayer::new(name, i_size, o_size, "softmax".to_string()),
        }
    }
}

impl AbstractLayerTrait for SoftmaxLayer {
    fn build(&mut self) {}

    fn forward(&self, x: &[f32]) -> Vec<f32> {
        softmax(x)
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