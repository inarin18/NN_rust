use crate::losses::base_loss::{AbstractLossFunction, AbstractLossFunctionTrait};

pub struct CrossEntropyLoss {
    base: AbstractLossFunction,
}

impl CrossEntropyLoss {
    pub fn new(name: String) -> Self {
        Self { base: AbstractLossFunction::new(name) }
    }
}

impl AbstractLossFunctionTrait for CrossEntropyLoss {
    fn forward(&self, y_true: &[f32], y_pred: &[f32]) -> f32 {
        const EPSILON: f32 = 1e-7;
        let loss = -y_true.iter()
            .zip(y_pred.iter())
            .map(|(y_true_i, y_pred_i)| {
                let y_pred_clipped = (y_pred_i + EPSILON).min(1.0 - EPSILON);
                y_true_i * y_pred_clipped.ln()
            })
            .sum::<f32>();
        loss
    }

    fn backward(&self, y_true: &[f32], y_pred: &[f32]) -> Vec<f32> {
        // Cross Entropy Loss の勾配: dL/dy_pred = -y_true / y_pred
        const EPSILON: f32 = 1e-7;
        let gradient: Vec<f32> = y_true.iter()
            .zip(y_pred.iter())
            .map(|(y_true_i, y_pred_i)| {
                let y_pred_clipped = (y_pred_i + EPSILON).min(1.0 - EPSILON);
                -y_true_i / y_pred_clipped
            })
            .collect();
        gradient
    }

    fn name(&self) -> &str {
        &self.base.name
    }

    fn build(&mut self) {}
}