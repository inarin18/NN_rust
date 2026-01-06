use crate::model::Model;
use crate::optimizers::base_optimizer::AbstractOptimizerTrait;
use crate::losses::base_loss::AbstractLossFunctionTrait;
use crate::data::MnistData;

pub struct Trainer<O, L>
where
    O: AbstractOptimizerTrait,
    L: AbstractLossFunctionTrait,
{
    pub model: Model,
    pub optimizer: O,
    pub loss_function: L,
    pub dataset: MnistData,
    // pub device: Device,
    pub epoch: usize,
    // pub batch_size: usize,
    pub verbose: bool,
    // pub callbacks: Vec<Box<dyn Callback>>,
    // pub metrics: Vec<Box<dyn Metric>>,
    // pub visualization: bool,
}

impl<O, L> Trainer<O, L> where
    O: AbstractOptimizerTrait,
    L: AbstractLossFunctionTrait,
{
    pub fn new(
        model: Model, 
        optimizer: O, 
        loss_function: L, 
        dataset: MnistData, 
        epoch: usize, 
        verbose: bool
    ) -> Self {
        Self { model, optimizer, loss_function, dataset, epoch, verbose }
    }

    pub fn run(&mut self) {
        for epoch in 0..self.epoch {
            if self.verbose {
                println!("Epoch {}/{}", epoch + 1, self.epoch);
            }
            // TODO: 実際の訓練ループを実装
            // 1. データを取得
            // 2. forward
            // 3. loss計算
            // 4. backward
            // 5. optimizer.update
        }
    }
}