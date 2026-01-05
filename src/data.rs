// src/data.rs
use std::fs::File;
use std::io::{BufReader, Read};

pub struct MnistData {
    pub images: Vec<f32>,  // フラットな配列
    pub labels: Vec<u8>,
    pub num_samples: usize,
    pub num_features: usize,
}

impl MnistData {
    pub fn load_from_binary(filename: &str) -> std::io::Result<Self> {
        let mut file = BufReader::new(File::open(filename)?);
        
        // ヘッダー読み込み: [num_samples (u64), num_features (u64)]
        let mut header = [0u8; 16];
        file.read_exact(&mut header)?;
        
        let num_samples = u64::from_le_bytes([
            header[0], header[1], header[2], header[3],
            header[4], header[5], header[6], header[7]
        ]) as usize;
        
        let num_features = u64::from_le_bytes([
            header[8], header[9], header[10], header[11],
            header[12], header[13], header[14], header[15]
        ]) as usize;
        
        // 画像データ読み込み (f32 = 4 bytes)
        let image_size = num_samples * num_features * 4;
        let mut image_bytes = vec![0u8; image_size];
        file.read_exact(&mut image_bytes)?;
        
        let images: Vec<f32> = image_bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();
        
        // ラベル読み込み
        let mut labels = vec![0u8; num_samples];
        file.read_exact(&mut labels)?;
        
        Ok(MnistData {
            images,
            labels,
            num_samples,
            num_features,
        })
    }
    
    pub fn get_image(&self, index: usize) -> Option<&[f32]> {
        if index < self.num_samples {
            let start = index * self.num_features;
            let end = start + self.num_features;
            Some(&self.images[start..end])
        } else {
            None
        }
    }
    
    pub fn get_label(&self, index: usize) -> Option<u8> {
        self.labels.get(index).copied()
    }
}