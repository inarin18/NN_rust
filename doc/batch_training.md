# バッチ学習のメモ

## 概要
- `Trainer` にバッチ処理を追加し、バッチ内の勾配を平均して更新するように変更。
- レイヤーの勾配バッファにミュータブルにアクセスできるように拡張。
- `main.rs` でバッチサイズを指定するように更新。

## 仕組み
- バッチごとにサンプル単位で forward/backward を実行し、レイヤーごとに勾配を加算。
- `1 / batch_size` で平均した勾配をセットして、バッチ単位で更新。

## 変更ファイル
- `src/trainer.rs`
- `src/layers/base_layer.rs`
- `src/layers/fc_layer.rs`
- `src/layers/softmax_layer.rs`
- `src/main.rs`
