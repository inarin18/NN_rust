# Repository Guidelines

## Project Structure & Module Organization
- `src/`: Rust ソースの中心。`main.rs` がエントリポイントで、`layers/`, `losses/`, `optimizers/` が機能別モジュール。
- `data/`: 学習データ（例: `data/mnist.bin`）。
- `load_mnist.py`: MNIST を `data/mnist.bin` に生成する補助スクリプト。
- `doc/`: メモや資料（必要に応じて参照）。
- `target/`: ビルド成果物（編集不要）。

## Build, Test, and Development Commands
- `cargo build`: Rust バイナリをビルド。
- `cargo run`: `main.rs` を実行（`data/mnist.bin` が必要）。
- `cargo test`: テスト実行（現状はテスト未整備のため追加後に利用）。
- `python load_mnist.py`: MNIST データを再生成（`data/mnist.bin` を上書き）。

## Coding Style & Naming Conventions
- Rust の標準スタイルに準拠（`snake_case` の関数/変数、`CamelCase` の型名）。
- インデントは 4 スペース。`cargo fmt` で整形することを推奨。
- モジュール分割は `layers/`, `losses/`, `optimizers/` 配下に実装を追加する方針。

## Testing Guidelines
- 現在は自動テスト未整備。追加する場合は `cargo test` で実行できる構成にする。
- テスト名は機能名を反映した説明的な名前にする（例: `test_sgd_update_applies_gradient`）。

## Commit & Pull Request Guidelines
- コミットは英語の命令形で簡潔に（例: `Add Trainer`, `Refactor Trainer`）。
- 変更理由と影響範囲を PR 説明に記載すること。データ変更がある場合は再生成手順も明記。
- 大きな機能追加は小さな論理単位に分割し、レビューしやすい単位で提出する。

## Data & Configuration Notes
- 実行には `data/mnist.bin` が必要。欠落時は `load_mnist.py` で生成する。
- 学習/評価の入口は `src/main.rs` の `Trainer` 実行部分。
