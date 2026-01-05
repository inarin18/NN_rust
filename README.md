# Neural Network in Rust

## Description

This is a simple neural network implementation in Rust.

## Features

- Simple neural network implementation
- Easy to understand
- Easy to modify

## TODO

  - [x] Add some layers
  - [x] Add some activation functions
  - [x] Add forward path
  - [ ] Add loss function
  - [ ] Add optimization algorithm
  - [x] Add data loading method
  - [ ] Add evaluation metric
  - [ ] Add visualization method
  - [ ] Add logging
  - [ ] Add testing
  - [ ] Add documentation
  - [ ] Add examples

## Memo

trait はあくまでも共通の fn のシグネチャを定義するためのモノであり，実装はそれぞれの型によって行われる。

また，impl はあくまで struct に対して実装を行うためのモノであり，必ずしも trait を継承しなくてよい．