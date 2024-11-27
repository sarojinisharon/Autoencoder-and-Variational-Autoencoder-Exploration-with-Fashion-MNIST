# Autoencoder and Variational Autoencoder Exploration with Fashion MNIST

## Overview
This project explores Autoencoders (AEs) and Variational Autoencoders (VAEs) using the Fashion MNIST dataset. It demonstrates:
- Dimensionality reduction and data compression using AEs.
- Generative modeling capabilities of VAEs.
- The impact of hyperparameter tuning on model performance.
- A comparative analysis of AEs and VAEs.

## Objectives
1. Select and preprocess an image dataset.
2. Implement and train an Autoencoder (AE).
3. Implement and train a Variational Autoencoder (VAE).
4. Experiment with various tuning parameters.
5. Analyze and compare AE and VAE performance.

## Dataset
The Fashion MNIST dataset, included in Keras datasets, consists of 70,000 grayscale images in 10 categories. Each image is 28x28 pixels. It is widely used for benchmarking image processing models.

## Implementation Steps
1. **Dataset Preprocessing**: Load and normalize the Fashion MNIST dataset.
2. **Autoencoder Development**:
   - Design an encoder-decoder architecture.
   - Train the model on the dataset.
   - Evaluate reconstruction quality.
3. **Variational Autoencoder Development**:
   - Implement latent space sampling.
   - Train the model and generate new images.
   - Assess reconstruction and generative quality.
4. **Performance Tuning**: Experiment with hyperparameters like learning rate, latent space size, and activation functions.
5. **Analysis**: Compare the effectiveness of AEs and VAEs for reconstruction and generation tasks.

## Prerequisites
- Python 3.7+
- TensorFlow 2.0+
- NumPy
- Matplotlib
- Seaborn

## Usage
1. Clone the repository:
   ```bash
   git clone <repository-link>
   cd <repository-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook SarojiniSharon.ipynb
   ```
4. Follow the notebook cells to execute the pipeline.

## Results
- **Autoencoders**: Demonstrated effective dimensionality reduction and reconstruction capabilities.
- **Variational Autoencoders**: Generated new samples resembling the dataset distribution while retaining reconstruction performance.

## Future Work
- Extend the experiment to other datasets.
- Explore advanced architectures like Convolutional Autoencoders.
- Experiment with alternative loss functions and optimization strategies.

## References
- [Fashion MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- TensorFlow Documentation
