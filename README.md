# ResNet Autoencoder — Midterm Project

## 1. Project Overview

This project explores image reconstruction using ResNet50 and ResNet101 autoencoders implemented with PyTorch.

The goal is to train both models to reconstruct images from the ImageNette dataset and compare their performance using Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR).

The project uses the ResNet encoder and decoder architecture from the [ResNet-decoder GitHub repository](https://github.com/JiahongChen/ResNet-decoder).

## 2. Dataset

The project uses the ImageNette dataset, which contains images from 10 classes.

**Dataset information:**

- Dataset: ImageNette
- Total training images: 9,469
- Total validation images: 3,925
- Training images used in the improved experiment: 2,000
- Validation images used: 200
- Image dimensions: 221 × 221 pixels
- Image channels: 3 (RGB)
- Pixel range: 0 to 1

The images are resized and converted into PyTorch tensors before being passed into the models.

## 3. Models Used

### ResNet50 Autoencoder

The ResNet50 autoencoder consists of an encoder and decoder. The encoder extracts image features, and the decoder attempts to reconstruct the original image from those features.

### ResNet101 Autoencoder

The ResNet101 autoencoder uses a deeper residual network architecture. It follows the same general encoder-decoder process but contains more layers.

Both models are trained to minimize the difference between the original and reconstructed images.

## 4. Model Improvements

During the initial experiments, the reconstructed images appeared mostly gray and contained very little recognizable detail.

Several changes were tested to improve reconstruction quality:

1. Increased the training dataset from 200 to 2,000 images.
2. Increased the validation dataset from 40 to 200 images.
3. Preserved the encoder's 7 × 7 spatial feature maps instead of reducing them to 1 × 1.
4. Removed the decoder's initial upsampling operation because the spatial feature maps were preserved.
5. Replaced the decoder's final ReLU activation with Sigmoid.
6. Performed additional training and evaluation.

The Sigmoid activation restricts the reconstructed pixel values to the range between 0 and 1.

## 5. Training Configuration

| Parameter | Value |
|---|---|
| Framework | PyTorch |
| Optimizer | Adam |
| Loss Function | Mean Squared Error (MSE) |
| Learning Rate | 0.0001 |
| Batch Size | 2 |
| Training Images | 2,000 |
| Validation Images | 200 |
| Input Dimensions | 3 × 221 × 221 |
| Device | Apple Silicon MPS |

The improved ResNet50 experiment also included two additional epochs of fine-tuning after changing the final activation to Sigmoid.

## 6. Results

The following results were obtained by evaluating the trained models on the same 200 validation images.

| Model | Validation MSE | PSNR |
|---|---:|---:|
| ResNet50 with Sigmoid fine-tuning | 0.011028 | 19.57 dB |
| ResNet101 | 0.018450 | 17.34 dB |

The fine-tuned ResNet50 achieved a lower MSE and higher PSNR than ResNet101 in this comparison.

The ResNet50 and ResNet101 models did not have identical training histories, so these results should not be interpreted as proving that ResNet50 is inherently better than ResNet101.

The improved ResNet50 was also evaluated separately and achieved:

- MSE: 0.010648
- PSNR: 19.73 dB
- Validation images: 200

## 7. Reconstruction Quality

The initial models produced mostly gray and blurry images.

After modifying the architecture and training process, the reconstructed images became more recognizable. Objects such as vehicles, dogs, a gas pump, and a boombox could be identified.

Some details, colors, and textures were still blurry or noisy.

## 8. Project Structure

```text
MidTerm Project/
│
├── MidTerm.ipynb
├── README.md
├── spatial_resnet50_sigmoid.pth
│
└── ResNet-decoder-main/
    ├── res_encoder.py
    ├── res_decoder.py
    └── README.md
```

The `.pth` file contains the saved trained ResNet50 model checkpoint.

The ImageNette dataset is stored separately and must be available at the dataset path configured in the notebook.

## 9. Requirements

The project uses the following Python libraries:

- PyTorch
- TorchVision
- NumPy
- Pandas
- Matplotlib
- Jupyter

Python and the required libraries must be installed before running the notebook.

## 10. How to Run the Project

1. Download or clone the ResNet-decoder repository.
2. Download the ImageNette dataset.
3. Open `MidTerm.ipynb` using Jupyter Notebook, JupyterLab, or VS Code.
4. Select the Python environment containing PyTorch and the required libraries.
5. Verify the dataset and repository paths in the notebook.
6. Run the notebook cells in order to initialize the models, load the dataset, train the autoencoders, and evaluate the reconstructed images.

Training may take several minutes depending on the hardware and configuration.

## 11. Limitations and Future Improvements

The reconstructed images still contain blurring and noise.

Possible future improvements include:

- Training for additional epochs.
- Using more images from the ImageNette dataset.
- Testing different reconstruction loss functions.
- Experimenting with decoder architectures.
- Comparing ResNet50 and ResNet101 using identical training conditions.

## 12. References

### ResNet Decoder Repository

JiahongChen — ResNet-decoder

https://github.com/JiahongChen/ResNet-decoder

### ImageNette Dataset

FastAI — ImageNette

https://github.com/fastai/imagenette

### PyTorch Documentation

https://docs.pytorch.org/