# Brain MRI Segmentation and Classification

This project involves brain MRI image segmentation and classification using advanced deep learning algorithms. The models implemented include UNet, ResUNet, ResNet, and Transformer-based architectures to accurately segment and classify brain MRI scans.

## Algorithms Used
- **UNet**: A convolutional network for biomedical image segmentation.
- **ResUNet**: A variant of UNet with residual connections for improved performance.
- **ResNet**: A deep residual network for image classification.
- **Transformer-based models**: Used for classification tasks leveraging attention mechanisms.

## Project Structure

- `app.py`: Main application script for model predictions.
- `brain_resunet_pred1 - Copy.ipynb`: Notebook for ResUNet prediction on brain MRI data.
- `brain_unet_pred1 - Copy.ipynb`: Notebook for UNet prediction on brain MRI data.
- `resnet-brain.ipynb`: Notebook for ResNet-based brain MRI classification.
- `transformer_brain (2).ipynb`: Notebook for Transformer-based brain MRI classification.
- `resunet_brain_mri_seg.hdf5`: Pretrained ResUNet model for brain MRI segmentation.

## Dataset
The dataset used for training and evaluation can be accessed from the following link:
[Brain MRI Segmentation Dataset](<INSERT-DATASET-LINK-HERE>)

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/brain-mri-segmentation.git
    ```
2. Install required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

- To perform segmentation or classification, you can run the relevant Jupyter notebooks (`.ipynb` files) or the `app.py` script for real-time predictions.

## Results

- Segmentation and classification results will be stored in the `outmodel/` directory.

## License
This project is licensed under the MIT License.
