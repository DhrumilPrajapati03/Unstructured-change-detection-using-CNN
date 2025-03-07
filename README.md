### Change Detection using VGG19 Feature Maps
This project implements a simple change detection mechanism using deep feature extraction from the VGG19 architecture. It computes the difference between feature maps of two images and highlights changes using Otsu's thresholding.

### Requirements
- Ensure you have the following dependencies installed:
`pip install tensorflow numpy matplotlib scikit-image`

### Usage
Place your two images (img1.png and img2.png) in the working directory.
![img1](https://github.com/user-attachments/assets/01c76ab3-ac52-4814-ad92-c4181a2ddf57)
![img2](https://github.com/user-attachments/assets/48244723-c627-47f8-9ace-b0991aa33fa6)


### Run the script:
`python change_detection.py`

### How It Works
1. Feature Extraction:
- The script builds a simplified VGG19 model, extracting feature maps from different convolutional blocks.
- Features from both images are extracted and compared.

2. Change Detection:
- The squared difference between feature maps is computed.
- Feature maps are resized to match the first feature map size.
- The final change map is obtained by averaging the difference maps and applying Otsu’s threshold.

3. Visualization:
The detected change regions are displayed as a binary mask.

### Example Output:
The script generates a binary change map highlighting differences between the two images.

### Author:
Developed using TensorFlow and Scikit-Image.

### License:
This project is licensed under the MIT License.

