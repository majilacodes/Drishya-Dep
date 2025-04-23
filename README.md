# Drishya - Product Image Replacement Tool

Drishya is an AI-powered tool that allows you to seamlessly replace products in advertisement images using Meta's Segment Anything Model (SAM).

## Features

- **Automatic Segmentation**: Draw a bounding box around any product and let SAM generate precise segmentation masks
- **Advanced Blending**: Multiple blending techniques (alpha, poisson, advanced) for realistic product replacement
- **Color Grading**: Match the color tone of replacement products with the background
- **Edge Feathering**: Control the softness of edges for seamless integration
- **Interactive UI**: Simple, step-by-step interface built with Streamlit

## How It Works

1. Upload an advertisement image
2. Draw a bounding box around the product you want to replace
3. Generate segmentation mask using SAM
4. Upload a new product image to replace the original
5. Adjust blending settings and download the final image

## Requirements

- Python 3.8+
- PyTorch 2.0.1
- TorchVision 0.15.2
- Segment Anything Model (SAM)
- Streamlit 1.25.0+
- OpenCV
- NumPy
- Matplotlib
- PIL

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Download the SAM model file (`sam_vit_b_01ec64.pth`) and place it in the project directory
4. Run the app:
   ```
   streamlit run sam-roboflow.py
   ```

## Usage Tips

- For best results, use product images with transparent backgrounds
- The "Advanced" blending method combines multiple techniques for the most realistic results
- Adjust the "Edge Feathering" slider to control the softness of transitions
- Use "Color Grading" to match the replacement product's colors with the target area

## License

This project uses Meta's Segment Anything Model (SAM) which is licensed under the Apache 2.0 license.

## Acknowledgements

- [Meta AI Research](https://ai.meta.com/) for the Segment Anything Model
- [Streamlit](https://streamlit.io/) for the interactive web framework
- [OpenCV](https://opencv.org/) for image processing capabilities

Created with ❤️ using Streamlit and Meta's Segment Anything Model (SAM)