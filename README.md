

```markdown
# VGG16 Density Map – Bee Counting (YOLO Dataset)

This project trains a **density-map based object counter** (adapted for bees) using a **VGG16 encoder–decoder network**.  
The dataset uses **YOLO TXT annotations**, converting each bounding box to a **Gaussian point density**.  
The model predicts a **128×128 density map**, whose sum represents the predicted count.

---

## Features
- Loads YOLO TXT labels (`class cx cy w h`)
- Automatically creates per-object **Gaussian kernels scaled by box size**
- GPU-accelerated Gaussian generation (if CUDA is available)
- VGG16 encoder with multi-scale skip connections
- Density map supervision with MSE loss
- Automatic train/validation split
- Evaluation: predicted vs. ground truth count + visualization

---

## Dataset Structure
Your dataset must follow:

```

dataset/
└── train/
├── images/
│   ├── frame0001.jpg
│   ├── frame0002.jpg
│   └── ...
└── labels/
├── frame0001.txt
├── frame0002.txt
└── ...

```

Each TXT annotation file (YOLO format):

```

class x_center y_center width height

```

All coordinates are normalized.

---

## Model Summary
- **Encoder**: VGG16 convolutional layers up to layer 23  
- **Decoder**: 5-stage transpose convolution network  
- **Skip connections** at 4 spatial levels  
- **Output**: 1 × 128 × 128 density map  

Counting:

```

predicted_count = density_map.sum() / scale_factor

````

---

## Install Requirements
```bash
pip install torch torchvision matplotlib tqdm opencv-python
````

---

## How to Run

```bash
python vgg16_density_map_fixed_eval_fast.py
```

Or run directly in Jupyter/Kaggle notebooks.

---

## Evaluation

The script provides visualizations:

* Original image
* Ground truth density map
* Predicted density map

It prints **GT vs predicted counts** for quick verification.


