## Binary Classification using CNNs: Dogs vs Cats

A deep learning project for binary image classification (dogs vs cats) using **MobileNetV2 transfer learning** with two-phase training, achieving **97.0% validation accuracy**.

### Model

The model uses **MobileNetV2** pretrained on ImageNet with a custom classification head, trained in two phases:

1. **Feature extraction** (15 epochs) — MobileNetV2 base frozen, only the classification head trains
2. **Fine-tuning** (20 epochs) — Last 30 MobileNetV2 layers unfrozen with 10x lower learning rate

MobileNetV2 → GlobalAveragePooling2D → Dense(256, relu) → Dropout(0.5) → Dense(1, sigmoid)

Trained on 20,000 images (80/10/10 split) with data augmentation.

| Metric | Phase 1 (frozen base) | Phase 2 (fine-tuned) |
|---|---|---|
| Validation accuracy | 96.5% | **97.0%** |
| Test accuracy | 96.9% | 96.1% |
| Best val_loss | 0.0849 | **0.0765** |
| Parameters | 2.6M (330K trainable) | 2.6M (1.6M trainable) |
| Model size | 17MB | 17MB |

### Architecture

![Architecture](model/architecture.png)

### Training Curves

![Training Curves](model/training_curves.png)

## Prerequisites

* **Python** 3.9+
* **TensorFlow** 2.14+
* **OpenCV** 4.8+
* **Flask** 3.0+

## How to run

* Create a virtual environment and install dependencies

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

* Download the dataset and place it in `data/` (should contain `train/`, `validation/`, `test/` subdirectories). Link to the training dataset is provided in `src/train/README.md`.

* To train the model:

```
python src/train/dogs_and_cats.py
```

* Start the server:

```
python src/server/server.py
```

* Open http://localhost:5000 in your browser to use the drag-and-drop UI, or use curl:

```
curl -X POST -F 'image=@src/test/1510.jpg' http://localhost:5000/api/class_pred
```

* To evaluate the model on the test dataset:

```
python src/test/test.py
```

# Contributing

Feel free to clone it and make changes to it. Pull requests are welcome.

# Author

Navneet Sharma

# Acknowledgements

Deep Learning for Python Book, Keras Documentation etc.
