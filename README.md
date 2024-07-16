# AdaRand

This repository is an unofficial implementation of AdaRand[1].

And this README also serves as a memo on how to run the Cars dataset.

It is not guaranteed to work correctly.

## File Specifications

- `models.py`: class for sampling random feature vectors and a custom ResNet50 implementation

- `fine_tuning.py`: implementation of ordinary fine-tuning

- `adarand.py`: implementation of fine-tuning with AdaRand[1]

## Usage

### 1. Setup

If you are using poetry, you can add the following dependencies to your `pyproject.toml`:

```toml
[tool.poetry.dependencies]
python = "^3.10"
torch = "^2.3.1"
tqdm = "^4.66.4"
torchvision = "^0.18.1"
```

### 2. Prepare Stanford Cars Dataset

Following the torchvision official instruction from issue [7545](https://github.com/pytorch/vision/issues/7545#issuecomment-1631441616), you can prepare the Stanford Cars Dataset as follows:

```bash
mkdir stanford_cars
cd stanford_cars

# Downloas Stanford Cars Dataset from kaggle
#   https://www.kaggle.com/jessicali9530/stanford-cars-dataset

unzip Stanford\ Cars\ Dataset.zip
mv cars_test/cars_test/*.jpg cars_test/
rm -r cars_test/cars_test/
mv cars_train/cars_train/*.jpg cars_train/
rm -r cars_train/cars_train/

# Download car_devkit
wget https://github.com/pytorch/vision/files/11644847/car_devkit.tgz
tar -xvzf car_devkit.tgz

# Download cars_test_annos_withlabels+(1).mat and rename it to cars_test_annos_withlabels.mat
#   https://www.kaggle.com/code/subhangaupadhaya/pytorch-stanfordcars-classification/input?select=cars_test_annos_withlabels+(1).mat
```

The directory structure should look like this:

```plaintext
stanford_cars/
├── Stanford Cars Dataset.zip
├── car_devkit.tgz
├── cars_annos.mat
├── cars_test_annos_withlabels.mat
├── cars_test/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   ├── ...
│   ├── 08039.jpg
│   ├── 08040.jpg
│   └── 08041.jpg
├── cars_train/
│   ├── 00001.jpg
│   ├── 00002.jpg
│   ├── 00003.jpg
│   ├── ...
│   ├── 08142.jpg
│   ├── 08143.jpg
│   └── 08144.jpg
└── devkit/
    ├── README.txt
    ├── cars_meta.mat
    ├── cars_test_annos.mat
    ├── cars_train_annos.mat
    ├── eval_train.m
    └── train_perfect_preds.txt
```

Then you can read the dataset using the following code:

```python
from pathlib import Path
import torchvision

train_dataset = torchvision.datasets.StanfordCars(
    root=Path('/path/to/stanford_cars').parent,
    split='train',
    transform=torchvision.transforms.ToTensor(),
)
test_dataset = torchvision.datasets.StanfordCars(
    root=Path('/path/to/stanford_cars').parent,
    split='test',
    transform=torchvision.transforms.ToTensor(),
)
```

### 3. Run Fine-tuning

```bash
poetry run python fine_tuning.py
poetry run python adarand.py
```

You can also run the experiments by modifying the following code in the python files:

```python
if __name__ == "__main__":
    config = Config() # set the configuration here
    run_exp(config)
```

## References

[1] Yamaguchi, S. Y., Kanai, S., Adachi, K., & Chijiwa, D. (2024). Adaptive Random Feature Regularization on Fine-tuning Deep Neural Networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 23481-23490)

https://openaccess.thecvf.com/content/CVPR2024/papers/Yamaguchi_Adaptive_Random_Feature_Regularization_on_Fine-tuning_Deep_Neural_Networks_CVPR_2024_paper.pdf
