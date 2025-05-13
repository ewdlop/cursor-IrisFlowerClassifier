# Iris Flower Classifier

This project contains a machine learning classifier for the Iris flower dataset. The classifier is trained to predict the species of Iris flowers based on their sepal and petal measurements.

## Dataset

The dataset contains the following features:
- sepal_length
- sepal_width
- petal_length
- petal_width

And the target species:
- Iris-setosa
- Iris-versicolor
- Iris-virginica

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have a Hugging Face account and are logged in:
```bash
huggingface-cli login
```

## Training the Model

To train the model and upload it to Hugging Face:

```bash
python train_iris_classifier.py
```

The script will:
1. Load and preprocess the data
2. Train a Random Forest Classifier
3. Evaluate the model's performance
4. Save the model locally
5. Upload the model to Hugging Face Hub

## Model Performance

The model is evaluated using accuracy score and a detailed classification report, which will be displayed during training.

## Using the Model

You can load and use the model from Hugging Face Hub using:

```python
from huggingface_hub import hf_hub_download
import joblib

# Download the model
model_path = hf_hub_download(repo_id="your-username/iris-classifier", filename="iris_classifier.joblib")

# Load the model
model = joblib.load(model_path)

# Make predictions
predictions = model.predict([[5.1, 3.5, 1.4, 0.2]])  # Example measurements
``` 