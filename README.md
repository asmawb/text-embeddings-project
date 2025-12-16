## Running the Project

This project implements a complete, script-based machine learning pipeline
for **movie genre classification from plot descriptions**.
The steps below describe what each stage does and how the system is used.

---

### Step 1: Prepare the dataset

The dataset preparation step cleans the raw **TMDB 5000 Movie Dataset**
and converts it into a format suitable for machine learning.

Dataset source:
https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata

During this step:
- Movie plot overviews are extracted
- Movies are filtered to a fixed set of genres
- Samples with missing or invalid text are removed
- The dataset is split into training and test sets

This step produces processed files that are later used for training.

---

### Step 2: Train the models

In the training stage, two different text representations are evaluated:

1. TF-IDF features with Logistic Regression  
2. SBERT (Sentence-BERT) embeddings with Logistic Regression  

Both models are trained on the same dataset and evaluated on a held-out test set.
The model with the higher performance is automatically selected as
the final model.

After training:
- The best-performing model is saved locally
- Evaluation metrics are stored for reference
- The selected model is used for inference in the demo

All trained models and metrics are stored in the `outputs/` directory,
which is intentionally excluded from GitHub.

---

### Step 3: Run the interactive demo

The project includes an interactive command-line demo that allows a user
to enter any movie description and receive a genre prediction.

The demo loads the best model selected during training and performs
real-time inference on the input text.

For each input, the system outputs:
- The predicted movie genre
- A confidence score indicating prediction certainty

---

### Example

Input movie description:

A detective races against time to stop a serial killer before the next murder.

Model output:

Predicted genre: Thriller  
Confidence score: 0.78

This demonstrates how natural language plot descriptions can be mapped
to high-level movie genres using text embeddings and classical classifiers.

---

### Notes

- The dataset is not included in the repository and must be downloaded
  separately from Kaggle.
- Trained models are excluded from version control to keep the repository clean.
- All results can be reproduced by following the documented steps.
