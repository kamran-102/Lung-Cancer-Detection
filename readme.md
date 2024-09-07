
# Lung Cancer Detection Model
This project implements a machine learning model to detect lung cancer using classification algorithms with hyperparameter tuning. The project is written in Python and makes use of scikit-learn for model training and evaluation.

## Dataset

The dataset contains 309 entries with the following columns:
- **GENDER**: Gender of the patient
- **AGE**: Age of the patient
- **SMOKING**: Whether the patient is a smoker or not (YES=2 , NO=1)
- **YELLOW_FINGERS**: Whether the patient has yellow fingers (YES=2 , NO=1)
- **ANXIETY**: Anxiety levels of the patient (YES=2 , NO=1)
- **PEER_PRESSURE**: Peer pressure levels (YES=2 , NO=1)
- **CHRONIC DISEASE**: Chronic disease status (YES=2 , NO=1)
- **FATIGUE**: Fatigue levels (YES=2 , NO=1)
- **ALLERGY**: Whether the patient has allergies (YES=2 , NO=1)
- **WHEEZING**: Whether the patient is wheezing (YES=2 , NO=1)
- **ALCOHOL CONSUMING**: Whether the patient consumes alcohol (YES=2 , NO=1)
- **COUGHING**: Coughing status (YES=2 , NO=1)
- **SHORTNESS OF BREATH**: Shortness of breath status (YES=2 , NO=1)
- **SWALLOWING DIFFICULTY**: Whether the patient has swallowing difficulty (YES=2 , NO=1)
- **CHEST PAIN**: Whether the patient has chest pain (YES=2 , NO=1)
- **LUNG_CANCER**: Target variable indicating if the patient has lung cancer (1 for Yes, 0 for No)

## Model Pipeline

The project includes the following steps:

1. **Data Preprocessing**:
    - Loading and exploring the dataset.
    - Handling missing values (if any).
    - Splitting the data into training and testing sets.
    
2. **Model Training**:
    - **RandomForestClassifier** is used for training the model.
    - Hyperparameter tuning is performed using `GridSearchCV` to find the optimal parameters:
        - `n_estimators`: Number of trees in the forest.
        - `max_depth`: Maximum depth of each tree.
        - `min_samples_split`: Minimum number of samples required to split an internal node.
        - `min_samples_leaf`: Minimum number of samples required to be at a leaf node.
        - `bootstrap`: Whether to use bootstrap samples when building trees.

3. **Model Evaluation**:
    - Evaluation metrics used:
        - **Confusion Matrix**: A confusion matrix is plotted to show the performance of the classification model.
        - **Accuracy Score**: Measures how often the classifier is correct.
    - Visualization of the confusion matrix with a heatmap for a clearer understanding of performance.

## Dependencies

- Python 3.x
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib

To install the required libraries, run:

```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

## How to Run

1. Clone this repository or download the project files.
2. Install the necessary dependencies listed above.
3. Run the Jupyter notebook or the Python script to train the model and evaluate its performance.

## Results

The model's performance is evaluated using the accuracy score and confusion matrix. The confusion matrix gives an insight into how many positive and negative cases were correctly and incorrectly classified.

## Disclaimer

This project is intended for educational purposes only. The performance of the model may vary significantly based on the dataset, preprocessing techniques, and model parameters used. The results obtained in this project should not be considered as conclusive for any medical diagnosis or application. 

