# AQI Prediction Model

## Overview

This project focuses on predicting Air Quality Index (AQI) values using machine learning models. The dataset includes various features related to AQI, and two regression models, `RandomForestRegressor` and `AdaBoostRegressor`, are employed to predict AQI values based on these features.

## Project Structure

- `data/`: Directory containing the dataset file.
- `aqi_prediction.py`: Python script for loading the dataset, training models, and making predictions.
- `README.md`: This documentation file.

## Dataset

The dataset used in this project is `AQI and Lat Long of Countries.csv`, which includes information about AQI values and other features relevant to the prediction.

## Installation

To run this project, you need Python installed along with the following libraries:

- `pandas`
- `scikit-learn`

You can install the required libraries using `pip`:
    ```bash
    pip install pandas scikit-learn
    ```

## Usage

1. **Clone the Repository**

    ```bash
    git clone https://github.com/username/AQI-PREDICTION-MODEL.git
    cd AQI-PREDICTION-MODEL
    ```

2. **Place the dataset in the project directory**

    Ensure that the dataset file `AQI and Lat Long of Countries.csv` is in the same directory as the script.

3. **Run the Python script**

    Execute the script to train the models and make predictions:

    ```bash
    python aqi_prediction.ipynb
    ```

## Script Details

The script performs the following steps:

1. **Load Dataset**

The dataset `AQI and Lat Long of Countries.csv` is loaded into a pandas DataFrame:
    ```bash
    import pandas as pd
    train = pd.read_csv("AQI and Lat Long of Countries.csv")
    ```
2. **Preprocess Data**

Drop the target column (`AQI Value`) from the dataset.
Separate features and target values:
    ```bash
    train1 = train.drop(['AQI Value'], axis=1)
    target = train['AQI Value']
    ```
3. **Train Models**

RandomForestRegressor: Trained on the dataset:
    ```bash
    from sklearn.ensemble import RandomForestRegressor
    m1 = RandomForestRegressor()
    m1.fit(train1, target)
    ```

AdaBoostRegressor: Trained on the dataset:
    ```bash
    from sklearn.ensemble import AdaBoostRegressor
    m2 = AdaBoostRegressor()
    m2.fit(train1, target)
    ```
4. **Evaluate Models**

Print the accuracy score of each model:
    ```bash
    print("RandomForestRegressor accuracy:", m1.score(train1, target) * 100)
    print("AdaBoostRegressor accuracy:", m2.score(train1, target) * 100)
    ```

Predict AQI values for given feature values using both models:
    ```bash
    prediction_result = m1.predict([[1, 10, 5, 11, 10, 5]])
    print("RandomForestRegressor prediction:", prediction_result)
    prediction_result_ada = m2.predict([[1, 45, 67, 34, 5, 23]])
    print("AdaBoostRegressor prediction:", prediction_result_ada)
    ```

## Example Output

```bash
RandomForestRegressor accuracy: 99.9745506378791
AdaBoostRegressor accuracy: 93.00690185649916
RandomForestRegressor prediction: [9.89]
AdaBoostRegressor prediction: [49.67240716]
```

## Contributing

Feel free to contribute to this project by submitting issues or pull requests. For major changes, please open an issue first to discuss what you want to change.

## Contact

For any questions or feedback, please reach out to <rastogikunal19@gmail.com>.
