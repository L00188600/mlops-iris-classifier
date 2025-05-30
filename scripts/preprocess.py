# scripts/preprocess.py


def preprocess_data(df):
    """
    Applies preprocessing steps to the input DataFrame.
    Currently, only scales numerical features.
    """
    # Assuming the first four columns are numerical features for scaling
    # numerical_features = ['sepal_length', 'sepal_width',
    #                       'petal_length', 'petal_width']

    # Check if a scaler exists from previous training, if not, create a dummy
    # For a real pipeline, the scaler would be trained and saved by train.py
    # and loaded here. For this example, we'll create a new one for inference
    # if not provided. In a strict sense, the scaler should be saved/loaded.

    # For the purpose of this MLOps pipeline, assume the model handles scaling
    # internally or that features are already scaled before prediction.
    # The current predict function directly uses the input without scaling.

    # If you later decide to use a scaler as part of your pipeline,
    # you would load it here and apply it.

    # For now, let's just ensure the data is in the expected format
    # and handle potential issues.

    # Example placeholder if a scaler were to be applied:
    # try:
    #     scaler = joblib.load('models/scaler.joblib')
    #     df[numerical_features] = scaler.transform(df[numerical_features])
    # except FileNotFoundError:
    #     print("Scaler not found. Assuming data is already scaled or no "
    #           "scaling needed.")
    #     # Or train a new scaler if this function is used for training data prep

    return df


if __name__ == '__main__':
    # This block would typically be used to test the preprocessing function
    # or apply it to a dataset for training.
    print("Preprocessing script executed.")
    # Example usage:
    # import pandas as pd
    # df = pd.read_csv('path/to/your/data.csv')
    # preprocessed_df = preprocess_data(df)
    # print(preprocessed_df.head())
