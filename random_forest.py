import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


if __name__ == "__main__":

    features_csv_path = "elon_features/elonmusk_features.csv"
    
    features_df = pd.read_csv(features_csv_path)
    # defining labels
    labels = np.array(features_df['tweet_bool'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    features_df = features_df.drop('tweet_bool', axis = 1)

    # Convert to numpy array
    features = np.array(features_df)

    # Using Skicit-learn to split data into training and testing sets    
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)

    # printing shapes to ensure everything is fine
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Testing Features Shape:', test_features.shape)
    print('Testing Labels Shape:', test_labels.shape)

     
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # Use the forest's predict method on the test data
    predictions = rf.predict(test_features)
    # Calculate the absolute errors
    errors = abs(predictions - test_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2))

    # Calculate mean absolute percentage error (MAPE)
    import ipdb
    ipdb.set_trace()
    mape = 100 * (errors / test_labels)
    # Calculate and display accuracy
    accuracy = 100 - np.mean(mape)
    print('Accuracy:', round(accuracy, 2), '%.')
