import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
import pydot
from sklearn.tree import export_graphviz
import joblib


if __name__ == "__main__":

    features_csv_path = "elon_features/elonmusk_features.csv"
    out_dir = "random_forest"    

    os.makedirs(out_dir, exist_ok=True)
    
    features_df = pd.read_csv(features_csv_path, index_col=0)   # index col=0 doesn't read first column into dataframe
    # defining labels
    labels = np.array(features_df['tweet_bool'])

    # Remove the labels from the features
    # axis 1 refers to the columns
    features_df = features_df.drop('tweet_bool', axis = 1)

    # Convert to numpy array
    features = np.array(features_df)

    # Using Skicit-learn to split data into training and validation sets    
    # train_features, val_features, train_labels, val_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
    
    # splitting features into train and val set.
    # we will take the last 2 months as validation set
    train_features = features[0:-861,:]
    val_features = features[-861:,:]
    train_labels = labels[0:-861]
    val_labels = labels[-861:]

    # printing shapes to ensure everything is fine
    print('Training Features Shape:', train_features.shape)
    print('Training Labels Shape:', train_labels.shape)
    print('Validation Features Shape:', val_features.shape)
    print('Validation Labels Shape:', val_labels.shape)

     
    # Instantiate model with 1000 decision trees
    rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
    # Train the model on training data
    rf.fit(train_features, train_labels)

    # exporting the model    
    joblib.dump(rf, os.path.join(out_dir, "random_forest.joblib"))
    # use compress to reduce model size. I need to check if it has any impact on model's accuracy
    # joblib.dump(rf, os.path.join(out_dir, "random_forest.joblib"), compress=3)
    # to load model, use the following command
    # rf = joblib.load(os.path.join(out_dir, "random_forest.joblib"))

    # Use the forest's predict method on the test data
    predictions = rf.predict(val_features)
    # Calculate the absolute errors
    errors = abs(predictions - val_labels)
    # Print out the mean absolute error (mae)
    print('Mean Absolute Error:', round(np.mean(errors), 2))

    # Get accuracy
    predicted_labels = np.round(predictions)
    accuracy = accuracy_score(val_labels, predicted_labels)   
    print('Accuracy:', round(accuracy*100, 2), "%")

    # saving gt and predicted labels
    data = {"gt_labels": val_labels.tolist(), "predicted_labels": predicted_labels.tolist()}
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(out_dir, "gt_and_predicted_labels.csv"))


    # exporting visualization for a single decision tree
    # Pull out one tree from the forest
    tree = rf.estimators_[5]
    # Export the image to a dot file    
    feature_list = list(features_df.columns)
    export_graphviz(tree, out_file = os.path.join(out_dir, 'tree.dot'), feature_names = feature_list, rounded = True, precision = 1)
    # Use dot file to create a graph
    (graph, ) = pydot.graph_from_dot_file(os.path.join(out_dir, 'tree.dot'))
    # Write graph to a png file
    graph.write_png(os.path.join(out_dir, 'tree.png'))

    # Get numerical feature importances
    importances = list(rf.feature_importances_)
    # List of tuples with variable and importance
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    # Sort the feature importances by most important first
    feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)
    # Print out the feature and importances 
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]