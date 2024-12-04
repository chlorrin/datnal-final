import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(data, target_column='Cannabis', test_size=0.2, random_state=42):
    # Transform the target column
    data[target_column] = data[target_column].apply(lambda x: int(x[-1]))

    # Select numerical features and drop irrelevant columns
    X = data.select_dtypes(include=['int64', 'float64']).drop(columns=['ID'], errors='ignore')
    y = data[target_column]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test
