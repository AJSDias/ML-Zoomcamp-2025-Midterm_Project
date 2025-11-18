
import pandas as pd
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split

# categorical columns to encode
cat_cols = ['artists', 'track_genre']

# target encoding of categorical features
def target_encode_genre(cat_cols, X_train, X_test, y_train):

    df = X_train.join(y_train)

    for col in cat_cols:
    
        col_target_map = df.groupby(col)["popularity"].mean()
        global_mean = y_train.mean()

        X_train[f"{col}_popularity"] = X_train[col].map(col_target_map).fillna(global_mean)
        X_test[f"{col}_popularity"]  = X_test[col].map(col_target_map).fillna(global_mean)
        
        X_train = X_train.drop(columns=[col])
        X_test  = X_test.drop(columns=[col])
    
    return X_train, X_test, y_train

# prepare dataset for model
def prepare_dataset(data_path, encode_categorical=False):
    
    df = pd.read_csv(data_path)
    
    df.drop(columns=['artists', 'track_genre'], inplace=True)
    
    X = df.drop(columns=["popularity"])
    y = df["popularity"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    if encode_categorical:
        X_train, X_test, y_train = target_encode_genre(cat_cols, X_train, X_test, y_train)
    
    return X_train, X_test, y_train, y_test


best_xgb_params = {
    'learning_rate': 0.1,
    'max_depth': 6,
    'n_estimators': 1000,
    'subsample': 0.8
}

# train model
def train_model(X_train, y_train):
    model = xgb.XGBRegressor(**best_xgb_params, random_state=42)
    model.fit(X_train, y_train)
    return model

# save model
def save_model(model, output_path):
    with open(output_path, 'wb') as f_out:
        pickle.dump(model, f_out)