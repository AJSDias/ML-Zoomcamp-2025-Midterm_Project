import model_utils as model_utils

# path to dataset
data_path = "./data/dataset_fe.csv"

# prepare dataset for model training
X_train, X_test, y_train, y_test = model_utils.prepare_dataset(data_path)

# train model
model = model_utils.train_model(X_train, y_train)

# save model in pickle file
save_path = 'model.pkl'
model_utils.save_model(model, save_path)

print('Model saved to model.bin')