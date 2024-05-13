import pickle
from keras.models import load_model

# Load the Keras model from the HDF5 file
model = load_model('Trained_Model.h5')

# Save the model architecture and weights separately
model_architecture = model.to_json()
model_weights = model.get_weights()

# Combine architecture and weights into a dictionary
model_data = {
    'architecture': model_architecture,
    'weights': model_weights
}

# Save the model data as a pickle file
with open('model_data.pkl', 'wb') as f:
    pickle.dump(model_data, f)
