import tensorflow as tf

# Load the existing model
model = tf.keras.models.load_model('SER_model.h5')

# Save the model in the newer format
model.save('SER_model_tf_version.h5')
