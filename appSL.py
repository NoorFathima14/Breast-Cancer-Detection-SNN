import tensorflow as tf
import tensorflow as tf
print(tf.__version__)

# Load the model
try:
    model = tf.keras.models.load_model('/Users/noorfathima/Documents/college/year 3/sem 5/capstone/best_model.h5')  # For .h5 format
    # or
    # model = tf.keras.models.load_model('path_to_your_model/my_siamese_model')  # For SavedModel format

    print("Model loaded successfully!")

    # You can check the model summary
    model.summary()

    # Test the model with a sample input
    import numpy as np

    # Assuming your model input shape is (100, 100, 3)
    sample_input = np.random.rand(1, 100, 100, 3)  # Generate a random input
    prediction = model.predict(sample_input)
    print("Sample prediction:", prediction)

except Exception as e:
    print("Error loading model:", e)
