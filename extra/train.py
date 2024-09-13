import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate synthetic data (replace this with your actual dataset loading code)
np.random.seed(42)

# Features: dummy network traffic data (you would replace this with your features)
X_normal = np.random.normal(0, 1, size=(1000, 10))
X_attack = np.random.normal(10, 2, size=(100, 10))

# Labels: 0 for normal, 1 for attack
y_normal = np.zeros(1000)
y_attack = np.ones(100)

# Concatenate normal and attack data
X = np.concatenate([X_normal, X_attack])
y = np.concatenate([y_normal, y_attack])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a simple neural network model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),  # Update the input_shape
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

# Save the trained model to a file (e.g., model_ddos.h5)
model.save('model_ddos.h5')
