import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from PIL import Image
import sys

# Optional: Fix for UnicodeEncodeError in Windows terminals
sys.stdout.reconfigure(encoding='utf-8')

# Function to load images and labels from the dataset
def load_data(image_folder):
    images = []
    labels = []
    class_names = os.listdir(image_folder)

    print(f"Found {len(class_names)} disease categories in dataset.")

    for label in class_names:
        class_folder = os.path.join(image_folder, label)
        if os.path.isdir(class_folder):
            print(f"Processing images from: {label}")
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                try:
                    img = Image.open(img_path).resize((100, 100))
                    img = np.array(img)
                    if img.shape == (100, 100, 3):  # Ensure image is RGB
                        images.append(img)
                        labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_name}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    print(f"Loaded {len(images)} images.")
    return images, labels

# Function to preprocess the images and labels
def preprocess_data(images, labels):
    images = images.astype('float32') / 255.0
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    labels_one_hot = to_categorical(labels_encoded)
    return images, labels_one_hot, label_encoder

# Build the CNN model
def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the CNN model
def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return history

# Test the model and output accuracy
def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Predict disease for user-provided image
def predict_disease(model, img_path, label_encoder):
    # Check if the image file exists
    if not os.path.exists(img_path):
        print(f"Error: The image file at {img_path} does not exist.")
        return None
    try:
        # Attempt to load and process the image
        img = Image.open(img_path).resize((100, 100))  # Resize to match the input size of the model
        img = np.array(img).astype('float32') / 255.0  # Normalize the image
        img = np.expand_dims(img, axis=0)  # Add batch dimension
        
        # Make prediction using the trained model
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction, axis=1)
        
        # Return the decoded class label
        return label_encoder.inverse_transform(predicted_class)[0]
    except Exception as e:
        print(f"Error processing input image: {e}")
        return None

# Main program
if __name__ == "__main__":
    image_folder = 'C:/IDP2/tomato/train'  # ✅ Update this path if needed

    # Load and preprocess data
    images, labels = load_data(image_folder)
    if len(images) == 0:
        print("No images loaded. Please check the dataset path.")
    else:
        images, labels_one_hot, label_encoder = preprocess_data(images, labels)
        
        # Split the dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(images, labels_one_hot, test_size=0.2, random_state=42)

        # Build and train the CNN model
        input_shape = (100, 100, 3)
        num_classes = len(label_encoder.classes_)
        model = build_cnn_model(input_shape, num_classes)

        train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32)
        test_model(model, X_test, y_test)

        # Predict disease for user-provided image
        user_img_path = input("Enter the path of the image to predict disease: ").strip()
        predicted_label = predict_disease(model, user_img_path, label_encoder)
        
        if predicted_label is not None:
            try:
                print(f"The disease predicted for the given image is: {predicted_label}")
            except UnicodeEncodeError:
                print("The disease predicted for the given image is: " + predicted_label.encode('utf-8', errors='replace').decode())
