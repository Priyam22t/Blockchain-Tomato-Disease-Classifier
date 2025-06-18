import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from PIL import Image


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
                    img = Image.open(img_path).resize((100, 100))  # Resize images to 100x100
                    img = np.array(img)
                    if img.shape == (100, 100, 3):  # Ensure the image is RGB
                        images.append(img)
                        labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_name}: {e}")

    images = np.array(images)
    labels = np.array(labels)
    print(f"Loaded {len(images)} images.")
    return images, labels


# Function to preprocess and flatten the images
def preprocess_images(images):
    return images.reshape(images.shape[0], -1)  # Flatten the images for Random Forest


# Train Random Forest model
def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model


# Test the model and output accuracy
def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy * 100:.2f}%")


# Predict disease for user-provided image
def predict_disease(model, img_path):
    if not os.path.exists(img_path):
        print(f"Error: The image file at {img_path} does not exist.")
        return None

    try:
        img = Image.open(img_path).resize((100, 100))  # Resize image to 100x100
        img = np.array(img).reshape(1, -1)  # Flatten the image for prediction
        prediction = model.predict(img)
        return prediction[0]
    except Exception as e:
        print(f"Error processing input image: {e}")
        return None


# Main program
if __name__ == "__main__":
    # Path to the dataset
    image_folder = "D:/IDP2/tomato/train"  # Update with the correct path

    # Load and preprocess the data
    images, labels = load_data(image_folder)
    if len(images) == 0:
        print("No images loaded. Please check the dataset path.")
    else:
        images = preprocess_images(images)

        # Encode labels into numerical format
        label_encoder = LabelEncoder()
        labels_encoded = label_encoder.fit_transform(labels)

        # Split dataset into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

        # Train the Random Forest model
        model = train_model(X_train, y_train)

        # Test the model
        test_model(model, X_test, y_test)

        # Take user input image and predict disease
        user_img_path = input("Enter the path of the image to predict disease: ")
        predicted_label = predict_disease(model, user_img_path)
        if predicted_label is not None:
            print(
                f"The disease predicted for the given image is: {label_encoder.inverse_transform([predicted_label])[0]}")
