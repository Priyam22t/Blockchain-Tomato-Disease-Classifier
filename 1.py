import os

image_path = r"C:\pic 2.JPG"
if os.path.exists(image_path):
    print("File exists!")
    # Proceed with loading the image
else:
    print(f"Error: The image file at {image_path} does not exist.")
