import cv2
import os
import numpy as np
from sklearn.decomposition import PCA

class ImageFeatureExtractor:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def extract_features(self, image_path):
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image from '{image_path}'")
            return None

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Flatten the image into a 1D array
        flattened_image = gray_image.flatten().astype(np.float32)

        # Apply PCA to extract features
        pca_features = self.pca.fit_transform([flattened_image])

        return pca_features

# Example usage:
if __name__ == "__main__":
    # Initialize ImageFeatureExtractor instance with desired number of PCA components
    feature_extractor = ImageFeatureExtractor(n_components=64)  # Adjust n_components as needed

    # Path to the image
    image_path = r'D:\\calorie_estimation\\IndianFoodImages\\IndianFoodImages\\adhirasam\\02d09e872d.jpg'

    # Extract features from the image
    features = feature_extractor.extract_features(image_path)

    if features is not None:
        print("Extracted features shape:", features.shape)
