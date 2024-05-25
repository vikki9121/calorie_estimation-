import cv2
import os
import numpy as np
from sklearn.decomposition import PCA
import pickle

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

    def extract_features_from_folder(self, folder_path, output_folder):
        # Create the output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                features = self.extract_features(image_path)
                if features is not None:
                    output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + '.jpg')
                    with open(output_file, 'wb') as f:
                        pickle.dump(features, f)
                    print(f"Features extracted from {filename} and saved to {output_file}")

# Example usage:
if __name__ == "__main__":
    # Initialize ImageFeatureExtractor instance with desired number of PCA components
    feature_extractor = ImageFeatureExtractor(n_components=1)  # Adjust n_components as needed

    # Input folder containing images
    input_folder = r'D:\\calorie_estimation\\IndianFoodImages\\IndianFoodImages\\adhirasam'

    # Output folder to store extracted features
    output_folder = 'D:\\calorie_estimation\\ExtractedFeatures'

    # Extract features from images in the input folder and save to output folder
    feature_extractor.extract_features_from_folder(input_folder, output_folder)
