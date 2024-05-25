import cv2
import os
import numpy as np
from sklearn.decomposition import PCA

class ImagePreprocessor:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def preprocess(self, image_path, output_folder):
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image from '{image_path}'")
            return False

        # Convert the image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Flatten the image into a 1D array
        flattened_image = gray_image.flatten().astype(np.float32)

        # Apply PCA
        pca_features = self.pca.fit_transform([flattened_image])

        # Reconstruct the image from PCA features
        reconstructed_image = self.pca.inverse_transform(pca_features).reshape(gray_image.shape)

        # Generate output file path
        output_file = os.path.join(output_folder, os.path.basename(image_path))

        # Write the reconstructed image to the output folder
        cv2.imwrite(output_file, reconstructed_image)
        print(f"Preprocessed image saved at: {output_file}")
        return True

if __name__ == "__main__":
    # Initialize ImagePreprocessor instance with desired number of PCA components
    image_preprocessor = ImagePreprocessor(n_components=1)

    # Folder containing the images
    input_folder = 'D:\\calorie_estimation\\IndianFoodImages\\IndianFoodImages\\adhirasam'

    # Output folder to store preprocessed images
    output_folder = 'D:\\calorie_estimation\\PreprocessedImages'

    # Iterate over all images in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".jpg"):
            # Path to the image
            image_path = os.path.join(input_folder, filename)
            # Preprocess the image and store it in the output folder
            image_preprocessor.preprocess(image_path, output_folder)
