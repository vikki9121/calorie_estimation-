import cv2
import os
import numpy as np
from sklearn.decomposition import PCA

class ImagePreprocessor:
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)

    def preprocess(self, input_folder, output_folder):
        # Iterate through all files in the input folder
        for filename in os.listdir(input_folder):
            # Create the full path for the input image
            image_path = os.path.join(input_folder, filename)

            # Read the image in grayscale
            gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray_image is None:
                print(f"Error: Unable to read image from '{image_path}'")
                continue

            # Convert grayscale image to 3-channel (CV_8UC3) for GrabCut
            color_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)

            # Create a mask initialized with zeros
            mask = np.zeros(color_image.shape[:2], np.uint8)

            # Define the foreground and background models for GrabCut
            bgd_model = np.zeros((1, 65), np.float64)
            fgd_model = np.zeros((1, 65), np.float64)

            # Define the rectangle enclosing the object to be segmented
            rectangle = (50, 50, color_image.shape[1] - 50, color_image.shape[0] - 50)

            # Apply GrabCut algorithm to segment the foreground
            cv2.grabCut(color_image, mask, rectangle, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

            # Create a mask where all definite and probable foreground areas are marked as 1
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

            # Apply the mask to the original image
            segmented_image = color_image * mask2[:, :, np.newaxis]

            # Find contours on the mask
            contours, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw bounding boxes around contours
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(segmented_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Generate output file path
            output_file = os.path.join(output_folder, filename)

            # Write the preprocessed image with bounding boxes to the output folder
            cv2.imwrite(output_file, segmented_image)
            print(f"Preprocessed image with background removed and bounding boxes saved at: {output_file}")

        return True

# Example usage:
if __name__ == "__main__":
    # Initialize ImagePreprocessor instance with the desired number of PCA components
    image_preprocessor = ImagePreprocessor(n_components=1)

    # Input folder containing images
    input_folder = r'D:\calorie_estimation - Copy\IndianFoodImages\IndianFoodImages\pizza'

    # Output folder to store preprocessed images
    output_folder = r'D:\calorie_estimation - Copy\ExtractedFeatures\pizza\grac'

    # Preprocess images in the input folder and save them in the output folder
    image_preprocessor.preprocess(input_folder, output_folder)
