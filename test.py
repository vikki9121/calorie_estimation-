import os
import cv2
import numpy as np
from sklearn.decomposition import PCA

def read_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

def apply_pca(images, bounding_boxes, max_components=None):
    transformed_images = []
    for img, bbox in zip(images, bounding_boxes):
        # Extract region within bounding box
        x, y, w, h = bbox
        roi = img[y:y+h, x:x+w]
        
        # Convert region to grayscale and resize
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_resized = cv2.resize(roi_gray, (100, 100))
        
        # Flatten and apply PCA
        flattened_roi = roi_resized.flatten()
        print("Shape of flattened ROI:", flattened_roi.shape)
        if max_components is None:
            max_components = min(flattened_roi.shape)
        n_components = min(max_components, min(flattened_roi.shape))
        print("Max components:", max_components)
        print("Computed components:", n_components)
        if n_components > 1:
            pca = PCA(n_components=n_components)
            pca.fit(flattened_roi.reshape(1, -1))
            transformed_roi = pca.transform(flattened_roi.reshape(1, -1)).reshape(100, 100).astype(np.uint8)
            transformed_images.append(transformed_roi)
        else:
            print("Warning: Skipping PCA for region with insufficient features.")
    
    return transformed_images

def save_transformed_images(transformed_images, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    for i, img in enumerate(transformed_images):
        filename = os.path.join(output_folder, f"transformed_image_{i}.jpg")
        cv2.imwrite(filename, img)

def main():
    input_folder = r'D:\\calorie_estimation\\IndianFoodImages\\IndianFoodImages\\adhirasam'
    output_folder = 'D:\\calorie_estimation\\ExtractedFeatures'

    # Read images and bounding boxes from input folder
    images = read_images_from_folder(input_folder)
    bounding_boxes = [(100, 100, 200, 200), (50, 50, 150, 150)]  # Example bounding boxes
    
    if not images:
        print("No images found in the input folder.")
        return

    # Apply PCA for feature extraction within bounding boxes
    transformed_images = apply_pca(images, bounding_boxes)

    # Save transformed images to output folder
    save_transformed_images(transformed_images, output_folder)

    print("Images processed and saved successfully.")

if __name__ == "__main__":
    main()
