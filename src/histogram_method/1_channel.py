import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_histogram(image):
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return histogram

def plot_histogram(histogram, title, filename):
    output_dir = os.path.dirname(filename)
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.bar(range(256), histogram, color='red')
    plt.title(title)
    plt.xlabel('Intensywność pikseli')
    plt.ylabel('Liczba wystąpień')

    plt.savefig(filename)
    plt.close()

def detect_stego_changes(original_image, stego_image, method, img_name):
    base_save_dir = os.path.join(images_dir, "histogram_images_gray")
    save_dir = os.path.join(base_save_dir, img_name, method)
    os.makedirs(save_dir, exist_ok=True)

    original_histogram = calculate_histogram(original_image)
    stego_histogram = calculate_histogram(stego_image)

    diff = np.abs(original_histogram - stego_histogram)

    original_hist_filename = os.path.join(save_dir, f"{img_name}_original_histogram.png")
    stego_hist_filename = os.path.join(save_dir, f"{img_name}_stego_histogram.png")
    diff_hist_filename = os.path.join(save_dir, f"{img_name}_difference_histogram.png")

    plot_histogram(original_histogram, f'Histogram obrazu oryginalnego - {img_name}', original_hist_filename)
    plot_histogram(stego_histogram, f'Histogram obrazu z ukrytymi danymi - {img_name}', stego_hist_filename)
    plot_histogram(diff, f'Różnica między histogramami - {img_name}', diff_hist_filename)

# Directories setup
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "images"))
original_images_dir = os.path.join(images_dir, "original_images")

methods = ['lsb', 'rgba', 'dct']

# Loop through images
for img in os.listdir(original_images_dir):
    if img.endswith(('.png', '.jpg', '.jpeg')):
        for method in methods:
            original_image = cv2.imread(os.path.join(original_images_dir, img), cv2.IMREAD_GRAYSCALE)
            stego_image_dir = os.path.join(images_dir, f"{method}_images")
            stego_filename = os.path.join(stego_image_dir, img)
            stego_image = cv2.imread(stego_filename, cv2.IMREAD_GRAYSCALE)

            if original_image is not None and stego_image is not None:
                detect_stego_changes(original_image, stego_image, method, img.split('.')[0])
            else:
                print(f"Warning: Could not load image pair for {img} with method {method}.")