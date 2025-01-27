import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_histogram(image):
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return histogram

def plot_histograms_9(original_histograms, stego_histograms, diffs, img_name, save_path=None):

    methods = ['LSB', 'RGBA', 'DCT']
    color='darkorange'
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Histogramy dla obrazu: {img_name}", fontsize=16)  # Global title

    for i, method in enumerate(methods):
        # Original histogram
        axes[i, 0].bar(range(256), original_histograms[i], color=color)
        axes[i, 0].set_title(f'Oryginalny - {method}')
        axes[i, 0].set_xlabel('Intensywność pikseli')
        axes[i, 0].set_ylabel('Liczba wystąpień')

        # Stego histogram
        axes[i, 1].bar(range(256), stego_histograms[i], color=color)
        axes[i, 1].set_title(f'Stego - {method}')
        axes[i, 1].set_xlabel('Intensywność pikseli')
        axes[i, 1].set_ylabel('Liczba wystąpień')

        # Difference histogram
        axes[i, 2].bar(range(256), diffs[i], color=color)
        axes[i, 2].set_title(f'Różnica - {method}')
        axes[i, 2].set_xlabel('Intensywność pikseli')
        axes[i, 2].set_ylabel('Liczba wystąpień')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for global title
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

def detect_stego_changes_9(original_image, stego_images, img_name):
    base_save_dir = os.path.join(images_dir, "histogram_images_gray")
    save_dir = os.path.join(base_save_dir, img_name)
    os.makedirs(save_dir, exist_ok=True)

    original_histograms = []
    stego_histograms = []
    diffs = []

    methods = ['lsb', 'rgba', 'dct']
    for method, stego_image in zip(methods, stego_images):
        original_histogram = calculate_histogram(original_image)
        print("Original histogram:\n",original_histogram)
        stego_histogram = calculate_histogram(stego_image)
        print("Stego histogram:\n",stego_histogram)
        diff = np.abs(original_histogram - stego_histogram)
        print("Difference:\n",diff)

        original_histograms.append(original_histogram)
        stego_histograms.append(stego_histogram)
        diffs.append(diff)

    hist_filename = os.path.join(save_dir, f"{img_name}_combined_histograms.png")
    plot_histograms_9(original_histograms, stego_histograms, diffs, img_name, hist_filename)

# Directories setup
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "images"))
original_images_dir = os.path.join(images_dir, "original_images")

methods = ['lsb', 'rgba', 'dct']

for img in os.listdir(original_images_dir):
    if img.endswith(('.png', '.jpg', '.jpeg')):
        original_image = cv2.imread(os.path.join(original_images_dir, img), cv2.IMREAD_GRAYSCALE)
        stego_images = []

        for method in methods:
            stego_image_dir = os.path.join(images_dir, f"{method}_images")
            stego_filename = os.path.join(stego_image_dir, img)
            stego_image = cv2.imread(stego_filename, cv2.IMREAD_GRAYSCALE)
            if stego_image is not None:
                stego_images.append(stego_image)
            else:
                print(f"Warning: Could not load stego image for {img} with method {method}.")
                break

        if original_image is not None and len(stego_images) == len(methods):
            detect_stego_changes_9(original_image, stego_images, img.split('.')[0])
        else:
            print(f"Warning: Could not load all image pairs for {img}.")
