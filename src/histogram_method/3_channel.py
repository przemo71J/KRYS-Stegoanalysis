import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_histogram(image):
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return histogram

def plot_histograms_rgb_9(original_histograms, stego_histograms, diffs, img_name, method, save_path=None):
    """
    Plots 9 subplots: 3 rows (channels: Blue, Green, Red) x 3 columns (Original, Stego, Difference).
    """
    channels = ['Niebieski', 'Zielony', 'Czerwony']
    colors = ['blue', 'green', 'red']

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"Histogramy dla obrazu: {img_name} - Metoda: {method.upper()}", fontsize=16)  # Global title

    for i, channel in enumerate(channels):
        # Original histogram
        axes[i, 0].bar(range(256), original_histograms[i], color=colors[i])
        axes[i, 0].set_title(f'Oryginalny - {channel}')
        axes[i, 0].set_xlabel('Intensywność pikseli')
        axes[i, 0].set_ylabel('Liczba wystąpień')

        # Stego histogram
        axes[i, 1].bar(range(256), stego_histograms[i], color=colors[i])
        axes[i, 1].set_title(f'Stego - {channel}')
        axes[i, 1].set_xlabel('Intensywność pikseli')
        axes[i, 1].set_ylabel('Liczba wystąpień')

        # Difference histogram
        axes[i, 2].bar(range(256), diffs[i], color='orange')  # Difference uses orange
        axes[i, 2].set_title(f'Różnica - {channel}')
        axes[i, 2].set_xlabel('Intensywność pikseli')
        axes[i, 2].set_ylabel('Liczba wystąpień')

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout for global title
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

def detect_stego_changes_rgb_9(original_image, stego_image, img_name, method):
    base_save_dir = os.path.join(images_dir, "histogram_images_rgb")
    save_dir = os.path.join(base_save_dir, img_name, method)
    os.makedirs(save_dir, exist_ok=True)

    original_histograms = []
    stego_histograms = []
    diffs = []

    channels = ['Niebieski', 'Zielony', 'Czerwony']
    for i in range(len(channels)):  # Iterate through Blue, Green, and Red channels
        original_channel = original_image[:, :, i]
        stego_channel = stego_image[:, :, i]

        original_histogram = calculate_histogram(original_channel)
        stego_histogram = calculate_histogram(stego_channel)
        diff = np.abs(original_histogram - stego_histogram)

        original_histograms.append(original_histogram)
        stego_histograms.append(stego_histogram)
        diffs.append(diff)

    hist_filename = os.path.join(save_dir, f"{img_name}_histograms_{method}.png")
    plot_histograms_rgb_9(original_histograms, stego_histograms, diffs, img_name, method, hist_filename)

# Directories setup
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "images"))
original_images_dir = os.path.join(images_dir, "original_images")

methods = ['lsb', 'rgba', 'dct']

for img in os.listdir(original_images_dir):
    if img.endswith(('.png', '.jpg', '.jpeg')):
        original_image = cv2.imread(os.path.join(original_images_dir, img))

        for method in methods:
            stego_image_dir = os.path.join(images_dir, f"{method}_images")
            stego_filename = os.path.join(stego_image_dir, img)
            stego_image = cv2.imread(stego_filename)

            if original_image is not None and stego_image is not None:
                detect_stego_changes_rgb_9(original_image, stego_image, img.split('.')[0], method)
            else:
                print(f"Warning: Could not load image pair for {img} with method {method}.")
