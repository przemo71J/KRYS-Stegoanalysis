import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_histogram(image):
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return histogram

def plot_histogram(histogram, title, color, save_path=None):
    plt.figure(figsize=(8, 6))
    plt.bar(range(256), histogram, color=color)
    plt.title(title)
    plt.xlabel('Intensywność pikseli')
    plt.ylabel('Liczba wystąpień')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()

def detect_stego_changes_rgb(original_image, stego_image, method, img_name):
    base_save_dir = os.path.join(images_dir, "histogram_images_rgb")
    save_dir = os.path.join(base_save_dir, img_name, method)
    os.makedirs(save_dir, exist_ok=True)
    
    channels = ['Blue', 'Green', 'Red']
    colors = ['blue', 'green', 'red']
    
    for i, channel in enumerate(channels):
        original_channel = original_image[:, :, i]
        stego_channel = stego_image[:, :, i]

        original_histogram = calculate_histogram(original_channel)
        stego_histogram = calculate_histogram(stego_channel)

        diff = np.abs(original_histogram - stego_histogram)

        original_hist_filename = os.path.join(save_dir, f"{img_name}_original_histogram_{channel}.png")
        stego_hist_filename = os.path.join(save_dir, f"{img_name}_stego_histogram_{channel}.png")
        diff_hist_filename = os.path.join(save_dir, f"{img_name}_difference_histogram_{channel}.png")

        plot_histogram(original_histogram, f'Histogram obrazu oryginalnego ({channel})', colors[i], original_hist_filename)
        plot_histogram(stego_histogram, f'Histogram obrazu z ukrytymi danymi ({channel})', colors[i], stego_hist_filename)
        plot_histogram(diff, f'Różnica między histogramami ({channel})', colors[i], diff_hist_filename)

        print(f"Suma różnic między histogramami dla kanału {channel}:", np.sum(diff))

# Directories setup
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "images"))
original_images_dir = os.path.join(images_dir, "original_images")

methods = ['lsb', 'rgba', 'dct']

# Loop through images
for img in os.listdir(original_images_dir):
    if img.endswith(('.png', '.jpg', '.jpeg')):
        for method in methods:
            original_image = cv2.imread(os.path.join(original_images_dir, img))
            stego_image_dir = os.path.join(images_dir, f"{method}_images")
            stego_filename = os.path.join(stego_image_dir, img)
            stego_image = cv2.imread(stego_filename)

            if original_image is not None and stego_image is not None:
                detect_stego_changes_rgb(original_image, stego_image, method, img.split('.')[0])
            else:
                print(f"Warning: Could not load image pair for {img} with method {method}.")
