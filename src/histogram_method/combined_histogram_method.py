import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def calculate_histogram(image):
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return histogram

def plot_histogram(histogram, title, color, save_path):
    plt.figure(figsize=(6, 4))
    plt.bar(range(256), histogram, color=color)
    plt.title(title)
    plt.xlabel('Intensywność pikseli')
    plt.ylabel('Liczba wystąpień')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_combined_histograms_grayscale(original_image, stego_image, method, img_name):
    base_save_dir = os.path.join(images_dir, "histograms_gray")
    save_dir = os.path.join(base_save_dir, method)
    os.makedirs(save_dir, exist_ok=True)

    original_histogram = calculate_histogram(original_image)
    stego_histogram = calculate_histogram(stego_image)
    diff_histogram = calculate_histogram(cv2.absdiff(original_image, stego_image))

    plot_histogram(original_histogram, 'Histogram Oryginalny', 'gray',
                   os.path.join(save_dir, f"{img_name}_original_histogram.png"))
    plot_histogram(stego_histogram, 'Histogram Stego', 'gray',
                   os.path.join(save_dir, f"{img_name}_stego_histogram.png"))
    plot_histogram(diff_histogram, 'Histogram Różnica', 'gray',
                   os.path.join(save_dir, f"{img_name}_difference_histogram.png"))

def plot_combined_histograms_rgb(original_image, stego_image, method, img_name):
    base_save_dir = os.path.join(images_dir, "histograms_rgb")
    save_dir = os.path.join(base_save_dir, method)
    os.makedirs(save_dir, exist_ok=True)

    channels = ['R', 'G', 'B']
    colors = ['red', 'green', 'blue']

    for i, channel in enumerate(channels):
        original_channel = original_image[:, :, 2 - i]
        stego_channel = stego_image[:, :, 2 - i]
        diff_channel = cv2.absdiff(original_channel, stego_channel)

        original_histogram = calculate_histogram(original_channel)
        stego_histogram = calculate_histogram(stego_channel)
        diff_histogram = calculate_histogram(diff_channel)

        plot_histogram(original_histogram, f'Kanał {channel} - Oryginalny', colors[i],
                       os.path.join(save_dir, f"{img_name}_original_{channel}.png"))
        plot_histogram(stego_histogram, f'Kanał {channel} - Stego', colors[i],
                       os.path.join(save_dir, f"{img_name}_stego_{channel}.png"))
        plot_histogram(diff_histogram, f'Kanał {channel} - Różnica', colors[i],
                       os.path.join(save_dir, f"{img_name}_difference_{channel}.png"))

# Directories setup
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "images"))
original_images_dir = os.path.join(images_dir, "original_images")

methods = ['lsb', 'rgba', 'dct']

for img in os.listdir(original_images_dir):
    if img.endswith(('.png', '.jpg', '.jpeg')):
        original_image_gray = cv2.imread(os.path.join(original_images_dir, img), cv2.IMREAD_GRAYSCALE)
        original_image_rgb = cv2.imread(os.path.join(original_images_dir, img))

        for method in methods:
            stego_image_dir = os.path.join(images_dir, f"{method}_images")
            stego_image_gray = cv2.imread(os.path.join(stego_image_dir, img), cv2.IMREAD_GRAYSCALE)
            stego_image_rgb = cv2.imread(os.path.join(stego_image_dir, img))

            if original_image_gray is not None and stego_image_gray is not None:
                plot_combined_histograms_grayscale(original_image_gray, stego_image_gray, method, img.split('.')[0])

            if original_image_rgb is not None and stego_image_rgb is not None:
                plot_combined_histograms_rgb(original_image_rgb, stego_image_rgb, method, img.split('.')[0])
