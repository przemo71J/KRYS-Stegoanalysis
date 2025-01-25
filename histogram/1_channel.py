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
    save_dir = f"histogram_images/{img_name}/{method}"
    os.makedirs(save_dir, exist_ok=True)

    original_histogram = calculate_histogram(original_image)
    stego_histogram = calculate_histogram(stego_image)

    diff = np.abs(original_histogram - stego_histogram)

    original_hist_filename = f"{save_dir}/{img_name}_original_histogram.png"
    stego_hist_filename = f"{save_dir}/{img_name}_stego_histogram.png"
    diff_hist_filename = f"{save_dir}/{img_name}_difference_histogram.png"

    plot_histogram(original_histogram, 'Histogram obrazu oryginalnego - analiza Grayscale', original_hist_filename)
    plot_histogram(stego_histogram, 'Histogram obrazu z ukrytymi danymi - analiza Grayscale', stego_hist_filename)
    plot_histogram(diff, 'Różnica między histogramami - analiza Grayscale', diff_hist_filename)

image_dir = './images/'
methods = ['lsb', 'rgba', 'dct']

for img in os.listdir(image_dir):
    if img.endswith(('.png', '.jpg', '.jpeg')):
        for method in methods:
            original_image = cv2.imread(os.path.join(image_dir, img), cv2.IMREAD_GRAYSCALE)
            stego_filename = os.path.join(f'./{method}_images/', img)
            stego_image = cv2.imread(stego_filename, cv2.IMREAD_GRAYSCALE)
            detect_stego_changes(original_image, stego_image, method, img.split('.')[0])