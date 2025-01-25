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
    save_dir = f"histogram_images_rgb/{img_name}/{method}"
    os.makedirs(save_dir, exist_ok=True)
    
    channels = ['Blue', 'Green', 'Red']
    colors = ['blue', 'green', 'red']
    
    for i, channel in enumerate(channels):
        original_channel = original_image[:, :, i]
        stego_channel = stego_image[:, :, i]

        original_histogram = calculate_histogram(original_channel)
        stego_histogram = calculate_histogram(stego_channel)

        diff = np.abs(original_histogram - stego_histogram)

        original_hist_filename = f"{save_dir}/{img_name}_original_histogram_{channel}.png"
        stego_hist_filename = f"{save_dir}/{img_name}_stego_histogram_{channel}.png"
        diff_hist_filename = f"{save_dir}/{img_name}_difference_histogram_{channel}.png"

        plot_histogram(original_histogram, f'Histogram obrazu oryginalnego ({channel})', colors[i], original_hist_filename)
        plot_histogram(stego_histogram, f'Histogram obrazu z ukrytymi danymi ({channel})', colors[i], stego_hist_filename)
        plot_histogram(diff, f'Różnica między histogramami ({channel})', colors[i], diff_hist_filename)

        print(f"Suma różnic między histogramami dla kanału {channel}:", np.sum(diff))

image_dir = './images/'
methods = ['lsb', 'rgba', 'dct']

for img in os.listdir(image_dir):
    if img.endswith(('.png', '.jpg', '.jpeg')):
        for method in methods:
            original_image = cv2.imread(os.path.join(image_dir, img))
            stego_filename = os.path.join(f'./{method}_images/', img)
            stego_image = cv2.imread(stego_filename)
            detect_stego_changes_rgb(original_image, stego_image, method, img.split('.')[0])
