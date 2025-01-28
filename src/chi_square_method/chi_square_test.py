import os
import cv2
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

def perform_chi_square_test(original_image, stego_image, num_bins=256):
    original_hist, bins = np.histogram(original_image.flatten(), bins=256, range=[0, 256])
    stego_hist, _ = np.histogram(stego_image.flatten(), bins=256, range=[0, 256])
    
    original_hist_corrected = original_hist + 0.5
    stego_hist_corrected = stego_hist + 0.5
    
    chi2, p, _, _ = chi2_contingency([original_hist_corrected, stego_hist_corrected], correction=False)
    return chi2, p

def plot_difference_histograms(original_image, stego_images, methods, img_name):
    save_dir = os.path.join(images_dir, "chi_square_histograms", img_name)
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(methods), figsize=(12, 4))
    if len(methods) == 1:
        axes = [axes]
    
    for ax, method, stego_image in zip(axes, methods, stego_images):
        diff_image = (original_image.astype(np.int16) - stego_image.astype(np.int16))
        ax.hist(diff_image.flatten(), bins=512, range=[-5, 5], color='r', alpha=0.7)
        ax.set_title(f'Difference Histogram ({method})')
        ax.set_xlabel('Pixel Intensity Difference')
        ax.set_ylabel('Frequency')
    
    hist_filename = os.path.join(save_dir, f"{img_name}_combined_difference_histograms.png")
    plt.tight_layout()
    plt.savefig(hist_filename)
    plt.close()

def detect_stego_changes_9(original_image, stego_images, methods, img_name):

    for method, stego_image in zip(methods, stego_images):
        chi2_gray, p_gray = perform_chi_square_test(original_image, stego_image)
        stego_detected = "Yes" if p_gray < 0.05 else "No"
        print(f"{img_name:<20}{method:<10}{chi2_gray:<15.5f}{p_gray:<15.5e}{stego_detected}")
    plot_difference_histograms(original_image, stego_images, methods, img_name)
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
            detect_stego_changes_9(original_image, stego_images, methods, img.split('.')[0])
        else:
            print(f"Warning: Could not load all image pairs for {img}.")
