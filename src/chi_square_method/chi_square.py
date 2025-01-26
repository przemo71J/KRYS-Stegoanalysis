import cv2
import numpy as np
from scipy.stats import chi2
import os

# Function to calculate LSB counts for a single channel
def calculate_lsb_counts(channel_data):
    lsb_0_count = 0
    lsb_1_count = 0

    # Count LSBs
    for value in channel_data:
        lsb = value & 1
        if lsb == 0:
            lsb_0_count += 1
        else:
            lsb_1_count += 1

    return lsb_0_count, lsb_1_count

# Function to calculate chi-square statistic given LSB counts
def calculate_chi_square(lsb_0_count, lsb_1_count):
    # Expected frequency (uniform distribution)
    expected_count = (lsb_0_count + lsb_1_count) / 2

    # Calculate chi-square statistic
    chi_square_stat = ((lsb_0_count - expected_count) ** 2 / expected_count) + \
                      ((lsb_1_count - expected_count) ** 2 / expected_count)

    return chi_square_stat

# Separate RGB channels
def extract_channels(data):
    r_channel = [pixel[0] for pixel in data]
    g_channel = [pixel[1] for pixel in data]
    b_channel = [pixel[2] for pixel in data]
    return r_channel, g_channel, b_channel

# Directories setup from the original script
current_dir = os.path.dirname(os.path.abspath(__file__))
images_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "images"))
original_images_dir = os.path.join(images_dir, "original_images")
methods = ['lsb', 'rgba', 'dct']  # Methods you are using

# Critical value for chi-square test (significance level 0.05, df=1)
critical_value = chi2.ppf(0.95, df=1)

# Function to process images, calculate chi-square and interpret results
def process_images(img, method):
    # Paths for original and stego images based on method
    original_image_path = os.path.join(original_images_dir, img)
    stego_image_dir = os.path.join(images_dir, f"{method}_images")
    stego_image_path = os.path.join(stego_image_dir, img)
    
    # Load the original and stego images
    original_image = cv2.imread(original_image_path)
    stego_image = cv2.imread(stego_image_path)

    if original_image is not None and stego_image is not None:
        # Convert images to RGB
        original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        stego_rgb = cv2.cvtColor(stego_image, cv2.COLOR_BGR2RGB)

        # Flatten the image data to extract all pixels
        height, width, _ = original_rgb.shape
        original_data = original_rgb.reshape((height * width, 3))

        height, width, _ = stego_rgb.shape
        stego_data = stego_rgb.reshape((height * width, 3))

        # Extract raw and secret channels
        original_r, original_g, original_b = extract_channels(original_data)
        stego_r, stego_g, stego_b = extract_channels(stego_data)

        # Calculate LSB counts for each channel
        original_r_lsb_0, original_r_lsb_1 = calculate_lsb_counts(original_r)
        original_g_lsb_0, original_g_lsb_1 = calculate_lsb_counts(original_g)
        original_b_lsb_0, original_b_lsb_1 = calculate_lsb_counts(original_b)

        stego_r_lsb_0, stego_r_lsb_1 = calculate_lsb_counts(stego_r)
        stego_g_lsb_0, stego_g_lsb_1 = calculate_lsb_counts(stego_g)
        stego_b_lsb_0, stego_b_lsb_1 = calculate_lsb_counts(stego_b)

        # Calculate chi-square statistics for each channel
        original_r_chi = calculate_chi_square(original_r_lsb_0, original_r_lsb_1)
        original_g_chi = calculate_chi_square(original_g_lsb_0, original_g_lsb_1)
        original_b_chi = calculate_chi_square(original_b_lsb_0, original_b_lsb_1)

        stego_r_chi = calculate_chi_square(stego_r_lsb_0, stego_r_lsb_1)
        stego_g_chi = calculate_chi_square(stego_g_lsb_0, stego_g_lsb_1)
        stego_b_chi = calculate_chi_square(stego_b_lsb_0, stego_b_lsb_1)

        # Combine LSB counts for all channels
        original_combined_lsb_0 = original_r_lsb_0 + original_g_lsb_0 + original_b_lsb_0
        original_combined_lsb_1 = original_r_lsb_1 + original_g_lsb_1 + original_b_lsb_1
        stego_combined_lsb_0 = stego_r_lsb_0 + stego_g_lsb_0 + stego_b_lsb_0
        stego_combined_lsb_1 = stego_r_lsb_1 + stego_g_lsb_1 + stego_b_lsb_1

        # Calculate chi-square statistic for combined channels
        original_combined_chi = calculate_chi_square(original_combined_lsb_0, original_combined_lsb_1)
        stego_combined_chi = calculate_chi_square(stego_combined_lsb_0, stego_combined_lsb_1)

        # Output results
        print(f"\nProcessing {img} with method {method}:")
        print(f"Original Image Chi-Square Statistics:")
        print(f"  R Channel: {original_r_chi}, G Channel: {original_g_chi}, B Channel: {original_b_chi}")
        print(f"  Combined Channels: {original_combined_chi}")
        print(f"Stego Image Chi-Square Statistics:")
        print(f"  R Channel: {stego_r_chi}, G Channel: {stego_g_chi}, B Channel: {stego_b_chi}")
        print(f"  Combined Channels: {stego_combined_chi}")
        
        # Interpret results for original and stego data
        print("\nInterpretation for Original Image:")
        interpret_results("R", original_r_chi)
        interpret_results("G", original_g_chi)
        interpret_results("B", original_b_chi)
        interpret_results("Combined", original_combined_chi)

        print("\nInterpretation for Stego Image:")
        interpret_results("R", stego_r_chi)
        interpret_results("G", stego_g_chi)
        interpret_results("B", stego_b_chi)
        interpret_results("Combined", stego_combined_chi)

    else:
        print(f"Warning: Could not load image pair for {img} with method {method}.")

# Function to interpret the chi-square test results
def interpret_results(channel_name, chi_square_stat):
    if chi_square_stat > critical_value:
        print(f"Possible steganography detected in {channel_name} channel!")
    else:
        print(f"No steganography detected in {channel_name} channel.")

# Loop through images and methods
for img in os.listdir(original_images_dir):
    if img.endswith(('.png', '.jpg', '.jpeg')):
        for method in methods:
            process_images(img, method)
