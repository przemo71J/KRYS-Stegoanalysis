from PIL import Image
import numpy as np
from scipy.stats import chi2

def chi_square_attack(image_path):
    # Load the image
    image = Image.open(image_path)
    image = image.convert('L')  # Convert to grayscale for simplicity

    # Calculate the histogram
    histogram = image.histogram()

    # Define the expected uniform distribution
    total_pixels = sum(histogram)
    expected = [total_pixels / 256] * 256

    # Perform the chi-square test
    chi_square_statistic = sum(((obs - exp) ** 2) / exp for obs, exp in zip(histogram, expected))
    degrees_of_freedom = 255  # 256 bins - 1
    p_value = chi2.sf(chi_square_statistic, degrees_of_freedom)

    # Analyze the result
    if p_value < 0.05:
        print("Significant deviation from expected distribution. Hidden message likely present.")
    else:
        print("No significant deviation from expected distribution. Hidden message unlikely.")

# Example usage
image_path = 'normal.png'
chi_square_attack(image_path)