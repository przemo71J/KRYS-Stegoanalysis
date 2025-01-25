import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return histogram

def plot_histogram(histogram, title):
    plt.figure(figsize=(8, 6))
    plt.bar(range(256), histogram, color='red')
    plt.title(title)
    plt.xlabel('Intensywność pikseli')
    plt.ylabel('Liczba wystąpień')
    plt.show()

def detect_stego_changes(original_image, stego_image):

    # Oblicz histogramy
    original_histogram = calculate_histogram(original_image)
    stego_histogram = calculate_histogram(stego_image)

    # Porównaj histogramy
    diff = np.abs(original_histogram - stego_histogram)

    # Rysowanie wyników
    plot_histogram(original_histogram, 'Histogram obrazu oryginalnego')
    plot_histogram(stego_histogram, 'Histogram obrazu z ukrytymi danymi')
    plot_histogram(diff, 'Różnica między histogramami')

    # Wyświetlenie metryk różnicowych
    print("Suma różnic między histogramami:", np.sum(diff))

img = 'neon.png'
path = 'C:/Users/prpustel/Desktop/PROJEKT_KRYS/KRYS-Stegoanalysis/'
method = 'dct'
original_image = cv2.imread(path + 'images/'+img, cv2.IMREAD_GRAYSCALE)
stego_filename = path + method+"_images/"+ img.split('.')[0] +'.'+ img.split('.')[1]
stego_image = cv2.imread(stego_filename, cv2.IMREAD_GRAYSCALE)

# Wykrywanie zmian
detect_stego_changes(original_image, stego_image)
