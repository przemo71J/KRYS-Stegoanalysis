import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    histogram, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    return histogram

def plot_histogram(histogram, title, color):
    plt.figure(figsize=(8, 6))
    plt.bar(range(256), histogram, color=color)
    plt.title(title)
    plt.xlabel('Intensywność pikseli')
    plt.ylabel('Liczba wystąpień')
    plt.show()

def detect_stego_changes_rgb(original_image, stego_image):
    channels = ['Blue', 'Green', 'Red']
    colors = ['blue', 'green', 'red']
    
    for i, channel in enumerate(channels):
        # Wyciągnięcie pojedynczego kanału
        original_channel = original_image[:, :, i]
        stego_channel = stego_image[:, :, i]

        # Obliczanie histogramów dla danego kanału
        original_histogram = calculate_histogram(original_channel)
        stego_histogram = calculate_histogram(stego_channel)

        # Porównanie histogramów
        diff = np.abs(original_histogram - stego_histogram)

        # Rysowanie wyników
        plot_histogram(original_histogram, f'Histogram obrazu oryginalnego ({channel})', colors[i])
        plot_histogram(stego_histogram, f'Histogram obrazu z ukrytymi danymi ({channel})', colors[i])
        plot_histogram(diff, f'Różnica między histogramami ({channel})', colors[i])

        # Wyświetlenie metryk różnicowych
        print(f"Suma różnic między histogramami dla kanału {channel}:", np.sum(diff))

# Wczytywanie obrazów
img = 'man.png'
method = 'dct'
path = 'C:/Users/prpustel/Desktop/PROJEKT_KRYS/KRYS-Stegoanalysis/'
original_image = cv2.imread(path + 'images/'+img)
stego_filename = path + method+'_images/'+img.split('.')[0] + img.split('.')[1]
stego_image = cv2.imread(stego_filename)

# Wykrywanie zmian dla kanałów RGB
detect_stego_changes_rgb(original_image, stego_image)
