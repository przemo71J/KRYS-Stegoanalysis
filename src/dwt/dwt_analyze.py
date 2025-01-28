import os
import numpy as np
import pywt
from PIL import Image
import matplotlib.pyplot as plt

def compute_dwt_coefficients(image_path, color_mode="L", wavelet="haar"):
    image = Image.open(image_path).convert(color_mode)
    image_array = np.array(image)

    if color_mode == "RGB":
        dwt_coefficients = []
        for c in range(3):  
            channel = image_array[:, :, c]
            coeffs = compute_dwt_for_channel(channel, wavelet)
            dwt_coefficients.append(coeffs)
        dwt_coefficients = np.array(dwt_coefficients)

    else:
        dwt_coefficients = compute_dwt_for_channel(image_array, wavelet)

    return dwt_coefficients

def compute_dwt_for_channel(channel, wavelet):
    height = channel.shape[0]
    width = channel.shape[1]
    block_size = 8
    dwt_coefficients = []

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = channel[i:i+block_size, j:j+block_size]

            if block.shape != (block_size, block_size):
                continue

            coeffs2d = pywt.dwt2(block, wavelet=wavelet)
            cA, (cH, cV, cD) = coeffs2d
            dwt_coefficients.append((cA, cH, cV, cD))

    return dwt_coefficients

def analyze_dwt_distribution(dwt_coefficients):
    all_coefficients = []

    for coeff_set in dwt_coefficients:
        all_coefficients.extend(np.array(coeff_set[0]).flatten()) 
        all_coefficients.extend(np.array(coeff_set[1]).flatten())
        all_coefficients.extend(np.array(coeff_set[2]).flatten())
        all_coefficients.extend(np.array(coeff_set[3]).flatten())

    all_coefficients = np.array(all_coefficients)
    mean = np.mean(all_coefficients)
    std_dev = np.std(all_coefficients)
    return {"mean": mean, "std_dev": std_dev, "coefficients": all_coefficients}

def create_comparison_plot(image_name, base_stats_gray, stego_stats_gray, base_stats_rgb, stego_stats_rgb, stego_folders, output_graph_folder):
    fig, axs = plt.subplots(4, 4, figsize=(20, 10))
    fig.suptitle(f"Porównanie współczynników DWT - {image_name}", fontsize=16)

    # Wykres dla grayscale
    axs[0, 0].hist(base_stats_gray["coefficients"], bins=50, color="blue", alpha=0.7, edgecolor="black")
    axs[0, 0].set_title("Histogram - Oryginał (Grayscale)")
    axs[0, 0].set_xlabel("Wartość współczynnika DWT")
    axs[0, 0].set_ylabel("Liczba wystąpień")
    axs[0, 0].set_yscale("log")

    for idx, stego_stat in enumerate(stego_stats_gray):
        axs[0, idx+1].hist(stego_stat["coefficients"], bins=50, color="green", alpha=0.7, edgecolor="black")
        axs[0, idx+1].set_title(f"Histogram - {os.path.basename(stego_folders[idx])} (Grayscale)")
        axs[0, idx+1].set_xlabel("Wartość współczynnika DWT")
        axs[0, idx+1].set_ylabel("Liczba wystąpień")
        axs[0, idx+1].set_yscale("log")

    base_hist, bins = np.histogram(base_stats_gray["coefficients"], bins=50)
    for idx, stego_stat in enumerate(stego_stats_gray):
        stego_hist, _ = np.histogram(stego_stat["coefficients"], bins=bins)
        diff_hist = stego_hist - base_hist
        axs[1, idx+1].bar(bins[:-1], diff_hist, width=np.diff(bins), color="red", alpha=0.7, edgecolor="black")
        axs[1, idx+1].set_title(f"Różnice histogramów - {os.path.basename(stego_folders[idx])} (Grayscale)")
        axs[1, idx+1].set_xlabel("Wartość współczynnika DWT")
        axs[1, idx+1].set_ylabel("Różnica liczby wystąpień")
        axs[1, idx+1].set_yscale("log")

    # Wykres dla RGB
    axs[2, 0].hist(base_stats_rgb["coefficients"], bins=50, color="blue", alpha=0.7, edgecolor="black")
    axs[2, 0].set_title("Histogram - Oryginał (RGB)")
    axs[2, 0].set_xlabel("Wartość współczynnika DWT")
    axs[2, 0].set_ylabel("Liczba wystąpień")
    axs[2, 0].set_yscale("log")

    for idx, stego_stat in enumerate(stego_stats_rgb):
        axs[2, idx+1].hist(stego_stat["coefficients"], bins=50, color="green", alpha=0.7, edgecolor="black")
        axs[2, idx+1].set_title(f"Histogram - {os.path.basename(stego_folders[idx])} (RGB)")
        axs[2, idx+1].set_xlabel("Wartość współczynnika DWT")
        axs[2, idx+1].set_ylabel("Liczba wystąpień")
        axs[2, idx+1].set_yscale("log")

    base_hist_rgb, bins_rgb = np.histogram(base_stats_rgb["coefficients"], bins=50)
    for idx, stego_stat in enumerate(stego_stats_rgb):
        stego_hist_rgb, _ = np.histogram(stego_stat["coefficients"], bins=bins_rgb)
        diff_hist_rgb = stego_hist_rgb - base_hist_rgb
        axs[3, idx+1].bar(bins_rgb[:-1], diff_hist_rgb, width=np.diff(bins_rgb), color="red", alpha=0.7, edgecolor="black")
        axs[3, idx+1].set_title(f"Różnice histogramów - {os.path.basename(stego_folders[idx])} (RGB)")
        axs[3, idx+1].set_xlabel("Wartość współczynnika DWT")
        axs[3, idx+1].set_ylabel("Różnica liczby wystąpień")
        axs[3, idx+1].set_yscale("log")

    axs[1, 0].axis("off")  # Usuń pusty wykres
    axs[3, 0].axis("off")  # Usuń pusty wykres
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = os.path.join(output_graph_folder, f"{image_name}")
    plt.savefig(output_path)
    plt.close()


def compare_images(base_folder, stego_folders, output_graph_folder):
    base_images = [f for f in os.listdir(base_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

    for image_name in base_images:
        print(f"Analiza obrazu: {image_name}")

        base_image_path = os.path.join(base_folder, image_name)
        base_dwt_gray = compute_dwt_coefficients(base_image_path, color_mode="L")
        base_stats_gray = analyze_dwt_distribution(base_dwt_gray)
        base_dwt_rgb = compute_dwt_coefficients(base_image_path, color_mode="RGB")
        base_stats_rgb = analyze_dwt_distribution(base_dwt_rgb)

        stego_stats_gray = []
        stego_stats_rgb = []

        for stego_folder in stego_folders:
            base_name = os.path.splitext(image_name)[0]
            stego_image_path = os.path.join(stego_folder, f"{base_name}.png")

            if not os.path.exists(stego_image_path):
                print(f"Brak obrazu {image_name} w folderze {stego_folder}")
                stego_stats_gray.append({"coefficients": [], "mean": 0, "std_dev": 0})
                stego_stats_rgb.append({"coefficients": [], "mean": 0, "std_dev": 0})
                continue

            stego_dwt_gray = compute_dwt_coefficients(stego_image_path, color_mode="L")
            stego_stats_gray.append(analyze_dwt_distribution(stego_dwt_gray))
            stego_dwt_rgb = compute_dwt_coefficients(stego_image_path, color_mode="RGB")
            stego_stats_rgb.append(analyze_dwt_distribution(stego_dwt_rgb))

        create_comparison_plot(image_name, base_stats_gray, stego_stats_gray, base_stats_rgb, stego_stats_rgb, stego_folders, output_graph_folder)

def main():
    base_folder = r"C:\Users\hgolebio\OneDrive - Eltel Group Corporation\Documents\VSCode\Mouse\KRYS"
    output_graph_folder = r"C:\Users\hgolebio\OneDrive - Eltel Group Corporation\Documents\VSCode\Mouse\KRYS\DWT ANALYZE"
    stego_folders = [
        r"C:\Users\hgolebio\OneDrive - Eltel Group Corporation\Documents\VSCode\Mouse\KRYS\KRYS DCT",
        r"C:\Users\hgolebio\OneDrive - Eltel Group Corporation\Documents\VSCode\Mouse\KRYS\KRYS RGBA",
        r"C:\Users\hgolebio\OneDrive - Eltel Group Corporation\Documents\VSCode\Mouse\KRYS\KRYS LSB"
    ]

    compare_images(base_folder, stego_folders, output_graph_folder)

if __name__ == "__main__":
    main()
