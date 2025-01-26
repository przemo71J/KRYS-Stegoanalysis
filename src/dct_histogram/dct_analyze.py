import os
import numpy as np
from scipy.fftpack import dct
from PIL import Image
import matplotlib.pyplot as plt

def compute_dct_coefficients(image_path, color_mode="L"):
    image = Image.open(image_path).convert(color_mode)
    image_array = np.array(image)

    if color_mode == "RGB":
        dct_coefficients = []
        for c in range(3): 
            channel = image_array[:, :, c]
            dct_channel = compute_dct_for_channel(channel)
            dct_coefficients.append(dct_channel)
        dct_coefficients = np.array(dct_coefficients)
    else:
        dct_coefficients = compute_dct_for_channel(image_array)
    
    return dct_coefficients

def compute_dct_for_channel(channel):
    height, width = channel.shape
    block_size = 8
    dct_coefficients = []

    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = channel[i:i+block_size, j:j+block_size]

            if block.shape != (block_size, block_size):
                continue

            dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')
            dct_coefficients.append(dct_block)

    dct_coefficients = np.array(dct_coefficients)
    return dct_coefficients

def analyze_dct_distribution(dct_coefficients):
    all_coefficients = dct_coefficients.flatten()
    mean = np.mean(all_coefficients)
    std_dev = np.std(all_coefficients)
    return {"mean": mean, "std_dev": std_dev, "coefficients": all_coefficients}

def create_comparison_plot(image_name, base_stats_gray, stego_stats_gray, base_stats_rgb, stego_stats_rgb, stego_folders, output_graph_folder):
    fig, axs = plt.subplots(4, 4, figsize=(20, 10))
    fig.suptitle(f"Porównanie współczynników DCT - {image_name}", fontsize=16)

    # Wykres dla grayscale
    axs[0, 0].hist(base_stats_gray["coefficients"], bins=50, color="blue", alpha=0.7, edgecolor="black")
    axs[0, 0].set_title("Histogram - Oryginał (Grayscale)")
    axs[0, 0].set_xlabel("Wartość współczynnika DCT")
    axs[0, 0].set_ylabel("Liczba wystąpień")

    for idx, stego_stat in enumerate(stego_stats_gray):
        axs[0, idx+1].hist(stego_stat["coefficients"], bins=50, color="green", alpha=0.7, edgecolor="black")
        axs[0, idx+1].set_title(f"Histogram - {os.path.basename(stego_folders[idx])} (Grayscale)")
        axs[0, idx+1].set_xlabel("Wartość współczynnika DCT")
        axs[0, idx+1].set_ylabel("Liczba wystąpień")

    base_hist, bins = np.histogram(base_stats_gray["coefficients"], bins=50)
    for idx, stego_stat in enumerate(stego_stats_gray):
        stego_hist, _ = np.histogram(stego_stat["coefficients"], bins=bins)
        diff_hist = stego_hist - base_hist
        axs[1, idx+1].bar(bins[:-1], diff_hist, width=np.diff(bins), color="red", alpha=0.7, edgecolor="black")
        axs[1, idx+1].set_title(f"Różnice histogramów - {os.path.basename(stego_folders[idx])} (Grayscale)")
        axs[1, idx+1].set_xlabel("Wartość współczynnika DCT")
        axs[1, idx+1].set_ylabel("Różnica liczby wystąpień")

    # Wykres dla RGB
    axs[2, 0].hist(base_stats_rgb["coefficients"], bins=50, color="blue", alpha=0.7, edgecolor="black")
    axs[2, 0].set_title("Histogram - Oryginał (RGB)")
    axs[2, 0].set_xlabel("Wartość współczynnika DCT")
    axs[2, 0].set_ylabel("Liczba wystąpień")

    for idx, stego_stat in enumerate(stego_stats_rgb):
        axs[2, idx+1].hist(stego_stat["coefficients"], bins=50, color="green", alpha=0.7, edgecolor="black")
        axs[2, idx+1].set_title(f"Histogram - {os.path.basename(stego_folders[idx])} (RGB)")
        axs[2, idx+1].set_xlabel("Wartość współczynnika DCT")
        axs[2, idx+1].set_ylabel("Liczba wystąpień")

    base_hist_rgb, bins_rgb = np.histogram(base_stats_rgb["coefficients"], bins=50)
    for idx, stego_stat in enumerate(stego_stats_rgb):
        stego_hist_rgb, _ = np.histogram(stego_stat["coefficients"], bins=bins_rgb)
        diff_hist_rgb = stego_hist_rgb - base_hist_rgb
        axs[3, idx+1].bar(bins_rgb[:-1], diff_hist_rgb, width=np.diff(bins_rgb), color="red", alpha=0.7, edgecolor="black")
        axs[3, idx+1].set_title(f"Różnice histogramów - {os.path.basename(stego_folders[idx])} (RGB)")
        axs[3, idx+1].set_xlabel("Wartość współczynnika DCT")
        axs[3, idx+1].set_ylabel("Różnica liczby wystąpień")

    axs[1, 0].axis("off")  # Usuń pusty wykres
    axs[3, 0].axis("off")  # Usuń pusty wykres
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_path = os.path.join(output_graph_folder, f"{image_name}")
    plt.savefig(output_path)
    plt.close()

def compare_images(base_folder, stego_folders, output_graph_folder):
    base_images = [f for f in os.listdir(base_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(base_images)
    for image_name in base_images:
        print(f"Analiza obrazu: {image_name}")

        base_image_path = os.path.join(base_folder, image_name)
        base_dct_gray = compute_dct_coefficients(base_image_path, color_mode="L")
        base_stats_gray = analyze_dct_distribution(base_dct_gray)
        base_dct_rgb = compute_dct_coefficients(base_image_path, color_mode="RGB")
        base_stats_rgb = analyze_dct_distribution(base_dct_rgb)

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

            stego_dct_gray = compute_dct_coefficients(stego_image_path, color_mode="L")
            stego_stats_gray.append(analyze_dct_distribution(stego_dct_gray))
            stego_dct_rgb = compute_dct_coefficients(stego_image_path, color_mode="RGB")
            stego_stats_rgb.append(analyze_dct_distribution(stego_dct_rgb))

        create_comparison_plot(image_name, base_stats_gray, stego_stats_gray, base_stats_rgb, stego_stats_rgb, stego_folders, output_graph_folder)

def main():
    base_folder = os.path.dirname(os.path.abspath(__file__))
    output_graph_folder = os.path.abspath(os.path.join(base_folder, "..", "..", "images","dct_analysis-histograms"))
    stego_folders = [
        os.path.abspath(os.path.join(base_folder, "..", "..", "images","dct_images")),
        os.path.abspath(os.path.join(base_folder, "..", "..", "images","rgba_images")),
        os.path.abspath(os.path.join(base_folder, "..", "..", "images","lsb_images"))
    ]
    print(stego_folders)

    compare_images(base_folder, stego_folders, output_graph_folder)

if __name__ == "__main__":
    main()
