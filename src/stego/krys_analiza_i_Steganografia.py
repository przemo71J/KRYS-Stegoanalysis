import os
from PIL import Image, ExifTags
import pandas as pd
import numpy as np
import cv2

def analyze_images_in_folder(folder_path, output_csv_path, latex_output_path):
    data = []
    latex_content = []

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)

        if not os.path.isfile(file_path):
            continue
        try:
            with Image.open(file_path) as img:
                file_size = os.path.getsize(file_path) 
                width, height = img.size 
                mode = img.mode 
                format = img.format
                grayscale_img = img.convert("L")
                histogram = grayscale_img.histogram()
                histogram_mean = np.mean(histogram)
                histogram_std = np.std(histogram)

                exif_data = {}
                if hasattr(img, '_getexif') and img._getexif() is not None:
                    for tag, value in img._getexif().items():
                        tag_name = ExifTags.TAGS.get(tag, tag)
                        exif_data[tag_name] = value

                data.append({
                    "File Name": file_name,
                    "Width": width,
                    "Height": height,
                    "Mode": mode,
                    "Format": format,
                    "File Size (Bytes)": file_size,
                    "Histogram Mean": histogram_mean,
                    "Histogram Std Dev": histogram_std,
                    "EXIF Data": exif_data
                })

                exif_text = ', '.join([f"{key}: {value}" for key, value in exif_data.items()]) if exif_data else "Brak metadanych EXIF"
                latex_content.append(f"""
                \section*{{Informacje o obrazie: {file_name}}}
                \begin{{itemize}}
                    \item Wymiary: {width} x {height}
                    \item Tryb: {mode}
                    \item Format: {format}
                    \item Rozmiar pliku: {file_size} bajtów
                    \item Średnia jasności histogramu: {histogram_mean:.2f}
                    \item Odchylenie standardowe jasności: {histogram_std:.2f}
                    \item Metadane EXIF: {exif_text}
                \end{{itemize}}
                """)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_csv_path, index=False)

    with open(latex_output_path, 'w', encoding='utf-8') as latex_file:
        latex_file.write(r"\documentclass{article}\n\usepackage[utf8]{inputenc}\n\begin{document}\n")
        latex_file.write(r"\section*{Raport analizy obrazów}\n")
        latex_file.write(r"\n".join(latex_content))
        latex_file.write(r"\n\end{document}")

def hide_data_dct(image_path, output_path, message, block_size=8):
    image = cv2.imread(image_path)  
    (h, w, c) = image.shape  

    binary_message = ''.join(format(ord(char), '08b') for char in message) + '1111111111111110'  # Znacznik końca
    binary_index = 0

    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            if binary_index >= len(binary_message):
                break

            block = image[i:i+block_size, j:j+block_size, :]
            block_h, block_w, _ = block.shape

            if block_h != block_size or block_w != block_size:
                continue

            for channel in range(3):  
                dct_block = cv2.dct(np.float32(block[:, :, channel]))

                dct_block[block_size-1, block_size-1] = dct_block[block_size-1, block_size-1] - (dct_block[block_size-1, block_size-1] % 2) + int(binary_message[binary_index])
                binary_index += 1

                idct_block = cv2.idct(dct_block)
                block[:, :, channel] = np.uint8(np.clip(idct_block, 0, 255))
            image[i:i+block_size, j:j+block_size, :] = block  

    cv2.imwrite(output_path, image)  

def hide_data_alpha(image_path, output_path, message):
    image = Image.open(image_path).convert("RGBA")
    pixels = image.load()

    binary_message = ''.join(format(ord(char), '08b') for char in message) + '1111111111111110'  
    binary_index = 0

    for y in range(image.height):
        for x in range(image.width):
            if binary_index >= len(binary_message):
                break
            r, g, b, a = pixels[x, y]
            a = a & ~1 | int(binary_message[binary_index])  # Modyfikujemy LSB kanału alfa
            binary_index += 1
            pixels[x, y] = (r, g, b, a)

    image.save(output_path)

from PIL import Image

def hide_data_lsb(image_path, output_path, message):
    image = Image.open(image_path)
    pixels = image.load()

    binary_message = ''.join(format(ord(char), '08b') for char in message) + '1111111111111110' 
    binary_index = 0

    for y in range(image.height):
        for x in range(image.width):
            if binary_index >= len(binary_message):
                break
            pixel = list(pixels[x, y])  
            for i in range(3): 
                if binary_index < len(binary_message):
                    pixel[i] = pixel[i] & ~1 | int(binary_message[binary_index])
                    binary_index += 1
            pixels[x, y] = tuple(pixel)

    image.save(output_path)


def hide_messages_in_folder(input_folder, output_folder_dct, output_folder_alpha, output_folder_lsb, message):
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        if not os.path.isfile(file_path):
            continue
        try:
            base_name = os.path.splitext(file_name)[0]
            dct_output_path = os.path.join(output_folder_dct, f"{base_name}.png")
            alpha_output_path = os.path.join(output_folder_alpha, f"{base_name}.png")
            lsb_output_path = os.path.join(output_folder_lsb, f"{base_name}.png")

            hide_data_dct(file_path, dct_output_path, message)

            hide_data_alpha(file_path, alpha_output_path, message)

            hide_data_lsb(file_path, lsb_output_path, message)

        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

def analyze_and_compare_images(input_folder, output_folder_dct, output_folder_alpha, output_folder_lsb, output_comparison_csv_path, output_comparison_latex_path):
    data = []
    latex_content = []
    
    def analyze_image(file_path):
        with Image.open(file_path) as img:
            file_size = os.path.getsize(file_path) 
            width, height = img.size 
            mode = img.mode 
            format = img.format
            grayscale_img = img.convert("L")
            histogram = grayscale_img.histogram()
            histogram_mean = np.mean(histogram)
            histogram_std = np.std(histogram)

            exif_data = {}
            if hasattr(img, '_getexif') and img._getexif() is not None:
                for tag, value in img._getexif().items():
                    tag_name = ExifTags.TAGS.get(tag, tag)
                    exif_data[tag_name] = value

            return {
                "File Size (Bytes)": file_size,
                "Width": width,
                "Height": height,
                "Mode": mode,
                "Format": format,
                "Histogram Mean": histogram_mean,
                "Histogram Std Dev": histogram_std,
                "EXIF Data": exif_data
            }
    
    for file_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, file_name)

        if not os.path.isfile(file_path):
            continue
        
        try:
            base_name = os.path.splitext(file_name)[0]
        
            original_data = analyze_image(file_path)
            dct_image_path = os.path.join(output_folder_dct, f"{base_name}.png")
            alpha_image_path = os.path.join(output_folder_alpha, f"{base_name}.png")
            lsb_image_path = os.path.join(output_folder_lsb, f"{base_name}.png")
            
            dct_data = analyze_image(dct_image_path)
            alpha_data = analyze_image(alpha_image_path)
            lsb_data = analyze_image(lsb_image_path)

            data.append({
                "File Name": file_name,
                "Original File Size (Bytes)": original_data["File Size (Bytes)"],
                "DCT File Size (Bytes)": dct_data["File Size (Bytes)"],
                "Alpha File Size (Bytes)": alpha_data["File Size (Bytes)"],
                "LSB File Size (Bytes)": lsb_data["File Size (Bytes)"],
                "Original Histogram Mean": original_data["Histogram Mean"],
                "DCT Histogram Mean": dct_data["Histogram Mean"],
                "Alpha Histogram Mean": alpha_data["Histogram Mean"],
                "LSB Histogram Mean": lsb_data["Histogram Mean"],
                "Original Histogram Std Dev": original_data["Histogram Std Dev"],
                "DCT Histogram Std Dev": dct_data["Histogram Std Dev"],
                "Alpha Histogram Std Dev": alpha_data["Histogram Std Dev"],
                "LSB Histogram Std Dev": lsb_data["Histogram Std Dev"],
            })

            latex_content.append(f"""
            \section*{{Porównanie obrazu: {file_name}}}
            \begin{{tabular}}{{|c|c|c|c|c|}}
                \hline
                \textbf{{Parametr}} & \textbf{{Przed}} & \textbf{{DCT}} & \textbf{{Alpha}} & \textbf{{LSB}} \\
                \hline
                Rozmiar pliku (B) & {original_data['File Size (Bytes)']} & {dct_data['File Size (Bytes)']} & {alpha_data['File Size (Bytes)']} & {lsb_data['File Size (Bytes)']} \\
                Średnia histogramu & {original_data['Histogram Mean']:.2f} & {dct_data['Histogram Mean']:.2f} & {alpha_data['Histogram Mean']:.2f} & {lsb_data['Histogram Mean']:.2f} \\
                Odchylenie std. histogramu & {original_data['Histogram Std Dev']:.2f} & {dct_data['Histogram Std Dev']:.2f} & {alpha_data['Histogram Std Dev']:.2f} & {lsb_data['Histogram Std Dev']:.2f} \\
                \hline
            \end{{tabular}}
            """)
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")

    df = pd.DataFrame(data)
    df.to_csv(output_comparison_csv_path, index=False)

    with open(output_comparison_latex_path, 'w', encoding='utf-8') as latex_file:
        latex_file.write(r"\documentclass{article}\n\usepackage[utf8]{inputenc}\n\begin{document}\n")
        latex_file.write(r"\section*{Raport porównania obrazów przed i po ukryciu wiadomości}\n")
        latex_file.write(r"\n".join(latex_content))
        latex_file.write(r"\n\end{document}")


input_folder = r"C:\Users\hgolebio\OneDrive - Eltel Group Corporation\Documents\VSCode\Mouse\KRYS"
output_folder_dct = r"C:\Users\hgolebio\OneDrive - Eltel Group Corporation\Documents\VSCode\Mouse\KRYS\KRYS DCT"
output_folder_alpha = r"C:\Users\hgolebio\OneDrive - Eltel Group Corporation\Documents\VSCode\Mouse\KRYS\KRYS RGBA"
output_folder_lsb = r"C:\Users\hgolebio\OneDrive - Eltel Group Corporation\Documents\VSCode\Mouse\KRYS\KRYS LSB"
message = "To jest tajna wiadomość."

# Analiza obrazów

output_csv_path = "image_analysis_report.csv"
latex_output_path = "image_analysis_report.tex"
analyze_images_in_folder(input_folder, output_csv_path, latex_output_path)


# Ukrywanie obrazow 
hide_messages_in_folder(input_folder, output_folder_dct, output_folder_alpha, output_folder_lsb, message)

# Analiza ukrytych obrazow i porownanie - tabela latex 

output_comparison_csv_path = "image_comperation_analysis_report.csv"
output_comparison_latex_path = "image_comperation_analysis_report.tex"
analyze_and_compare_images(input_folder, output_folder_dct, output_folder_alpha, output_folder_lsb, output_comparison_csv_path, output_comparison_latex_path)