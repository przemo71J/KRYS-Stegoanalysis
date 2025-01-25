from stegano import lsb

secret_message = "Celem DAP jest optymalii."


# Load the original and stego images
img = 'man.png'
path = 'C:/Users/prpustel/Desktop/KRYS/proj/images/'
original_image = path+img
output_image = path + img.split('.')[0] + '_hidden.' + img.split('.')[1]

lsb.hide(original_image, message=secret_message).save(output_image)

revealed_message = lsb.reveal(output_image)
print(f"Revealed message: {revealed_message}")