from PIL import Image
import PIL

# creating a image object (main image)
im1 = Image.open(r"D:\\SIH fabric-test\\dataset\\results\\0001_002_00.png")

# save a image using extension
im1 = im1.save("C:\\Users\\admin\\Desktop\\t\\geeks.jpg")
