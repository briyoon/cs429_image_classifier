import torch
import torchvision

img_path = "data/whales/train/0a0b2a01.jpg"

image = torchvision.io.read_image(img_path, torchvision.io.ImageReadMode.RGB)
print(image.min(), image.max())

float_img = image.float()
norm_img = torchvision.transforms.Normalize((127.5, 127.5, 127.5), (127.5, 127.5, 127.5))(float_img)
print(norm_img.min(), norm_img.max())
