from PIL import Image 
from PIL import ImageFilter
 
im=Image.open('cat_dog.png')
im=im.filter(ImageFilter.GaussianBlur(radius=6))
 
im.show()
im.save('new_cat_dog.png')