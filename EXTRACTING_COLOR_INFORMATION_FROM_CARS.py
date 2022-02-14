#!/usr/bin/env python
# coding: utf-8

# In[1]:


**READ ME: STEPS INVOLVED IN COLOR IDENTIFICATION OF VARIOUS CARS IN THE IMAGE **
    
    1. IMPORT THE REQUIRED FILES
    2.USE CV2 AND MATLAB TO PROCESS AND RE SIZE IMAGE
    3. USE GREYING TECHNIQUE ON IMAGE
    4.APPLY IMAGE PROCESSING USING CV2 AND GET_ colors technique using python webcolors , hex , rgb and other libraries
    5.Display the colors of various cars [ graphical pi chart and represent in the form of list output]
    6.for certain un identified colors if exists: scipy, kdtree and webcolors_css3 for nearest color conversion


# In[ ]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import cv2
from collections import Counter
from skimage.color import rgb2lab, deltaE_cie76
import os

get_ipython().run_line_magic('matplotlib', 'inlin')


# In[2]:


get_ipython().system('pip install webcolors')
import webcolors
from webcolors import hex_to_name
from webcolors import rgb_to_name


# In[3]:


import cv2
import sys
import glob
import webcolors
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
from collections import Counter


# In[4]:


image = cv2.imread('imagemicron.jpg')
print("The type of this input is {}".format(type(image)))
print("Shape: {}".format(image.shape))
plt.imshow(image)


# In[5]:


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)


# In[6]:


gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
plt.imshow(gray_image, cmap='gray')


# In[7]:


resized_image = cv2.resize(image, (1200, 600))
plt.imshow(resized_image)


# In[8]:


def RGB2HEX(color):
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))


# In[9]:


def get_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# In[ ]:


for img_clr, img_hex in webcolors.CSS3_NAMES_TO_HEX.items():
    cur_clr = webcolors.hex_to_rgb(img_hex)


# In[13]:


def get_colors(image, number_of_colors, show_chart):
    
    modified_image = cv2.resize(image, (600, 400), interpolation = cv2.INTER_AREA)
    modified_image = modified_image.reshape(modified_image.shape[0]*modified_image.shape[1], 3)
    
    clf = KMeans(n_clusters = number_of_colors)
    labels = clf.fit_predict(modified_image)
    
    counts = Counter(labels)
    # sort to ensure correct color percentage
    counts = dict(sorted(counts.items()))
    
    center_colors = clf.cluster_centers_
    # We get ordered colors by iterating through the keys
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
    #print(rgb_colors)
    print(hex_colors)


    if (show_chart):
        plt.figure(figsize = (8, 6))
        plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    
    #return rgb_colors
    car_colors = []
    
    try:
        for i in hex_colors:
           #print(hex_to_name(i,spec='css3'))
           car_colors.append(hex_to_name(i))

    except ValueError as v_error:
        print("{}".format(v_error))
        
    print(car_colors)   
get_colors(get_image('imagemicron.jpg'), 10, True)


# In[ ]:





# In[ ]:


def color_rgb_to_name(rgb: tuple[int, int, int]) -> str:
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - rgb[2]) ** 2
        gd = (g_c - rgb[1]) ** 2
        bd = (b_c - rgb[0]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

image = cv2.imread('imagemicron.jpg')
colors = set([color_rgb_to_name(val) for val in np.unique(image.reshape(-1, 3), axis=0)])


# In[ ]:


from scipy.spatial import KDTree
get_ipython().system('pip install color')
from webcolors import (
    css3_hex_to_names,
    hex_to_rgb,
)
def convert_rgb_to_names(rgb_tuple):
    
    # a dictionary of all the hex and their respective names in css3
    css3_db = css3_hex_to_names
    names = []
    rgb_values = []
    for color_hex, color_name in css3_db.items():
        names.append(color_name)
        rgb_values.append(hex_to_rgb(color_hex))
    
    kdt_db = KDTree(rgb_values)
    distance, index = kdt_db.query(rgb_tuple)
    return f'closest match: {names[index]}'


# In[ ]:


im = Image.open("imagemicron.jpg")
n, color = max(im.getcolors(im.size[0]*im.size[1]))
print(color)
(119, 172, 152)


# In[ ]:


import webcolors

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

requested_colour = (119, 172, 152)
actual_name, closest_name = get_colour_name(requested_colour)

print("Actual colour name:", actual_name, ", closest colour name:", closest_name)


# In[ ]:


get_ipython().system('pip install webcolors')
import webcolors
from webcolors import hex_to_name
car_colors = []
for i in hex_colors:
  car_colors.append(hex_to_name(i))
print(car_colors)


# In[ ]:


IMAGE_DIRECTORY = 'imagemicron.jpg'
COLORS = {
    'GREEN': [0, 128, 0],
    'BLUE': [0, 0, 128],
    'YELLOW': [255, 255, 0]
}
images = []

for file in os.listdir(IMAGE_DIRECTORY):
    if not file.startswith('.'):
        images.append(get_image(os.path.join(IMAGE_DIRECTORY, file)))
        


# In[ ]:





# In[ ]:




