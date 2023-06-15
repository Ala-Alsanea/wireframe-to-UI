import streamlit as st
import os
import shutil
import cv2
import numpy as np
import pandas as pd
from selenium import webdriver
from pathlib import Path


# page config
st.set_page_config(
    page_title="Wireframe-to-UI",
    page_icon="✏",
    layout="wide",
    initial_sidebar_state="expanded",

)


st.image('./assets/logo11.png', width=280)
st.header('Wireframe-to-UI')


imgPath = st.file_uploader('pick an image', type=['png', 'jpg'],)
if imgPath == None or imgPath == []:
    st.caption('please choose an image to convert it to html code')
    st.markdown("""
           Done by :
 - [Ala Al-Sanea](https://github.com/Ala-Alsanea) 
 - [Mai Al-Romaishi](https://www.instagram.com/malromaishi/) 
 - [Mohammed Al-Qiari]()
 - [Walid Talal]()
 
Supervisor:
 - Dr.Amin Shayea

            
                """)
    exit()

# st.write(dir(imgPath))
# st.write(type(imgPath.getvalue()))
# st.write(imgPath.type)
st.markdown('-----')
filename = imgPath.name[:-4]
file_bytes = np.asarray(bytearray(imgPath.read()), dtype=np.uint8)
opencv_image = cv2.imdecode(file_bytes, 1)

cv2.imwrite('predicted/2.png', opencv_image)

topCol = st.columns([1, 1, 1], gap="small")

with topCol[0]:
    st.header('Original Image')
    st.image(opencv_image)

st.markdown('-----')

if not st.button('convert', type="primary"):
    exit()

os.system(
    f'cd yolov5 &&  python3 detect.py --img ../predicted/2.png ')
os.system(
    f'cd compiler/compiler_pix2code/ && python3 web-compiler.py ../../predicted/index.txt ../../predicted/')

# pred_img = cv2.imread('predicted/1.jpg', 1)
# st.header(filename,)


with topCol[1]:
    st.header('Predicted Elements')
    st.image('./predicted/1.jpg')

with topCol[2]:
    st.header('Predicted Table',)
    pred_df = pd.read_csv('./predicted/DSL.csv', )
    st.table(pred_df)

pagePath = os.path.abspath('predicted')


webWindow = webdriver.Chrome()
webWindow.set_window_size(1000, 800)
webWindow.get((f'file:///{pagePath}/index.html').replace("\\", "/"))
# webWindow.save_screenshot(f'{pagePath}/index.png')
