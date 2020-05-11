#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# ### Cascade Files
# 
# OpenCV comes with pre-trained cascade files

# ### Face Detection

# In[2]:


face_detection = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# In[4]:


def detect_face(img):
    face_img = img.copy()
    face_rectangle = face_cascade.detectMultiScale(face_img)
    for (x,y,w,h) in face_rectangle:
        cv2.rectangle(face_img,
                     (x,y),
                     (x+w, y+h),
                     (255,255,255),
                     10)
    return face_img


# ### Face Detection with Live Video

# In[ ]:


cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(3) & 0xFF == 27:
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:





# In[ ]:




