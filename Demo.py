#!/usr/bin/env python
# coding: utf-8

# In[1]:

from pylab import *
import cv2


# In[2]:


rcParams['figure.figsize'] = 10, 10


# In[3]:


from dataset import load_image


# In[13]:


import torch


# In[ ]:


from utils import cuda


# In[ ]:


from generate_masks import get_model


# In[ ]:


from albumentations import Compose, Normalize


# In[ ]:


from albumentations.torch.functional import img_to_tensor


# In[ ]:


def img_transform(p=1):
    return Compose([
        Normalize(p=1)
    ], p=p)


# In[ ]:


def mask_overlay(image, mask, color=(0, 255, 0)):
    """
    Helper function to visualize mask on the top of the car
    """
    mask = np.dstack((mask, mask, mask)) * np.array(color)
    mask = mask.astype(np.uint8)
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.)
    img = image.copy()
    ind = mask[:, :, 1] > 0    
    img[ind] = weighted_sum[ind]    
    return img


# In[ ]:


model_path = 'data/models/unet11_binary_20/model_0.pt'
model = get_model(model_path, model_type='UNet11', problem_type='binary')


# In[ ]:


img_file_name = 'data/cropped_train/instrument_dataset_3/images/frame004.jpg'
gt_file_name = 'data/cropped_train/instrument_dataset_3/binary_masks/frame004.png'


# In[ ]:


image = load_image(img_file_name)
gt = cv2.imread(gt_file_name, 0) > 0


# In[ ]:


imshow(image)


# In[ ]:


with torch.no_grad():
    input_image = torch.unsqueeze(img_to_tensor(img_transform(p=1)(image=image)['image']).cuda(), dim=0)


# In[ ]:


mask = model(input_image)


# In[ ]:


mask_array = mask.data[0].cpu().numpy()[0]


# In[ ]:


imshow(mask_array > 0)


# In[ ]:


imshow(mask_overlay(image, (mask_array > 0).astype(np.uint8)))

