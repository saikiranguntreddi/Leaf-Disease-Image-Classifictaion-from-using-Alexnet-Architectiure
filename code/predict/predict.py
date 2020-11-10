# predicting an image
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()
image_path = "new-plant-diseases-dataset/test/TomatoEarlyBlight/TomatoEarlyBlight2.JPG"
new_img = image.load_img(image_path, target_size=(224, 224))
img = image.img_to_array(new_img)
img = np.expand_dims(img, axis=0)
img = img/255

print("Prediction:")
prediction = classifier.predict(img)
d = prediction.flatten()
j = d.max()
for index,item in enumerate(d):
    if item == j:
        class_name = li[index]


#img_class = classifier.predict_classes(img)
#img_prob = classifier.predict_proba(img)
#print(img_class ,img_prob )


#ploting image with class name        
plt.figure(figsize = (4,4))
plt.imshow(new_img)
plt.axis('off')
plt.title(class_name)
plt.show()
