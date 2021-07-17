import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import sys

file_dir=sys.argv[1]
img_shape=(150,150)
data_shape=(150,150,3)

import matplotlib.image as mpimg
img=mpimg.imread(file_dir)  

x=tf.cast(tf.image.resize(img,img_shape),tf.dtypes.int64)

labels=['AFRICAN FIREFINCH', 'ALBATROSS', 'ALEXANDRINE PARAKEET', 'AMERICAN AVOCET', 'AMERICAN BITTERN', 'AMERICAN COOT', 'AMERICAN GOLDFINCH', 'AMERICAN KESTREL', 'AMERICAN PIPIT', 'AMERICAN REDSTART']

model=tf.keras.models.load_model('model/model1')

pred=model.predict(tf.reshape(x,(1,150,150,3)))

ans=[]
for i,p in enumerate(pred[0]):
    ans.append((p,labels[i]))

print('bird species',' '*18,'confidence')
for p,l in sorted(ans,reverse=True):
    print(l,' '*(30-len(l)),round(p,2))