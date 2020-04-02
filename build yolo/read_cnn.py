from keras.models import load_model
import h5py
import numpy as np
import cv2

model = load_model('cnn.h5')
img= cv2.imread('images.jpg')
test = cv2.resize(img,(200,200))
test = np.array([test])
out = model.predict(test)[0][0]
print(out)
if out >=0.8:
    out = 'Dog'
elif out <=0.2:
    out = 'Cat'
else:
    out = 'Not a dog or cat'
print(out)
cv2.putText(img,out,(0,50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
cv2.imshow('test',img)
cv2.waitKey(0)
cv2.destroyAllWindows()





