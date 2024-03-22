# Computer Vision
# use haar features to detect faces
# use classifier to assign a label to the face using Nearest Neighbor
    # there may be false positives / false negatives
# use linear decision boundary

# Computer Vision using SIFT classical method
import cv2
import matplotlib.pyplot as plt

# load images
imgs = []
imgs.append(cv2.imread('IMG_1.jpg'))
imgs.append(cv2.imread('IMG_2.jpg'))
imgs.append(cv2.imread('IMG_3.jpg'))
imgs.append(cv2.imread('IMG_4.jpg'))

# load the face cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# convery to gray scale
for i in range(len(imgs)):
    grey_images = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey_images, 1.1, 4)
    print(f'Number of faces found in IMG_{i+1}: {len(faces)}')
    for (x, y, w, h) in faces:
        cv2.rectangle(imgs[i], (x, y), (x+w, y+h), (0, 0, 255), 2)
        
# convert images back to RGB
for i in range(len(imgs)):
    imgs[i] = cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB)
    
# display the image
fig, ax = plt.subplots(2, 2, figsize=(20, 20))
ax[0, 0].imshow(imgs[0])
ax[0, 0].set_title('IMG_1')

ax[0, 1].imshow(imgs[1])
ax[0, 1].set_title('IMG_2')

ax[1, 0].imshow(imgs[2])
ax[1, 0].set_title('IMG_3')

ax[1, 1].imshow(imgs[3])
ax[1, 1].set_title('IMG_4')

ax[0, 0].axis('off')
ax[0, 1].axis('off')
ax[1, 0].axis('off')
ax[1, 1].axis('off')
plt.savefig('output/face_detection.png')
plt.show()