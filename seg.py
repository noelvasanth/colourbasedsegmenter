import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob

def combine(img1, img2):
	foreground = img1
	background = img2
	alpha = 0.5
	 
	# Convert uint8 to float
	foreground = foreground.astype(float)
	background = background.astype(float)
	
	# Multiply the foreground with the alpha matte
	foreground = foreground
	 
	# Multiply the background with ( 1 - alpha )
	background = background
	
	# Add the masked foreground and background.
	outImage = cv2.add(foreground, background)
	
	return outImage

# for file in im_files:
# 	im = cv2.imread(file)
# 	im = cv2.resize(im, (640, 640))
# 	cv2.imwrite(file.split('/')[-1], im)


im = cv2.imread('/Users/noel/Desktop/sortedimages/Finished pairs/4/IMG_20190329_112835.jpg')
#im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
#im = im.astype(np.float32)/255.0
im = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)


#color_labels = ['RED', 'GREEN', 'YELLOW']
color_labels_h = ['PINK', 'PURPLE', 'BLUE']
#color_labels_t = ['CELLS', 'FOREGROUND', 'NUCLEI']
# for color in color_labels:
# 	if color == 'RED':
# 		mask_r = (im[:,  :, 0] > 0.5).astype(np.uint8)
# 		mask_g = (im[:,  :, 1] < 0.1).astype(np.uint8)
# 		mask_b = (im[:,  :, 2] < 0.1).astype(np.uint8)
# 	elif color == 'GREEN':
# 		mask_r = (im[:,  :, 0] < 0.1).astype(np.uint8)
# 		mask_g = (im[:,  :, 1] > 0.5).astype(np.uint8)
# 		mask_b = (im[:,  :, 2] < 0.1).astype(np.uint8)
# 	else:
# 		mask_r = (im[:,  :, 0] > 0.5).astype(np.uint8)
# 		mask_g = (im[:,  :, 1] > 0.5).astype(np.uint8)
# 		mask_b = (im[:,  :, 2] < 0.1).astype(np.uint8)
# 	print(mask_r.shape, mask_g.shape, mask_b.shape)
# 	mask = np.multiply(np.multiply(mask_r, mask_g), mask_b).astype(np.float32)
# 	seg_im = np.reshape(mask , (im.shape[0], im.shape[1], 1))
# 	seg_im = cv2.bitwise_and(im, im, mask = seg_im)
# 	cv2.imwrite(color+'.png', seg_im*im)

# define the list of boundaries
boundaries = [
	([170, 120, 70], [180, 255, 255]), #RED
	([65, 60, 60], [100, 255, 255]), #GREEN
	([20, 100, 100], [30, 255, 255]), #YELLOW
]
boundaries_h = [
	([125, 100, 30], [255, 255, 255]), #PINK
	([65, 50, 50],[130, 255, 255]), #PURPLE
	([110, 50, 50], [130, 255, 255]) #BLUE
]
boundaries_t = [
	([150, 10, 200], [180, 255, 255]), #CELL
	([140, 30, 200],[150, 90, 255]), #FOREGROUND
	([120, 40, 100], [140, 255, 255]) #NUCLEUS
]

mask_labels = [
	[255, 0, 0],
	[0, 255, 0], 
	[0, 0, 255]
]

i = 0
mask_i = []
# loop over the boundaries
for (lower, upper) in boundaries_t:
	# create NumPy arrays from the boundaries
	lower = np.array(lower, dtype = "uint8")
	upper = np.array(upper, dtype = "uint8")

	# find the colors within the specified boundaries and apply
	# the mask
	mask = cv2.inRange(im, lower, upper)
	if i==0: 
		mask_i = cv2.bitwise_and(im, im, mask = mask)
		mask_i[mask_i>0]=255
		mask_i[np.where((mask_i == [255,255,255]).all(axis = 2))] = mask_labels[i]
	else:
		mask_ = cv2.bitwise_and(im, im, mask = mask)
		mask_[mask_>0]=255
		mask_[np.where((mask_ == [255,255,255]).all(axis = 2))] = mask_labels[i]
		mask_i = combine(mask_i, mask_)

	output = cv2.bitwise_and(im, im, mask = mask)
	output_rgb = cv2.cvtColor(output, cv2.COLOR_HSV2RGB)
	#cv2.imwrite(color_labels_h[i]+'.jpg', output_rgb)
	i += 1
	# show the images
	# plt.imshow(np.hstack([cv2.cvtColor(im, cv2.COLOR_HSV2RGB), cv2.cvtColor(output, cv2.COLOR_HSV2RGB)]))
	# plt.show()
	# plt.imshow(np.hstack([im, output]))
	# plt.show()

#		cv2.imshow("IMG",mask_i)
#		cv2.waitKey(0)
	# cv2.imwrite('label_mask.jpg', mask_i)
	print(np.unique(mask_i))
	j = Image.fromarray(mask_i.astype(np.uint8), 'RGB')
	j.save('label_mask.png')

label = Image.open('label_mask.png').convert('P', palette = 'WEB')
p = label.getpalette()
print(np.unique(p))
print(np.sort(label.getcolors()))
label = np.asarray(label, dtype= np.uint8)
print(np.unique(label))
img = Image.fromarray(label, mode='P')
img.putpalette(p)
img.save('label_mask.png')

