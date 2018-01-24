import numpy as np
import cv2
import os

# this function is used to convert the images to 128*128 tensors
def image_to_tensor():
	image_path=raw_input("Enter Image Directory : ")
	if(image_path==""):
	    image_path="dataset"
	image_path = os.getcwd() + "/" + image_path
	file_reader=os.listdir(image_path)

	train=[]
	label=[]
	class_number=-1
	class_label_map={}
	for folders in file_reader:
		if(os.path.isdir(os.path.join(image_path,folders))):
			image_reader=os.listdir(os.path.join(image_path,folders))
			path=os.path.join(image_path,folders)
			class_number+=1
			class_label_map[class_number]=folders
			for image in image_reader:
				if(image.lower().endswith(('.png', '.jpg', '.jpeg'))):
					# parameter 0 for grayscale, 1 for RGB
					train.append(cv2.imread(os.path.join(path,image),0)) 
			        label.append(class_number)


    # training data and labels converted from list to numpy arrays and saved on the disk
	train=np.array(train)
	train=train.reshape(-1,128,128,1) 
	label=np.array(label)
	np.save("train.npy",train)
	np.save("label.npy",label)

	# mapping of class name with corresponding number. for ex -> A:0, B:1
	return class_label_map

# calling the function
image_to_tensor()