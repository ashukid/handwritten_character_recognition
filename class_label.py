import numpy as np
import cv2
import os

# this function is used to get the mapping between labels and class name
# for example -> A:0, B:1, C:3
def get_class_label():
	image_path=raw_input("Enter Image Directory : ")
	if(image_path==""):
	    image_path="dataset"
	image_path = os.getcwd() + "/" + image_path
	file_reader=os.listdir(image_path)


	class_number = 0
	class_label_map={}
	for folders in file_reader:
		if(os.path.isdir(os.path.join(image_path,folders))):
			image_reader=os.listdir(os.path.join(image_path,folders))
			path=os.path.join(image_path,folders)
			class_label_map[class_number]=folders
			class_number+=1

	return class_label_map

# calling the function
class_label_map=get_class_label()
# print(class_label_map)