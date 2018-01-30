import os
from shutil import copyfile

image_path="by_class"
image_path = os.getcwd() + "/" + image_path
file_reader=os.listdir(image_path)
final_path=os.getcwd()

i=0
for folders in file_reader:
	if(os.path.isdir(os.path.join(image_path,folders))):
		file_reader1=os.listdir(os.path.join(image_path,folders))
		image_path1 = os.path.join(image_path,folders)
		for folders1 in file_reader1:
			if("train" in folders1):
				image_reader=os.listdir(os.path.join(image_path1,folders1))
				path=os.path.join(image_path1,folders1)
				dest_path=final_path+"/train_dataset/"+folders1+str(i)
				if not os.path.exists(dest_path):
					os.makedirs(dest_path)

				count=0
				for image in image_reader:
					if(count < 250):
						if(image.lower().endswith(('.png', '.jpg', '.jpeg'))):
							copyfile(os.path.join(path,image), os.path.join(dest_path,image))
						count+=1
			i+=1;