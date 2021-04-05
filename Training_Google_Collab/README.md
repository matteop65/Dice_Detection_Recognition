I trained using Google Colab, it has dedicated GPU's for this sort of stuff. 

If you want to train just copy over the Custom_YOLOv4_Trainer.ipynb to your Colab.
You will want to make sure it connects to your drive, and your drive has the following folder:
	1. yolov4 folder in \My Drive\
	2. Backup folder in \yolov4\
In the yolov4 folder at /My\ Drive/yolov4, have the following files:
	2. obj.data
	3. obj.names
	4. yolov4-custom.cfg
	5. Training_Images.zip, contains Train and Valid folders that have the labeled images (in txt)\

All files and directories are case sensitive, but if you want to change it, you may. But make sure you do it on YOUR copy of the colab and Drive and not on the Github. 

You will need to edit, obj.data, obj.names and yolov4-custom.cfg accordingly and upload Training_Images.zip to your drive. 
 
Whilst training, you can choose to use yolov4.conv.137 or darknet53.conv.74, I prefer darknet53.conv.74 as it is faster, the number represents number of convolutional layers. 

Whilst training, Colab will automatically save the .weights file in the /My\ Drive/yolov4/backup/ every 100 iterations, or if after 1000 iterations, it will calculate the mAP every 100 iterations, and then also save the best. 

At the bottom there is some code you can put into the console, in inspect element, to help prevent disconnection.