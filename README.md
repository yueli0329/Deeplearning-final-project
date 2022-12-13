# Deeplearning-final-project

In the project,  we used two pre-trained convolutional neural networks to detect IDC in the tissue slice images. In order to avoid the overfitting issue and improve the model performance, we implemented two methods. One is data augmentation method and the other is learning rate search method. In the data augmentation,  methods in the transformer were used to increase the diversity of the images and a general adversarial network was implemented to standardize the image.  In the training process, the cyclical learning rate (CLR) search method was used when training the Resnet18 and the VGG16 framework.  The best performance is the accuracy 0.85 on the test set on Resnet18 with CLR search. 

