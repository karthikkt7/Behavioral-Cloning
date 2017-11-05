#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* model.ipynb The juypter notebook containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 18-24) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

This project is entirely dependent on two main factors
- How well the network architecture is defined
- How good the model has been trained (with a considerable number of images including data augmentation) 

I have faced many challenges in carrying out this project. Since the errrors or issues that I was facing is quite entirely new to me, I am not able to complete the project.
Though I am submitting my trained model and the python file(including jupyter notebook). Due to time constraint, I had to submit the following files alone. I did not receive proper
support from mentor and tried to post my errors in slack community, but got less response. I have to thank personally to fellow nano-degree student Mr.Joao Sauso Pinto for helping me 
to solve the errors and atleast run my model. He also provided a huge data set of images which I find it pretty good.

I do not know the reason why the simulator is not running in Autonomous mode to test my trained model. Spent a lot of time on finding the reason, but unfortunately did not succeed.
The simulator hangs up when I run the command "python drive.py model.h5"

The data visualization is not discussed here since I find this is bit in the background (you can find it in .ipynb)


The network architecture is defined as follows :
- A lambda layer for normalization between -0.5...0.5
- A convolution layer(5X5) with 24 output channels
- RELU activation
- 2X2 Maxpooling
- Dropout (0.5)
- Convolution with 36 output channels
- RELU activation
- 2X2 Maxpooling
- Dropout (0.5)
- Convolution with 48 output channels
- RELU activation
- 2X2 Maxpooling
- Dropout
- Convolution with 64 output channels
- RELU activation
- 2X2 Maxpooling
- Dropout
- Convolution with 64 output channels
- RELU activation
- 2X2 Maxpooling
- Dropout
- Flatten layer
- Dense (200)
- RELU activation
- Dropout(0.5)
- Dense (50)
- Dense (10)
- Dense (1)

The total number of images that I feed in is 201700 and i use offset of 0.4 for the right and left camera images
Though I had split the entire set into 90% training and 10%validation, I did not use the validation set due to time constraint.
But i find that the MSE decreases from one EPOCH to the next which is good and since I have 3 EPOCHS, I do not see overfitting.
I also flip alternate images which I do it in generator_data. The ADAM optimizer with learning rate is kept at 0.00025(trial and errror)
The batch size i used is 50.

Problems :
- AWS instance access took a lot of time to get resolved
- Meantime when I run the model in my PC (which do not have high graphics processor), I had to reduce the data set to 12K.
  For such image data set, it took a day to run. So, I managed to conclude (atleast within deadline) with the best out of me in finally   running it on AWS instance after resolving the issues associated with AWS(near to the end of project deadline).
  
Due to the less support I got from Mentor, forums, I was able to finish to this level. In the end, the project aimed at 
defining the network architecture and how much good and number of image data set that it is trained on.
The files I am submitting are :
- drive.py (changed for cropping the image which is fed to the model)
- model.h5 (trained model)
- I am not including the video since I do not know the reason why it is not running in my PC.

