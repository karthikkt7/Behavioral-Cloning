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

###Model Architecture and Training Strategy

####1. Solution Design Approach

This project is entirely dependent on two main factors

How well the network architecture is defined
How good the model has been trained (with a considerable number of images including data augmentation)
I have faced many challenges in carrying out this project. Since the errrors or issues that I was facing is quite entirely new to me, I am not able to complete the project. Though I am submitting my trained model and the python file(including jupyter notebook). Due to time constraint, I had to submit the following files alone. I did not receive proper support from mentor and tried to post my errors in slack community, but got less response. I have to thank personally to fellow nano-degree student Mr.Joao Sauso Pinto for helping me to solve the errors and atleast run my model. He also provided a huge data set of images which I find it pretty good. The recorded data has the car driving in the center, several random recordings, car also drives close to the left lane, center of the lane, close to the right lane, car driving in reverse direction. It has in total 2.4L images from the lap recording.

I do not know the reason why the simulator is not running in Autonomous mode to test my trained model. Spent a lot of time on finding the reason, but unfortunately did not succeed. The simulator hangs up when I run the command "python drive.py model.h5"

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
The total number of images that I feed in is approx 2.1 lac images and i use offset of 0.15 for the right and left camera images Though I had split the entire set(~2.4L images) into 90% training and 10%validation, I did not use the validation set due to time constraint. But i find that the MSE decreases from one EPOCH to the next which is good and since I have 3 EPOCHS, I do not see overfitting. I also flip alternate images which I do it in generator_data. The ADAM optimizer with learning rate is kept at 0.00025(trial and errror) The batch size i used is 50.

Problems :

- AWS instance access took a lot of time to get resolved
- Meantime when I run the model in my PC (which do not have high graphics processor), I had to reduce the data set to 12K. For such image data set, it took a day to run. So, I managed to conclude (atleast within deadline) with the best out of me in finally running it on AWS instance after resolving the issues associated with AWS(near to the end of project deadline).
  Due to the less support I got from Mentor, forums, I was able to finish to this level. In the end, the project aimed at defining the network architecture and how much good and number of image data set that it is trained on. The files I am submitting are :

drive.py (changed for cropping the image which is fed to the model)
- model.h5 (trained model renamed from model15.h5)
- model15.h5 (original model trained)
I am not including the video since I do not know the reason why it is not running in my PC.
Still I am not able to find the reason why the simulaotr is not running in my PC.

Problem : 
I train the model in AWS. Download it to my PC. 
Run the command "python drive.py model.h5". Everything works fine. waiting for simulator to run.
I run the simulator. I select the display and resolution. Then select Autonomous mode. Simulator hangs.
It has been difficult to complete the project without running the simulator.

In the end, i gave the model for testing to Mr.Joao. Based on his inputs (runs to some extent and then swirls to left or right, etc)
I atleast changed my code. But due to time constraint, without the simulator it is difficult to change the code and do some testing.
