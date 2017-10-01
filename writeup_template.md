# **Behavioral Cloning** 

---

**Build a Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center_lane_driving.jpg "Center Lane Driving"
[image2]: ./examples/reverse_driving.jpg "Reverse Driving"
[image3]: ./examples/original_image.jpg "Normal Image"
[image4]: ./examples/flipped_image.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* CarND-Behavioral-Cloning-P3 containing the jupyter notebook to create and train the model
* drive.py for driving the car in autonomous mode
* model_8.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

You're reading it! and here is a link to my [project notebook](https://github.com/patriciapampanelli/CarND-Behavioral-Cloning-P3/blob/master/CarND-Behavioral-Cloning-P3.ipynb)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_8.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of three convolution layers with 5x5 filter sizes and depths 24, 36 and 48. I also used two additional convolution layers with 3x3 filter sizes and depths of 64. The following layers are three fully connected layers with 100, 50, 10 neurons, respectively, and the output layer with 1 neuron. I also added 5 dropout layers.

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer. 

The final model is inspired in the Nvidia Autonomous team architecture: 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Lambda         		| x = x/255 - 0.5 Normalizing and mean centering - Input shape = (160, 320, 3)|
| Cropping2D            | (70, 25), (0,0)   							| 
| Convolution 5x5     	| 2x2 stride, valid padding, 24 depth			|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, 36 depth    		|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, 48 depth    		|
| RELU					|												|
| Dropout    		    | Probability: 0.5					    		|
| Convolution 3x3	    | 1x1 stride, valid padding, 64 depth    		|
| RELU					|												|
| Dropout    		    | Probability: 0.5					    		|
| Convolution 3x3	    | 1x1 stride, valid padding, 64 depth    		|
| RELU					|												|
| Flatten				|												|
| Fully connected		| 100 neurons   								|
| Dropout    		    | Probability: 0.5					    		|
| Fully connected		| 50 neurons   									|
| Dropout    		    | Probability: 0.5					    		|
| Fully connected		| 10 neurons   									|
| Dropout    		    | Probability: 0.5					    		|
| Fully connected		| 1 neurons   			                 		|

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving and a reverse driving. More data help the model to generalize well. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to experiment different architectures varying the number of fully connected and convolutional layers and adding, or not, dropout layers.

My first step was to use a convolution neural network model similar to the Nvidia Autonomous team architecture, as I said before. Then I keep trying to include different 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model adding dropout layers. I also trained the model using more data. I added a reverse driving to the original dataset.  

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I included recovering driving in different circuit sections.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Lambda         		| x = x/255 - 0.5 Normalizing and mean centering - Input shape = (160, 320, 3)|
| Cropping2D            | (70, 25), (0,0)   							| 
| Convolution 5x5     	| 2x2 stride, valid padding, 24 depth			|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, 36 depth    		|
| RELU					|												|
| Convolution 5x5	    | 2x2 stride, valid padding, 48 depth    		|
| RELU					|												|
| Dropout    		    | Probability: 0.5					    		|
| Convolution 3x3	    | 1x1 stride, valid padding, 64 depth    		|
| RELU					|												|
| Dropout    		    | Probability: 0.5					    		|
| Convolution 3x3	    | 1x1 stride, valid padding, 64 depth    		|
| RELU					|												|
| Flatten				|												|
| Fully connected		| 100 neurons   								|
| Dropout    		    | Probability: 0.5					    		|
| Fully connected		| 50 neurons   									|
| Dropout    		    | Probability: 0.5					    		|
| Fully connected		| 10 neurons   									|
| Dropout    		    | Probability: 0.5					    		|
| Fully connected		| 1 neurons   			                 		|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][image1]

To help the model generalize well I drived in the reverse direction:

![Reverse Driving][image2]

To augment the data set, I also flipped images and angles thinking that this would also prevent overfitting. For example, here is an image that has then been flipped:

![Normal Image][image3]
![Flipped Image][image4]

After the collection process, I had 36291 number of data points. I then preprocessed this data by adding to the dataset flipped images and angles. I finally randomly shuffled the data set and put 20% of the data into a validation set. Then I trained the model using 58065 and validated on 14517. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5. I used an adam optimizer so that manually training the learning rate wasn't necessary.
