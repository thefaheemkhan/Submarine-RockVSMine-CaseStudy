# Submarine-RockVSMine-Project

## This Case Study Objective : 
This use case consider that is a submarine and there is a war going on between two countries so submarine of a country is going underwater to another country and the enemy country have planted some mines in the ocean. So mines are nothing but the explosives that explodes when some object comes in contact with it right so there can also be rocks in the ocean.
### The submarine needs to predict whether it is crossing a mine or a rock. so our job is to make a system that can predict whether the object beneath the submarine is a mine or a rock. 

### so how this is done :
We are using sonar data which is mounted on submarine to detect signals from sea or ocean bed. 
what is Sonar : Active sonar transducers emit an acoustic signal or pulse of sound into the water. If an object is in the path of the sound pulse, the sound bounces off the object and returns an “echo” to the sonar transducer. If the transducer is equipped with the ability to receive signals, it measures the strength of the signal. and this case signals coming from sonar is processed to detect whether the object is a mine or it's just a rock. We are using dataset available publicly on kaggle. 
### We are using this dataset to feed our machine learning model and then our machine learning model can predict whether the object is made of metal or it is just a rock. 
So this is the principle we are going to use in our prediction, first we need to collect the data once we have the data we need to process the data because we cannot use the data directly, but in this case we are having data without lable so we have to pre-process the data before use.

In this case we are going to use a Logistic Regression model because logistic regression works really
well for binary classification problem so this problem is a binary classification problem because we are
going to predict whether the object is a rock or a mine. So we'll be using a logistic regression model
and this is a supervised learning algorithm. We need to train our machine learning
model with the training data we will get a trained logistic regression model so this model has learned from the data
on how  model can recognize it based on the sonar data, when we give a new data it can predict whether the object is
just a rock or it is a mine so this is the workflow we will be be following.
