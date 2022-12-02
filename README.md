# Submarine-RockVSMine-Project

## This Case Study Objective : 
This use case consider that is a submarine and there is a war going on between two countries so submarine of a country is going underwater to another country and the enemy country have planted some mines in the ocean. So mines are nothing but the explosives that explodes when some object comes in contact with it right so there can also be rocks in the ocean.
## The submarine needs to predict whether it is crossing a mine or a rock. so our job is to make a system that can predict whether the object beneath the submarine is a mine or a rock. 
so how this is done is
the submarine since a sonar signal okay sorry a sonar uses a sonar that sends sound signals
and a review switchbacks so this signal is then processed to detect whether the object is a mine or it's just a rock
in the ocean okay so let's try to understand how we are going to do this first of all let's see the workflow for
this project first of all we need to collect the sonar data okay so how this
data is collected so what happens is in the laboratory setup an experiment can be done where the
sonar is used to send and receive signals bounce back from a metal cylinder
and some rocks okay so because the mines will be made of metals right so we collect this data
which is nothing but the sonar data which is which we obtained from a rock and a metal cylinder okay and we use this sonar data and
we feed this sonar data to our machine learning model and then our machine learning model can predict whether the object is made of metal or
it is just a rock so this is the principle we are going to use in our prediction okay so first we need
to collect the data okay so once we have the data we will process the data so we cannot use the data
directly so we need to pre-process the data so there are various steps in that which we will do in this
video so we need to analyze the data so we need to understand more about it so once we process the data
we will split it into training and test data okay so why we are splitting the training and test data because so let's say there are unread
examples so under instance of data we will train our machine learning model with 90 uh you know instances
and we will test our uh machine learning model so we evaluate our model with another 10 data points okay so
there are totally 100 data points we may use 90 data points for training and we can use another 10 or 20 data
for testing our model okay so once we split our data into training and test data we feed it to our machine learning model
so in this video in this use case we are going to use a logistic regression model because logistic regulation works really
well for binary classification problem so this problem is a binary classification problem because we are
going to predict whether the object is a rock or amine okay so we'll be using a logistic regression model
and this is a supervised learning algorithm okay so once we uh train our machine learning
model with the training data we will get a trained logistic regression model so this model has learned from the data
on how a metal cylinder will be and how rock will be so this model can recognize it
based on the sonar data now when we give a new data it can predict whether the object is
just a rock or it is a mine so this is the workflow we will be uh following in python to
make this you know script for this use case okay so now let's go into the coding
part so before that i'll show you the data via okay
so this is the data sonar data.csv so it is in a csv file so you can find this data in kaggle and
other data sites like uc machine learning repositories and other sites so i will be giving the link
for this file in the description of this video let's try to look at this video sorry in
this uh data set so as you can see here there are a lot of numbers it's so there are a lot of columns so
let's see there are how many instances so there will be almost 200 examples so
as you can see here there are do not eat examples that means do not eat data points so on the in the last column
we have something that tells you know r
and em okay so r represents the values for rock and m represents the
values of mines okay so as i told earlier this values are obtained in a laboratory
setup where the sonar data is collected for a metal cylinder and for a rock so as you can see here there are several
features so features represents the columns of this data set so we feed this dataset to
our machine learning model and it can learn from this data on how a rock can be and our our mind
can be with the help of this sonar data okay so let's see how we can make this python
script so close this
so as i told earlier we will be doing our programming in google collaboratory so search for google collab
so this is the website for it so collab.research.google.com
okay so you just need to choose new notebook here so this google collaboratory will be
linked to your google drive account so if you have any collaboratory files so it will show up here so i'm going to use a new notebook so
these are nothing but uh python notebooks as you may have noticed this in jupyter notebooks so it it has an
extension of ipy and b so it is just like jupiter notebooks
so as you can see here this is a dot pynb file which is known as python notebooks so i will change this file name as
rock versus mind prediction okay now you can see this connect button
here so go ahead and connect this so what happens is our runtime gets connected to a google
server so it is completely free google collaboratory is completely free so you will be allocated a system of a
very good storage size and very good ram so as you can see here so we have 12 gb
of ram and gb of storage which is really good
so it is better than most of our systems so we will be doing all our programming in google collaboratory okay
okay so as you can see here this is called as a cell so we will write our python scripts in
this sales so as you can see here you can give this code option to create another cell
so if you give this text you can write some comments or a description about your code okay so i will tell about the features
of this google cloud once you know we use it for different purposes so
so as you can see here this is where we will upload our data files so i have already shown the sonar data
file for you so how you can upload this so you can click this folder option here
so there either you can choose this so it is to upload a particular file or you can right click here and click
upload so i'll upload the sonar data
okay so it is a very small data file
so you can find it in kaggle or you see machine learning depository okay so as i told here our agenda here
is to predict whether the object is a mine or a rock using the sonar data so
first of all we need to import the libraries we want to import the dependencies so we will require several functions and
libraries for this so i'll write a comment here so as i told you told you earlier so this is for writing
description and comment about your code so just type importing the dependencies
okay so you can press enter or you can place press shift press enter to complete it
and go to the next cell okay so once you write a python script you can click here
to run your code or you can press shift plus enter to run this code and go to the next cell
okay so first we need to import some libraries so we will require numpy for this so i'll import numpy smp
import numpy smp and we also need pandas so import pandas
as pd okay so numpy is basically for uh you know for arrays and pandas is for
several data processing steps so we will see about that later now we need
a train test split okay so we have seen earlier that we need to split our data into training
data under test data so we will require a function for that so from sp
loan dot modern selection
input train test split okay so this function is used to split
our data into training and the test data so we don't need to do it manually okay so then we need our logistic regression
model so sqlon is a very good uh python library for machine learning algorithms and other functions so we
will encounter it in various different uh places here
there is a small mistake here scale on dot model selection so input train test
split now we need to import our logistic regression model so scale learn
dot linear model so this is how you can import logistic
regression so input logistic regression
and we need the function accuracy score so from scale on
dot matrix
import accuracy score so this is used to find the accuracy of our model okay so these are the libraries and
functions we need so first we import numpy so number is used for creating numpy arrays so and pandas so pandas is used for uh
loading our data loading our data and numbers into a good table so these tables are called as data
frames so we will discuss about this that in a later point so then we have a train test
split so we import it from the library scale on then we have imported the logistic regression model then we have
imported the function accuracy score so you can press shift press enter to run this cell and go to the next cell okay so if
you have any printing output it will show here okay now let's do the data collection
and processing steps so i'll just put a text here
data collection and data processing
okay so we already uploaded the data there are several methods to you know upload data in google
collaboratory we can uh upload the data straight to collaboratory using some apis so
we can do it with kaggle apis so we'll discuss about it in some other project video
so as you can see here we have the sonar data file here so now we need to import this sonar data into a pandas
data frame okay so i'll make a comment so in python if you write something
prefixed by hash you can comment it so loading
the data set to a pandas data frame
so i'll create a variable called as sonar data i will be loading this uh data to a data
frame and i have named this data frame as sonar data so as you can see here i have imported pandas as pd so this is just like an
abbreviation so i'll be using this application so pd dot read csv so as you as i have told you earlier we
have the data file as a csv file so we need to use the function read csv
okay so now we need to mention the name of the file name on the
location of the file so you can do it by you can go here and click here so there you will see this
option called as copy path so this will copy the path and name of the file
so we have to enclose it in quotes and put it in this brackets then so
as you can see here we don't have a header file for this
so as you can see here we don't have any editor files so either files means a name for the columns right so
there is no header file so we need to mention in our pandas data frame that there is no
header so editor is equal to none
so i'll press shift plus enter and this loads our data to a pandas data frame okay
now let's uh have a small look at our dataset so i'll just type sonar data
dot yet so what this function it does is it displays the first five uh
rows of our data set okay so i'll run this so as you can see here we have uh first
five rows of our data set and there are several columns so as you can see here there are 59
columns but actually it's 60 column because in python the indexing starts from zero so totally we have 61 columns and 59
columns we have 59 features and we have in the last last column we have uh r
or m so i have shown you right it's either rock or mine so it is that categorical value
so this is the use of this get function it tries to you know it prints the first five rows of our data set now what we
will do is let's try to see how many rows and columns are there
so number of rows and columns
so if you if you are not you know if you don't understand any functions you can just search in google about
let's see so you want to know what this read csv function does so you just can
go to the pandas documentation so pandas dot read csv
so this is the pandas official documentation page so you can go here
so you will uh so you can see the use of this uh particular function here so as you
can see here it read a comma separated value csv file into a data frame it supports optional iterating or breaking
the file into chunks so you can do this for any functions so in order to learn what this function actually does
so these are the parameters so we don't need all these parameters so in our case we have used only two parameters which are
the path of the file and we have included that there is no adder okay so if you have any doubt about any
function you can search it in google like this okay so now we need to find the number of rows and columns
so we can use it you can we can find it by using the function called as shape so sonar data
dot shape so this gives us how many uh rows and columns are there so totally
we have two not eight columns and sorry two not eight rows and sixty one columns the last 60 first column tells us whether it is a
rock or a mind and there are totally two not eight rows so two not eight rows means there are
two naughty instances or examples of data okay so on 61 represents the feature so
let's say for this zero zeroth instance so it is a value for one
rock and there are 60 features for this one rock and it is labeled as har okay
so like this there are two not eight instances now what we will do is let's try to get
some [Music] um statistical definitions for this data
so shown our data dot descript so this
gives the mean standard deviation and other parameters for our data
sorry just made a small mistake here
so now data.describe so as you can see here it gives the count so account represents
the number of uh instances we have for the 0th column so like this we have uh
all the way up to 59th column so it gives us the count so the number of uh values we have the mean of this
column standard deviation for this column minimum value 25th percentile 50th percentile 75th percentile and what is
the maximum value so percentile means like uh 25 percentage of the values are less than
this 0.0133 for first columns and 50 percent means 50 percentage of the
values are less than 0.022 so that is what percentile is so for some use cases uh it is really important
for us to find this mean and standard deviation it it gives us a better understanding of the data so hence you can use this
describe function to get some statistical measures so i'll just make a comment here so
describe gives statistical measures
of the data okay now
let's try to find how many examples are there for rog and how many examples are
there for mains okay so we can do that by sonar data
dot value counts so this value counts function uh
gives us how many rock and how many main examples are there okay so i have to include one more thing
so we just need to put a 60 here
the 60 is nothing but the column index so as you can see here the rock and mines are
specified in the 60th column so i'm specifying 60 in this value
count so value count function okay so why i am using sonar data because i
have loaded this data frame into a variable called as sona data so that's why i'm using this sonar data
and uh including the function with this okay so now let's see how many rock and mine
examples are there so as you can see here there are totally 111 examples are there for mine and 97
examples are there for rock so it is almost equal so it is not a problem so if we have uh data for one type of
instance more let's say for example if we have a thousand examples for mine and we have only 500 examples for rock then our
prediction won't be very much good okay so if we have almost equal number of example for both the two
categories so our prediction will be really good and we will get a very good accuracy score and our model
performs well okay so it is almost uh equal here so actually uh speaking so
totally there are almost 298 instances but this is not sufficient for a machine
learning model so we may require even thousands and several thousands of examples for making
a better model but we are just looking at some examples so we are okay with this so
you just need to note one thing more the data more accurate your model is okay so i just represent here that
m represents mine and r represents rob
okay now let's try to group this data based on
mine and rock so shown our data
dot group by 60
dot me so now i'll explain you what what is the use of this so as you can
see here so we got the mean value for
all the columns for mine for a mine we have the mean value for 0th column 0.034 but
for rock it is 0.022 as you can see here there is quite a difference between these two so
this difference is really important for us because using this only we are going to predict whether the object is either a mine or a rock
okay so we just found the mean for each of the 59 columns sorry 60 columns okay so the mean value
for mine is these values and for rock is these values and there is a quite difference between them okay so this is
really important for us now let's try to
um separate the data and the labels okay so here the
data i mean the these numerical values so these are the features and the last column is the label so we
need to separate this so we are going to do that let's see so i'll just make a comment here
separating data and label so this is a supervised learning problem
where we train our machine learning model with data and labels so in unsupervised learning we don't use labels so here we
have labels which are nothing but rock and mine okay so let's put all the
data in the variable x so so now sona data so i am going to
drop the last column 60th column so data dot drop
columns is equal to 60 so i am dropping 60th column and if i am dropping a column i need to
specify the axis as one so if i am dropping a row i will be specifying access as zero okay
and let's put the label in the variable y so sonar data
so we need to use a square bracket here 60 okay so what i'm basically doing here is i'm storing all
the values in x except the last column so i'm dropping the 60th column
and i'm storing storing the 60th column in y okay let's try to print and see this
so print x print y
so as you can see here now there are only 59 columns so actually it's 60 column and it starts with zero
and the last label is in the variable called space so we have successfully splitted
the data and the labels okay now what we will do is we will try to split this data
into training and test data okay so let's include a text here training
and test data
so as we have already uh imported the function drain test split so we will be using this function
to split our data okay so we need to give some
variable names here extreme so you can give any names here so for
convenient purpose i'm giving this name so explain x test
y train and y test so this order should be followed so
first we will take the training data on the test data then we will take the labels of training data labels
of test data is equal to
train test split and we have to include this x and y here so we are going to split this x and y
into training and test data so x comma y so there are several
parameters here so i'll explain you about that so x comma y test size
let's have the test size as 0.1 and stratify
stratified is equal to y and random state
c equal to let's say okay so now let's try to understand about these parameters so we are going to split our data into x
train and x test y train and y test so x train is nothing but the training data for this and
extess is the testing data and white train is the label of those training data and y test is the label of the test
data okay so now we are using the function train test split then in the parameters we have included
x and y so we are going to split this x and y into training and test data so here we have the parameter test size
so test says like if we give 0.1 means we need 10 percentage of the data to be test
data okay say for example we add uh almost 200 examples so what happens is like
10 percentage of 200 is 20 so we will have 20 test data so that is the use of this
test size you can use 0.1 or 0.2 so based on the number of data you have okay so here we will take
just 10 percentage of our data stream the stratify is equal to why so stratify why we are using this stratifies
yes we need to split the data based on rock on mine say for example we need to have equal almost equal number of rocks
in tested training data and equal number of mines in the training data okay so hence we include this stratify
so our data will be splitted based on the number of uh these two rock row conveying
okay and then we have random state so random state is to split the data in a particular order
say for example if you do the same code and in the code you include one your
data will be splitted uh you know in the same way as my data splitter so if ah i put two here so my data will be
splitted in a specific way and if you include two in your code it will also be splitted in the same way it
is basically to reproduce the code as it is so i'll use one okay so now we can
split our data okay now let's try to see how many uh
training data text test data are there so print x dot shape so it is the
original data without splitting into train and test and then we have
extreme dot shape
and fix test dot shape so i'll run this so as you can see here
in the original x we have two not eight examples and in the training data we have 187
instance and in the test data we have 21 instance okay so we have 21 test data and 187
training data now we need to train our machine learning model with this extreme with this training data okay now let's see how we
can train our model so model training
so we will be using a logistic regression model
so i'll create a variable called as model so as you can see here we have already imported the logistic regression
function here so model is equal to
logistic regression so this will load this logistic regression function into the
variable model now we are going to train the models for training the
logistic regression model
with training data
okay so for that we use the function model.fit here we need to include the training
data and training label okay which are extreme and
white ring okay so let's see what is this extreme unwind train once
so that you can understand it so i'm printing
extreme
and also white rain so this is the training data so exchange is the training data and white line is the
training label so as you can see here we have the data here so totally there are 187
examples and 60 columns and this is the label so as you can see here it is kind of random because we have used the training
testing spread okay so now we are going to feed this training data on this training data label to our logistic
regression model so that's why i've included model.fit xtrain and white rain so when i run this our model will be drained
so if you are having a lot of data and if you are using a
complex model like a neural network it will take a lot of time so here we are seeing a simple example
and simple data so it doesn't take much time okay so our model will be trained so
these are some parameters of our model now let's see how our model is predicting so now
let's check the accuracy score of our model so now we are going to evaluate our
model so model evaluation
so we have imported the function accuracy score for it so we will use this function to find the
accuracy of our model
accuracy on training data so first let's find the accuracy on the
training data
so what's happening here is so we will use this same data the data which the
machine learning model has learned so we will try to use this model to predict these examples
okay so then we will use the test data okay so what is the significance here is
the model has already seen what is this test data sorry the training data but it
it haven't seen what is this test data okay it is just like preparing for exam let's say you are
preparing all the example problems in a mathematics book for your exam so those example problems are nothing
but the training data so in the exam a new problem will be asked and you need to solve that
but but you have never seen that question right so that is nothing but that is data so
we need to test our model for accuracy on training data and accuracy on test data so it is
always in most of the cases the accuracy on training data will be more because the model has already seen uh
this training data and uh most of the times the accuracy and test data will be less okay so if the accuracy of your model is
somewhere greater than 70 which is it is actually good so and it also depends on
the amount of data you you use so as i have told earlier so if you use uh you know many data points and if you
use a better model you will get a good accuracy score okay so if we use uh you know quite less
number of data as we have in this case where we have only 200 data points our accuracy can be low okay so but the
idea of this you know video is for you to understand how you can implement these steps so
the accuracy is not that much important but we have to note here is so any accuracy greater than 70
percentage is good okay so now let's try to predict the training data
extreme prediction
is equal to model dot predict
next extreme
training data accuracy so we will store this accuracy value in this variable training data accuracy
so training data accuracy is equal to accuracy score
so this is the extreme prediction is the prediction our model makes okay based on uh its learning
so we need to include extreme prediction and all the correct values it is white
ring so okay so what happens here is we are going to compare the we are going to get the accuracy score
so extreme prediction is the prediction of our model and y train is the real answer of
our answer of uh this instances so as you know here so we have uh we have this test data
right so x test and y test is the label of this test data so now what happens is
we are going to compare the prediction of our model and the original label of these data
okay by that we will get the accuracy score so let's try to get the accuracy score for
the training data so i'll print the accuracy score
so accuracy on training data
and copy this so this will have the accuracy score here
training data accuracy so as you can see here we got almost 83.4 percentage of accuracy
so it is actually kind of good for these many data so now let's try to find the accuracy score
of test data okay so it will be the same part except for some changes
we just need to include the test data here so accuracy on test data so x test prediction
test data accuracy so the model has never seen this data okay so model dot predict x test
okay so y test
so now we are using our model to predict the test data and this uh prediction of this model
will be compared to the original labels which is why test okay is data accuracy
now we need to print this accuracy score
so accuracy on test data
is data accuracy so we got a 76 percentage as accuracy
score which is really good so which means out of 100 times it can predict 75 times
the correct object whether it is a rock or mine okay so we got accuracy score as 83
percentage for training data and 80 76 percentage for test data so our model is performing fine
okay so now let's what we are going to do is so we have a trained uh logistic regression model
and now we are going to make a predictive system that can predict whether the object is either rock
or it is mine using that sonar data okay so now let's see how we can make this predictive system
so making a predictive system
so we need to give the input data so i'm making a variable called as input data
okay so in this parenthesis we need to include the sonar data so we have seen this sonar data right
so we will be taking some examples for rock and mine and we will check whether our model is predicting uh the rock and mine
correctly okay so this is the use of this code snippet so once i complete the script i'll uh
copy some values and put it and let's see whether it's predicting correctly so input data so once we get the input
data we have to convert it to a numpy array because the processing and non numpy array is faster and it's more
uh easy so
it's basically changing the data type to a numpy array so at least to do a numpy array so
changing the input data so just make a comment here so changing
the input data to a numpy array
so we will use the function numpy.s array for this function so input data
as numpy array so this is the variable i am giving it
so input data's numper is equal to np dot s array so if you remember we have imported the
library numpy as np so i am using np instead of numpy okay so np dot ask
array input data
so basically we are converting this list into a numpy array
and let's let's take an example say for example um
i'll open this from our notepad
okay so let's take some random example okay let's take it i think it is a rock
so as you can see here this example is stock so if we feed this data to our machine learning model so it should predict that
this is a rock okay so i'll copy this
okay
so i'll put this in this input data so we have totally 60 features
so i have converted this input data so this is basically a list so we are converting it to a numpy array okay now we need to reshape
the data okay so because we are just predicting for uh one
instance so for that purpose we need to reshape this array otherwise our model can be confused by the number
of data points so we need to reshape the numpy array
as we are
predicting for one instance okay so
input data reshape is equal to
i'm copying this input data's number so we need to reshape this
reshape one comma minus one so
this one comma minus one represents there is one instance and we are going to predict uh the label for this one instance so that
is why we are reshaping this so once we reshape it we need to make the prediction so i will create a
variable called as prediction and i'll store the prediction of our model so model dot predict function
is used to edit it so we have stored our trained logistic regression model in the
variable called as model so as you can see here so i am calling that model function so model variable so model.predict
this input data reset so this contains
the features of our data okay so this data is present in input data reshape okay so basically
what happens is that this model dot predict returns either r or m as value
okay so it uh you know tells us whether it is either a rock or a mine okay now let's
try to print this prediction
so let's try and run this so you should predict that this object represents a rock because we have copied the data for a log right
so i'll run this and yeah it predicted correctly that the object represents rock from this
example okay and let's include a if condition here
here that if we get r as our prediction it should say that the object is a rock and if we get m the
object is a mine okay so as you can see here
this r is included in a list so if this prediction is equal to r
prediction 0 is equal to r
so i am using this 0 because this is 0 represents the first element of the list here the list is this
prediction okay so the first element is r and we need to represent it with this index as 0 so
that's why i'm using 0 as the index here so if it is not in a list i won't
include this as 0 okay so if the first element of this list is equal to r
we need to print that the object is rob
okay otherwise so else
we need to print it as a mine so when it is m we need to print it is the object smile
okay let's try to run this so as you can see
here so we get the first element in the list as r so it tells us the object is a rock and we
know that we copied the data of a rock now let's try to see whether our model is predicting
the mind correctly so let's try to get some random value for mine and let's try to
let's find whether our model is predicting correctly i'll take this value so this is this value represents mine as
you can see here there is an m so i'll copy this value let's see whether it's predicting
correctly so if if our model is working correctly it should say that uh it is m which means that the object is a
rock so i'll just replace this
okay now let's try to run this now it should say that say that the object is a mine so as you
can see here it is predicting correctly that the object is a mine so this is how our predicting system works so i hope you have
understood all the process we have done here so i'll just give you a short recap of how we are doing this
so first we have imported the dependencies so numpy is used for uh making arrays and pandas is used for
making a data frames and we are using the libraries scalon for using the function drain test split so
it is used to split our data into training and test data and in this case we are using a logistic regression model so we are
importing that logistic regression function from sklearn.linear model and we are importing the accuracy score
to find the accuracy score of our model from scaling.matrix then data collection processing so we
have imported the sonar.csv file into our google collab environment
okay so now we are feeding it to a pandas data frame by pd.read csv
function so and i am storing that data frame in a variable called as sonar data so here we need to give the path of the
file and since we have no editor in this file so we have to mention that there is none and by the function yet we are just
printing the first five columns of our pandas data frame and we find that the last column is a categorical
column which says whether it is a rock or a mine so r represents stock and m represents mine then we are determining the number of uh
rows and columns we have so we have two not eight columns which represents two not eight data points so and uh 61
features okay so 60 features on one label which is rock or mine then we have used the function
sonar data.describe which gives us the count the number of values we have mean and standard
deviation and other statistical values then we are counting how many main examples are then
how many rock examples are there and we find out that it is almost equal and then and then we are grouping the
data based on mine and rock and we find their mean values and
we get a quite difference in their mean values okay so as you can see here there is a difference in mean value of
rock and mine for each of the column okay so now we are splitting the data into all the features
and all on the labels so we are feeding all the features to the variable x and
all the labels to the variable y okay now we are splitting our uh x and y our data into training and test
data okay so the training data is used to train our model and our model is evaluated with the help
of test data right and then we are loading our logistic regression model in the variable model
and uh by the function model dot fit our model is trained it is just like a graph okay so in the x axis there will
be this features under y-axis there will be labels okay and uh the graph will be plotted so this is how model is trained
so once we have trained our model using the function model.fit we are finding the accuracy score so first we
find the accuracy score of uh training data which is around 85 84 percentage
and then we find the accuracy score for the test data which is around 76 percentage then we are making a predictive system
where if we give the features if we give the data it can predict whether the object is a rock
or mi okay so this is so these are these are all the procedures we have used in
this use case so i hope you are clear with all these so i will be giving uh the
link for the sonar data so the data file and also this follow-up file in the link of this
link in the description of this video so you can download it from here so do try google
collab so try to do all these things we have done here okay so try to write all this python script by
your own and try to understand it so if you have any doubt if you if you run into any problems uh
mentioned in the comment i will try to solve your problem okay so that's it from my side i hope you have understood the
topic we are covered in this video so i'll see you in the next video thanks for watching

