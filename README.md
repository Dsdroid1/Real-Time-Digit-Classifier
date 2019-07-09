# Summer-Project-2019
Contains code,trained parameters etc for the NNs
This project was to explore the field of ML by implementing the 'Hello, World' of ML
i.e the MNIST Digit Recognition without using any external NN library(like tensorflow,keras etc)
It required studying the NNs from the base , knowing how their implementations work.
This contains all the basic functions written in Python for forward pass,cost function,backward pass and parameter update.

Later when the standard NN was implemented,we shifted to the CNNs.
For this, in assignments we wrote the code for forward and backward pass involving convolutions and pooling
but while implementation we did it in Pytorch(lib for NN).
This was done to ease the work and helped learn Pytorch from its documentation.
We did learn how the convolutions work in forward as well as in backward pass.
So with that knowledge,we were ready to move on to a NN library.

Thus ,we implemented the MNIST digit recog. via CNN to get a better accuracy(as expected). 

The thrid file(Opencv one) is implemented using CNNs and uses the trained model(stored in the CNN folder) to predict on images.
The difference?
We draw those images using 'mouse as a paintbrush' via OpenCV!!!

Gives a pretty good accuracy.One issue which is currently being faced is that while drawing '1',
if the tilt of 1 is bit more than the vertical on either side(maybe a margin of 5 deg.),then it is predicted as an 8.
Rest is working fine.
Also dont use the fancy script to draw the numbers in OpenCV(baseline and tilt at top in 1,middle cut in 7)
They decrease the prediction accuracy significantly.

The fourth file is even better than the previous one.It involves drawing the digit with a 'bottlecap'in front of the 
webcam.Then it predicts which digit was drawn.
The file has 2 modes for calibrating the 'stylus'('Calibrate') or using previously stored values('Read').It is recommended that calibration is done whenever lighting conditions change for better performance.(In calibration, only the stylus should visible.If noise exists try changing stylus and/or background).
Further, 2 modes exist for input as well, 'maunal' and 'auto'.The manual mode involves the user to select when an image is complete,when to send this image to predict('SPACE') and when to clear the screen('c').
The auto mode considers the image to be ready when it detects no stylus(no.of contours with 'conditions' zero).
It also clears the screen at that instant and simultaneously predicts.Refer the video for better understanding of this part.

THE END
