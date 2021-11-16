# ECG-Smart-Machine
This is the repo for my senior design project to develop an AI that would be capable of determining blood serum potassium values given an ECG.

Aspects of the program:
  1. The project used the pytorch library for creating the AI instance that would analyze the incoming image.
  2. Each image was processed to standardize the input file.
  3. Signal processing was handled using matlab to transfer the ECG into a digital format and then convert the signal into a visual graph.
  4. The PQRST complex was used as the focal point of our research and how subtle changes in these values in the graph correlate to different conditions.
  5. We found that the QRS complex and T wave as well as few other variables were distinguishable when the conditions of hyper- and hypokalemia were present.
  6. Our data set for training came from a series of medical documentation of these two conditions in addition to normal state graphs.
  
Things to note:
  1. The final version was never tested using a patient dataset, although attempts were made to acquire access.
  2. The signal processing of an ECG signal and converting it into a usable graph still requires further tweaking.
 
     a.  Currently, we were able to get readings from our ECG we made though the signal was experiencing some noise.
     
     b.  Due to the noise and variance of signal output, our ability to transfer the live data into the CNN was limited.
     
     c.  We did try to use a standard ECG algorithm for this purpose and did get an acceptable output to feed the CNN, but did not get enough    to train.
  3. The current state of the CNN is operational, but please consider these above points as they do pose potential problems with furthering of this project.
