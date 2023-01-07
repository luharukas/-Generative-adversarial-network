# Generative-Adversarial-Network
 Python Program of Generative adversarial network models using Pytorch Framework

**Steps are Following to make a GAN Network:**

**Step 1.** Load all libraries and Frameworks

**Step 2.** Declare all constants

**Step 3.** Make a generator Model

**Step 4.** Make a Descriminator Model

**Step 5.** Make function to generate random Noise

**Step 6.** Function to generate the original Input Data

**Step 7.** Func to generate True Labels

**Step 8.** Fucn to generate False Label

**Step 9.** Initialize optimizers and loss Function

**Step 10.** Run loop for the epochs

In the loop

**Step 11.** Set Generator Gradient to Zero

**Step 12.** Generate Noise and pass to generator

**Step 13.**  Pass the generator output in the above step to descriminator

**Step 14.** Generate True Data and True Labels

**Step 15.** Find loss of the descriminator out in the step 13 and true labels of step 14 

**Step 16.** BackwardPropagate of generator 

**Step 17.** Set Descriminator Gradient to Zero

**Step 18.** Pass true data (from step 14) to descriminator 

**Step 19.** Calculate error of the (Result from step18 and true labels from step 14 )

**Step 20.** Pass the generator output to descriminator after detaching it from its gradient

**Step 21.** Calculate the error of the above result and fake data labels

**Step 22.** calculate the total error of comes in step 21 and step 19

**Step 23.** Backpropagation of Descriminator

For loop Ends



## Description of Files and Folders

1. Filename: Firstgan.py

Problem Statement : Make a GAN model to generate the binary Data point which is even. 

2. Filename: mnist_gan.py 

Problem Statement: Make a GAN model to generate the MNIST Dataset.

3.


