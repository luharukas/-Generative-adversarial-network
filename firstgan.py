import torch
import torchvision
import numpy
import torch.nn as nn
import math
import random

class Generator(nn.Module):
    def __init__(self,input_length):
        super(Generator,self).__init__()
        self.dense=nn.Linear(int(input_length),int(input_length))
        self.activation=nn.Sigmoid()


    def forward(self,x):
        return self.activation(self.dense(x))

class Discriminator(nn.Module):
    def __init__(self,input_length):
        super(Discriminator,self).__init__()
        self.dense=nn.Linear(int(input_length),int(input_length))

        self.output=nn.Linear(int(input_length),1)
        self.activation_sig=nn.Sigmoid()
        self.activation_relu=nn.ReLU()


    def forward(self,x):
        x=self.activation_relu(self.dense(x))
        return self.activation_sig(self.output(x))

def generate_even_data(max_int=128,batch_size=16):
    input_length=int(math.log(max_int,2))
    listing=[]
    # even_number=torch.randint(0,2,size=(batch_size,input_length)).float()
    for i in range(batch_size):
        listing.append([random.randrange(0,2) for _ in range(input_length-1)]+[1])

    array=numpy.array(listing)
    array=numpy.reshape(array,(batch_size,1,input_length))
    even_number=torch.from_numpy(array)

    even_labels=torch.tensor([[1] for _ in range(batch_size)])

    return even_labels,even_number



    


def train(max_int=128,batch_size=16,training_step=2000):
    input_length=int(math.log(max_int,2))
    generator=Generator(input_length)
    discriminator=Discriminator(input_length)

    generator_optimizer=torch.optim.Adam(generator.parameters(),lr=0.001)

    discriminator_optimizer=torch.optim.Adam(discriminator.parameters(),lr=0.001)

    loss=nn.BCELoss()

    for step in range(training_step):

        generator_optimizer.zero_grad()

        noise=torch.randint(0,2,size=(batch_size,input_length)).float()

        generated_data=generator(noise)

        print("Generated Data",generated_data)

        true_labels,true_data=generate_even_data(max_int,batch_size)

        true_labels=torch.tensor(true_labels).float()
        true_data=torch.tensor(true_data).float()

        generator_discrimantor_out=discriminator(generated_data)
        # print(generator_discrimantor_out.shape)
        # print(true_labels.shape)
        generator_loss=loss(generator_discrimantor_out,true_labels)

        print("Loss of Generator",generator_loss)

        generator_loss.backward()
        generator_optimizer.step()


        discriminator_optimizer.zero_grad()
        true_discriminator_out=discriminator(true_data)
        # print(true_discriminator_out.shape,true_labels.shape)
        true_discriminator_loss=loss(true_discriminator_out,true_labels.reshape_as(true_discriminator_out))

        generator_discrimantor_out=discriminator(generated_data.detach())

        # print(generator_discrimantor_out.shape,torch.zeros(batch_size).reshape_as(generator_discrimantor_out).shape)
        generator_discrimantor_loss=loss(generator_discrimantor_out,torch.zeros(batch_size).reshape_as(generator_discrimantor_out))

        discriminator_loss=(true_discriminator_loss+generator_discrimantor_loss)/2
        print("Loss of Discriminator",discriminator_loss)
        discriminator_loss.backward()
        discriminator_optimizer.step()

train()


