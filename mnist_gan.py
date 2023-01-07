import torch
from tqdm import tqdm
import torchvision
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import transforms
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

writer=SummaryWriter(log_dir='./runs')

EPOCHS=15
BATCH=1000
criterion=nn.BCELoss()
loss_g=[]
loss_d=[]
images=[]


transform = transforms.Compose(
    [
        transforms.ToTensor(), 
        transforms.Normalize((0.5,), (0.5,))
    ]
)

train_data=datasets.MNIST(
    root='./MNIST',
    train=True,
    download=True,
    transform=transform
)

train_data=DataLoader(
    dataset=train_data,
    batch_size=BATCH,
    shuffle=True,

)

samples_image,sample_label=iter(train_data).__next__()
image_grid=torchvision.utils.make_grid(samples_image)

writer.add_image(tag="MNIST DATA",img_tensor=image_grid,global_step=0)
writer.add_text("Batch Size",f"Batch Size of model {samples_image.shape[0]}")
writer.add_text("Shape of Image",f"Shape of a image {samples_image.shape[1:]}")
writer.close()

class Generator(nn.Module):
    def __init__(self,image_channels):
        super(Generator,self).__init__()
        self.model=nn.Sequential(
            nn.ConvTranspose2d(image_channels,64*4,4,2,0,),
            nn.BatchNorm2d(64*4),
            nn.ReLU(),
            nn.ConvTranspose2d(64*4,64*2,4,1,1),
            nn.BatchNorm2d(64*2),
            nn.ReLU(),
            nn.ConvTranspose2d(64*2,1,4,1,1),
            nn.BatchNorm2d(1),
            nn.Tanh()
        )

    def forward(self,input_img):
        return self.model(input_img)


class Descriminator(nn.Module):
    def __init__(self,input_channel):
        super(Descriminator,self).__init__()
        self.model=nn.Sequential(
            nn.Conv2d(input_channel,32,4,1,1),
            nn.MaxPool2d(3,1,1),
            nn.Conv2d(32,32,4,1,1),
            nn.MaxPool2d(2,1,1,),
            nn.Conv2d(32,8,3,1,1),
            nn.MaxPool2d(2,1,1),
            nn.Flatten(),
            nn.Linear(6272,512,),
            nn.ReLU(),
            nn.Linear(512,1),
            nn.Sigmoid(),
        )


    def forward(self,input_img):
        return self.model(input_img)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


# print(f"Summary of Generator\n")
generator=Generator(100)
generator.apply(weights_init)
# summary(generator.model,(100,12,12),batch_size=16)
# print("\n\nSummary of Descriminator")
descriminator=Descriminator(1)
descriminator.apply(weights_init)

# summary(descriminator.model,(3,28,28),batch_size=16)


# Optimizers
generator_optm=torch.optim.Adam(generator.parameters(),lr=0.001,)
descriminator_optm=torch.optim.Adam(descriminator.parameters(),lr=0.001)


def true_labels():
    return torch.ones(size=(BATCH,1))

def fake_labels():
    return torch.zeros_like(true_labels())

def generate_noise():
    
    return torch.rand(size=(BATCH,100,12,12))

criterion=nn.BCELoss()


def train():
    print("Training Loop Started...")
    for epoch in tqdm(range(EPOCHS)):
        print(epoch)
        for i,data in enumerate(train_data):
            if i%10==0:
                print(i,"completed")
            descriminator.zero_grad()
            image=data[0]
            truelabel=true_labels()
            real_descriminator_out=descriminator(image)
            # if real_descriminator_out.shape==truelabel.shape:
            #     print("Descriminator Shape Pass")
            error_from_descriminator=criterion(real_descriminator_out,truelabel)
            loss_d.append(error_from_descriminator)
            # print(error_from_descriminator)
            # print("Error from Descriminator",error_from_descriminator)
            error_from_descriminator.backward()

            # D_x = output.mean().item()

            noise=generate_noise()

            fake_generator_out=generator(noise)
            if i%10==0:
                images.append(fake_generator_out)
            fakelabel=fake_labels()
            
            output=descriminator(fake_generator_out.detach())
            error_fake=criterion(output,fakelabel)
            loss_g.append(error_fake)
            # print(error_fake)
            error_fake.backward()

            total_error_descriminator=error_fake+error_from_descriminator

            descriminator_optm.step()


            generator.zero_grad()

            output=descriminator(fake_generator_out)
            error_from_gen=criterion(output,truelabel)
            error_from_gen.backward()

            generator_optm.step()


train()

print("Total Descriminator loss",loss_d)
print("Total Generator Loss",loss_g)

import shelve
with shelve.open('data') as db:
    db['loss_d'] = loss_d
    db['loss_g']=loss_g
    db['images']=images

db.close()


        







 