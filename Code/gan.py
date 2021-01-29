import torch
from torch import nn, optim
from torch.autograd.variable import Variable

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# Preprocessing
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(0.5, 0.5)])

# Training data
train_set = datasets.MNIST(root='D:\Atik\pythonScripts',
                                       train= True,
                                       download=True,
                                       transform=transform)
train_loader = torch.utils.data.DataLoader(train_set,
                                           batch_size=32,
                                           shuffle=True)

# Labels
classes = [str(i) for i in range(0,10)]

# Discriminator class
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.model(x.view(x.size(0), 784))
        out = out.view(out.size(0), -1)
        return out.cuda()
        
discriminator = Discriminator()

# Generator class
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 784),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.view(x.size(0), 100)
        out = self.model(x).cuda()
        return out

generator = Generator()

# If GPU with CUDA
if torch.cuda.is_available():
    print('Using CUDA')
    discriminator.cuda()
    generator.cuda()

# Loss function and optimizers
lr = 0.0001
num_epochs = 40
num_batch = len(train_loader)

criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr = lr)
g_optimizer = torch.optim.Adam(generator.parameters(), lr = lr)

# Convenience function for training our Discriminator
def train_discriminator(discriminator, real_images, real_labels, fake_images, fake_labels):
    discriminator.zero_grad()

    # Get the predictions, loss, and score of the real images
    predictions = discriminator(real_images)
    real_loss = criterion(predictions, real_labels)
    real_score = predictions

    # Get the predictions, loss, and score of the fake images
    predictions = discriminator(fake_images)
    fake_loss = criterion(predictions, fake_labels)
    fake_score = predictions

    # Calculate the total loss, update the weights, and update the optimizer
    d_loss = real_loss + fake_loss
    d_loss.backward()
    d_optimizer.step()
    return d_loss, real_score, fake_score

# Convenience function for training our Generator
def train_generator(generator, discriminator_outputs, real_labels):
    generator.zero_grad()

    # Calculate the total loss, update the weights, and update the optimizer
    g_loss = criterion(discriminator_outputs, real_labels)
    g_loss.backward()
    g_optimizer.step()
    return g_loss
    
for epoch in range(num_epochs):
    for n, (images, _) in enumerate(train_loader):

        # (1) Prepare the real data for the Discriminator
        real_images = Variable(images).cuda()
        real_labels = Variable(torch.ones(images.size(0))).cuda()

        # (2) Prepare the random noise data for the Generator
        noise = Variable(torch.randn(images.size(0), 100)).cuda()

        # (3) Prepare the fake data for the Discriminator
        fake_images = generator(noise)
        fake_labels = Variable(torch.zeros(images.size(0))).cuda()

        # (4) Train the discriminator on real and fake data
        d_loss, real_score, fake_score = train_discriminator(discriminator,
                                                             real_images, real_labels,
                                                             fake_images, fake_labels)

        # (5a) Generate some new fake images from the Generator.
        # (5b) Get the label predictions of the Discriminator on that fake data.
        noise = Variable(torch.randn(images.size(0), 100)).cuda()
        fake_images = generator(noise)

        outputs = discriminator(fake_images)

        # (6) Train the generator
        g_loss = train_generator(generator, outputs, real_labels)




