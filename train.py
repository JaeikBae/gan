import os
import torch.nn as nn
import torch.utils.data
import torchvision
from torchvision import transforms
from torchvision.utils import save_image


# Hyper-parameters & Variables setting
num_epoch = 200
batch_size = 100
learning_rate = 0.0002
image_size = 32 * 32 * 3
num_channel = 3
dir_name = "GAN_results"
model_dir = "GAN_models"

noise_size = 100
hidden_size = 256


# Device setting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Now using {} devices".format(device))


# Create a directory for saving samples
if not os.path.exists(dir_name):
    os.makedirs(dir_name)

# Create a directory for saving models
if not os.path.exists(model_dir):
    os.makedirs(model_dir)


# Dataset transform setting
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                            std=(0.5, 0.5, 0.5))])


# CIFAR-10 dataset
CIFAR10_dataset = torchvision.datasets.CIFAR10(root='./data/',
                                                  train=True,
                                                    transform=transform,
                                                        download=True)


# Data loader
data_loader = torch.utils.data.DataLoader(dataset=CIFAR10_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)


# Declares discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(image_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

    def forward(self, x):
        x = self.net(x)
        return x


# Declares generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, image_size),
            nn.Tanh())

    def forward(self, x):
        x = self.net(x)
        return x


# Initialize generator/Discriminator
discriminator = Discriminator()
generator = Generator()

# Device setting
discriminator = discriminator.to(device)
generator = generator.to(device)

# Loss function & Optimizer setting
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)

# Restore model lastest checkpoint
if os.path.exists(model_dir):
    ckpt_list = os.listdir(model_dir)
    #find lastest checkpoint
    last = 0
    for ckpt in ckpt_list:
        if int(ckpt.split('_')[2].split('.')[0]) > last:
            last = int(ckpt.split('_')[2].split('.')[0])

    if len(ckpt_list) != 0:
        ckpt_list.sort()
        discriminator.load_state_dict(torch.load(os.path.join(model_dir, 'GAN_discriminator_{}.ckpt'.format(last))))
        generator.load_state_dict(torch.load(os.path.join(model_dir, 'GAN_generator_{}.ckpt'.format(last))))
        print("Checkpoints are restored")
        print("Last epoch : {}".format(last))

"""
Training part
"""
loss_for_graph = []
print("Training start")
print("=====================================")
print("Now training {} epoch".format(num_epoch))
print("Batch size : {}".format(batch_size))
print("Learning rate : {}".format(learning_rate))
print("=====================================")
for epoch in range(num_epoch):
    for i, (images, label) in enumerate(data_loader):

        # make ground truth (labels) -> 1 for real, 0 for fake
        real_label = torch.full((batch_size, 1), 1, dtype=torch.float32).to(device)
        fake_label = torch.full((batch_size, 1), 0, dtype=torch.float32).to(device)

        # reshape real images from MNIST dataset
        real_images = images.reshape(batch_size, -1).to(device)

        # +---------------------+
        # |   train Generator   |
        # +---------------------+

        # Initialize grad
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()

        # make fake images with generator & noise vector 'z'
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z)

        # Compare result of discriminator with fake images & real labels
        # If generator deceives discriminator, g_loss will decrease
        g_loss = criterion(discriminator(fake_images), real_label)

        # Train generator with backpropagation
        g_loss.backward()
        g_optimizer.step()

        # +---------------------+
        # | train Discriminator |
        # +---------------------+

        # Initialize grad
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        # make fake images with generator & noise vector 'z'
        z = torch.randn(batch_size, noise_size).to(device)
        fake_images = generator(z)

        # Calculate fake & real loss with generated images above & real images
        fake_loss = criterion(discriminator(fake_images), fake_label)
        real_loss = criterion(discriminator(real_images), real_label)
        d_loss = (fake_loss + real_loss) / 2

        # Train discriminator with backpropagation
        # In this part, we don't train generator
        d_loss.backward()
        d_optimizer.step()

        d_performance = discriminator(real_images).mean()
        g_performance = discriminator(fake_images).mean()

        # Save loss for graph
        if (i + 1) % 10 == 0:
            loss_for_graph.append([d_loss.item(), g_loss.item()])

        if (i + 1) % 100 == 0:
            print("Epoch [ {}/{} ]  Step [ {}/{} ]  d_loss : {:.5f}  g_loss : {:.5f}"
                  .format(epoch, num_epoch,  i+1, len(data_loader), d_loss.item(), g_loss.item()))

    # print discriminator & generator's performance
    print(" Epoch {}'s discriminator performance : {:.2f}  generator performance : {:.2f}"
          .format(epoch+last, d_performance, g_performance))

    if (epoch) % 5 == 0:
        # Save fake images in each 5 epoch
        samples = fake_images.reshape(batch_size, num_channel, 32, 32)
        save_image(samples, os.path.join(dir_name, 'GAN_fake_samples{}.png'.format(epoch+last)))
        # Save model in each 5 epoch
        torch.save(generator.state_dict(), os.path.join(model_dir, 'GAN_generator_{}.ckpt'.format(epoch+last)))
        torch.save(discriminator.state_dict(), os.path.join(model_dir, 'GAN_discriminator_{}.ckpt'.format(epoch+last)))

print("Training finished")
print("=====================================")
print("Now saving model")
print("=====================================")
# Save model
torch.save(generator.state_dict(), os.path.join(model_dir, 'GAN_generator_{}.ckpt'.format(epoch+last)))
torch.save(discriminator.state_dict(), os.path.join(model_dir, 'GAN_discriminator_{}.ckpt'.format(epoch+last)))
print("Model saved")
print("=====================================")
print("Now saving loss for graph")
print("=====================================")
# Load loss for graph
loss_for_graph_append = []
if os.path.exists(os.path.join(dir_name, 'GAN_loss_for_graph.txt')):
    with open(os.path.join(dir_name, 'GAN_loss_for_graph.txt'), 'r') as f:
        for line in f:
            loss_for_graph_append.append([float(line.split(' ')[0]), float(line.split(' ')[1])])
loss_for_graph_append.extend(loss_for_graph)
# Save loss for graph
with open(os.path.join(dir_name, 'GAN_loss_for_graph.txt'), 'w') as f:
    for loss in loss_for_graph_append:
        f.write(str(loss[0]) + ' ' + str(loss[1]) + '\n')
print("Loss saved")
print("=====================================")
print("Training finished")
print("=====================================")

# Plot loss for graph
import matplotlib.pyplot as plt
plt.figure()
plt.title("GAN loss for graph")
plt.xlabel("Step")
plt.ylabel("Loss")
plt.plot(loss_for_graph_append)
plt.show()