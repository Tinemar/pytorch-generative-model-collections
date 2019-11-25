import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import numpy as np

import DRAGAN
import resnet
import utils

use_cuda=True
image_nc=1
batch_size = 64

def visualize_results(G,batch_size,images):
        G.eval()
        tot_num_samples = batch_size
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
        samples = images
        samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                   'H:/pytorch-generative-model-collections/results/cifar10/DRAGAN'+ '.png')
# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
net = resnet.ResNet18()
net = net.cuda()
net = torch.nn.DataParallel(net)
checkpoint = torch.load("H:/pytorch-cifar/checkpoint/DataPackpt.pth")
net.load_state_dict(checkpoint['net'])
target_model = net
target_model.eval()
# load the generator of adversarial examples
pretrained_generator_path = './models/cifar10/DRAGAN/DRAGAN_G.pkl'
pretrained_G = DRAGAN.generator(input_dim=2352, output_dim=3, input_size=28).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in MNIST training dataset
transform = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
cifar10_dataset = torchvision.datasets.CIFAR10('H:/cifar-10-batches-py', train=True, transform=transform, download=True)
train_dataloader = DataLoader(cifar10_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
num_correct = 0

for i, data in enumerate(train_dataloader):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    try:
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
    
    except:
        break
    
    adv_img = torch.clamp(adv_img, 0, 1)
    visualize_results(pretrained_G,batch_size,adv_img)
    pred_lab = torch.argmax(target_model(adv_img),1)
    num_correct += torch.sum(pred_lab==test_label,0)

print('cifar10 training dataset:')
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(cifar10_dataset)),len(cifar10_dataset))

# test adversarial examples in MNIST testing dataset
cifar10_dataset_test = torchvision.datasets.CIFAR10('H:/cifar-10-batches-py', train=False, transform=transform, download=True)
test_dataloader = DataLoader(cifar10_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
num_correct = 0

for i, data in enumerate(test_dataloader):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    try:
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -0.3, 0.3)
        adv_img = perturbation + test_img
    except:
        break
    adv_img = torch.clamp(adv_img, 0, 1)
    pred_lab = torch.argmax(target_model(adv_img),1)
    num_correct += torch.sum(pred_lab==test_label,0)

print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(cifar10_dataset)),len(cifar10_dataset_test))
