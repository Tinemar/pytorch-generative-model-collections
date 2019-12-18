import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import transforms as transforms
import numpy as np
import torch.nn.functional as F
import utils.image_utils as img_utils


import WGAN_GP
import resnet
import Utils
import cifar10.cifar_resnets as cifar_resnets
import cifar10.cifar_loader as cifar_loader
use_cuda = True
image_nc = 1
batch_size = 64


def visualize_results(G, batch_size):
    G.eval()
    tot_num_samples = 64
    image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

    # sample_z_ = torch.rand((64, 2352))
    sample_z_ = torch.rand((batch_size, 62))
    sample_z_ = sample_z_.cuda()
    samples = G(sample_z_)

    samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)

    samples = (samples + 1) / 2
    Utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                      'H:/adversarial_attacks/pytorch-generative-model-collections/results/cifar10/WGAN_GP.png')


# Define what device we are using
print("CUDA Available: ", torch.cuda.is_available())
device = torch.device("cuda" if (
    use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
# net = resnet.ResNet18()
# net = net.cuda()
# net = torch.nn.DataParallel(net)
# checkpoint = torch.load("H:/pytorch-cifar/checkpoint/DataPackpt.pth")
# net.load_state_dict(checkpoint['net'])
# target_model = net
# target_model.eval()
target_model, _ = cifar_loader.load_pretrained_cifar_resnet(
            flavor=32, return_normalizer=True)
target_model = cifar_resnets.resnet20()
#advtrain model
# target_model.load_state_dict(torch.load('./tutorial_fgsm.resnet32.000050.path.tar'))
target_model = target_model.cuda()
target_model.eval()
# load the generator of adversarial examples
pretrained_generator_path = './models/cifar10/WGAN_GP/WGAN_GP_G_best.pkl'
pretrained_discriminator_path = './models/cifar10/WGAN_GP/WGAN_GP_D_best.pkl'

pretrained_G = WGAN_GP.generator(input_dim=62, output_dim=3, input_size=32).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

pretrained_D = WGAN_GP.discriminator(input_dim=3, output_dim=1, input_size=32).to(device)
pretrained_D.load_state_dict(torch.load(pretrained_discriminator_path))
pretrained_D.eval()

# test adversarial examples in MNIST training dataset
transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
cifar10_dataset = torchvision.datasets.CIFAR10(
    '../cifar-10-batches-py', train=True, transform=transform, download=True)
train_dataloader = DataLoader(
    cifar10_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
num_correct = 0
label4 = [2]*64
target_label = torch.LongTensor(64).zero_()
# target_label = torch.LongTensor(label4)

target_label = target_label.cuda()
for i, data in enumerate(train_dataloader):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    z_ = torch.rand((batch_size, 62))
    z_ = z_.cuda()
    # try:
    # perturbation = pretrained_G(test_img)
    # perturbation = torch.clamp(perturbation, -0.3, 0.3)
    # adv_img = perturbation + test_img
    # adv_img = torch.clamp(adv_img, 0, 1)
    # except:
    #     break
    # try:
    # tot_num_samples = 64
    # image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))
    # test_img = test_img.cpu().data.numpy().transpose(0, 2, 3, 1)
    # test_img = (test_img + 1) / 2
    # Utils.save_images(test_img[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
    #                   'H:/adversarial_attacks/pytorch-generative-model-collections/results/cifar10/cifar10.png')
    adv_img = pretrained_G(z_)
    visualize_results(pretrained_G, batch_size)
    pred_lab = torch.argmax(target_model(adv_img), 1)
    print(pred_lab)
    num_correct += torch.sum(pred_lab == target_label, 0)
    # exit()
    # except:
    #     break

print('cifar10 training dataset:')
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n' %
      (num_correct.item()/len(cifar10_dataset)), len(cifar10_dataset))

# test adversarial examples in MNIST testing dataset
cifar10_dataset_test = torchvision.datasets.CIFAR10(
    '../cifar-10-batches-py', train=False, transform=transform, download=True)
test_dataloader = DataLoader(
    cifar10_dataset_test, batch_size=batch_size, shuffle=False, num_workers=0)
num_correct = 0

for i, data in enumerate(test_dataloader):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    z_ = torch.rand((batch_size, 62))
    z_ = z_.cuda()
    # try:
    #     perturbation = pretrained_G(test_img)
    #     perturbation = torch.clamp(perturbation, -0.3, 0.3)
    #     adv_img = perturbation + test_img
    #     adv_img = torch.clamp(adv_img, 0, 1)
    # except:
    #     break
    try:
        adv_img = pretrained_G(z_)
        pred_lab = torch.argmax(target_model(adv_img), 1)
        num_correct += torch.sum(pred_lab == target_label, 0)
    except:
        break
    

print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in testing set: %f\n' %
      (num_correct.item()/len(cifar10_dataset_test)), len(cifar10_dataset_test))
