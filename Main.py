from torch.utils.data import DataLoader
from Data.ImageDataset import *
import torch
from Model import UNET_resnet34
import torch.nn.functional as F
import time
import gc
import matplotlib.pyplot as plt

# created by Nitish Sandhu
# date 05/feb/2021

def main():
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument("loadModelFlag", help="flag for loading model",
    #                     type=int)
    # args = parser.parse_args()

    image_path ='/root/.fastai/data/camvid/images/'
    extensions = [".jpg", ".png"]
    label_path = '/root/.fastai/data/camvid/labels/'
    imageDataset = ImageDataset(image_path, label_path, extensions)

    train_size = int(0.8 * len(imageDataset))
    val_size = len(imageDataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(imageDataset, [train_size, val_size])

    imageDatasetDict = {"train": train_dataset, "val":val_dataset}

    dataset_sizes = {x: len(imageDatasetDict[x]) for x in ['train', 'val']}

    dataloader = {x: DataLoader(imageDatasetDict[x], batch_size=20,
                            shuffle=True, num_workers=4, pin_memory=True) for x in ['train', 'val']}
    use_cuda = torch.cuda.is_available()  # check if GPU exists
    device = torch.device("cuda" if use_cuda else "cpu")  # use CPU or GPU
    print(device)

    unetModel = UNET_resnet34(30).to(device)

    optimizer = torch.optim.Adam(list(unetModel.parameters()), lr=0.001)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    if args.loadModelFlag > 0:
        train_model(unetModel, optimizer, exp_lr_scheduler, dataloader, dataset_sizes, device, loadModel=True, num_epochs=50)
    else :
        train_model(unetModel, optimizer, exp_lr_scheduler, dataloader, dataset_sizes, device)

def plot_stats(num_epochs, stats1, stats2):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(range(num_epochs), stats1['train'], marker='+', color='r', label='train_loss')
    ax.plot(range(num_epochs), stats1['val'], marker='.', color='b', label='val_loss')
    plt.xlim([0,30])
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(range(num_epochs), stats2['train'], marker='+', color='r', label='train_acc')
    ax.plot(range(num_epochs), stats2['val'], marker='.', color='b', label='val_acc')
    plt.xlim([0,30])
    plt.legend()
    plt.show()

def train_model(unet, optimizer, scheduler, dataloader, dataset_sizes, device, loadModel = False, num_epochs=25):
    since = time.time()

    OLD_PATH = ''
    PATH = ''
    epoch = 0
    if loadModel == True:
        checkpoint = torch.load(OLD_PATH)
        unet.load_state_dict(checkpoint['cnn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        unet = unet.to(device)

    # best_model_wts_cnn, best_model_wts_lstm = copy.deepcopy(unet.state_dict()), copy.deepcopy(lstm.state_dict())
    best_acc = 0.0
    epoch_losses = {}
    epoch_accuracies = {}
    for k in ['train', 'val']:
        epoch_losses[k] = []
        epoch_accuracies[k] = []

    for epoch in range(epoch, num_epochs):
            epoch_b = time.time()

            print(device)
            # print(torch.cuda.memory_summary(device=device, abbreviated=False)
            torch.cuda.empty_cache()
            gc.collect()

            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                    if phase == 'train':
                        unet.train()  # Set model to training mode
                    else:
                        unet.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    count = 0
                    it_begin = time.time()
                    for inputs, labels in dataloader[phase]:
                            if (count+9)%10==0:
                                it_begin = time.time()
                            labels = labels.squeeze(1)
                            inputs, labels = inputs.to(device), labels.to(device)

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward
                            # track history if only in train
                            with torch.set_grad_enabled(phase == 'train'):
                                outputs = unet(inputs)

                                _, preds = torch.max(outputs, 1)
                                loss = F.cross_entropy(outputs, labels)
                                # backward + optimize only if in training phase
                                if phase == 'train':
                                    loss.backward()
                                    optimizer.step()

                            # statistics
                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += torch.sum(preds == labels.data)
                            if count%10 == 0:
                                time_elapsed = time.time() - it_begin
                                print("Iterated over ", count, "LR=", scheduler.get_last_lr(),'Iteration Completed in {:.0f}m {:.0f}s'.format(
                                    time_elapsed // 60, time_elapsed % 60))
                            count+=1

                    if phase == 'train':
                        scheduler.step()

                    epoch_loss = running_loss / dataset_sizes[phase]
                    epoch_acc = running_corrects.double() / dataset_sizes[phase]

                    epoch_losses[phase].append(epoch_loss)
                    epoch_accuracies[phase].append(epoch_acc.item())

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

                    # deep copy the model
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        # best_model_wts_cnn, best_model_wts_lstm = copy.deepcopy(unet.state_dict()), copy.deepcopy(lstm.state_dict())

            torch.save({
                'epoch': epoch,
                'cnn_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, PATH)

            time_elapsed = time.time() - epoch_b
            print('epoch completed in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

            print()
            print(epoch_losses)
            print(epoch_accuracies)
            print('-'*30)
            # plot_stats(epoch + 1, epoch_losses, epoch_accuracies)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return unet

if __name__ == '__main__':
    main()