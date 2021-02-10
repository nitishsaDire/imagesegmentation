from torch.utils.data import DataLoader
from Data.ImageDataset import *
import torch
from Model import UNET_resnet34
import torch.nn.functional as F
import time
import gc
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import jaccard_similarity_score


# created by Nitish Sandhu
# date 05/feb/2021

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("loadModelFlag", help="flag for loading model",
                        type=int)
    args = parser.parse_args()

    image_path ='/root/.fastai/data/camvid/images/'
    extensions = [".jpg", ".png"]

    imageDataset = ImageDataset(image_path, extensions)

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

    unetModel = UNET_resnet34(32)
    unetModel = unetModel.to(device)
    # for i in unetModel.parameters():
    #     print(i.requires_grad)
    print(len(list(unetModel.parameters())))

    optimizer = torch.optim.Adam(unetModel.parameters(), lr=0.01)

    exp_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    if args.loadModelFlag > 0:
        train_model(unetModel, optimizer, exp_lr_scheduler, dataloader, dataset_sizes, device, loadModel=True, num_epochs=200)
    else :
        train_model(unetModel, optimizer, exp_lr_scheduler, dataloader, dataset_sizes, device)

def denormalize(input):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return np.multiply(std,input) + mean


def plot_stats(num_epochs, stats1, stats2):
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(range(num_epochs), stats1['train'], marker='+', color='r', label='train_loss')
    ax.plot(range(num_epochs), stats1['val'], marker='.', color='b', label='val_loss')
    plt.xlim([0,200])
    plt.legend()
    plt.show()

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(range(num_epochs), stats2['train'], marker='+', color='r', label='train_acc')
    ax.plot(range(num_epochs), stats2['val'], marker='.', color='b', label='val_acc')
    plt.xlim([0,200])
    plt.legend()
    plt.show()

def train_model(unet, optimizer, scheduler, dataloader, dataset_sizes, device, loadModel = False, num_epochs=100):
    since = time.time()
    epoch_losses = {}
    epoch_accuracies = {}
    for k in ['train', 'val']:
        epoch_losses[k] = []
        epoch_accuracies[k] = []

    OLD_PATH = '/content/drive/MyDrive/sem_is_dice-bce'
    PATH = '/content/drive/MyDrive/sem_is_dice-bce'
    epoch = 0
    if loadModel == True:
        checkpoint = torch.load(OLD_PATH)
        unet.load_state_dict(checkpoint['cnn_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        epoch_losses = checkpoint['epoch_losses']
        epoch_accuracies = checkpoint['epoch_accuracies']
        unet = unet.to(device)

    if loadModel == True:
        for g in optimizer.param_groups:
            g['lr'] = 0.001

    # best_model_wts_cnn, best_model_wts_lstm = copy.deepcopy(unet.state_dict()), copy.deepcopy(lstm.state_dict())
    best_acc = 0.0

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
                    unet = unet.to(device)
                    if phase == 'train':
                        unet.train()  # Set model to training mode
                    else:
                        unet.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    count = 0
                    it_begin = time.time()
                    for inputs, mask_1c in dataloader[phase]:
                            if (count+9)%10==0:
                                it_begin = time.time()
                            mask_1c = mask_1c.squeeze(1)
                            inputs, mask_1c = inputs.to(device), mask_1c.to(device)
                            # print("mask1",mask.shape)
                            mask = torch.nn.functional.one_hot(mask_1c.to(torch.int64), 32).permute(0,3,1,2)
                            if count%100 == 0:
                                indexx = 8
                                print(phase)
                                img = inputs[indexx].cpu()
                                fig, ax = plt.subplots(figsize=(10, 10))
                                plt.imshow(denormalize(img.permute(1,2,0)))
                                plt.imshow( mask_1c[indexx].cpu(), alpha = 0.5)
                                plt.show()
                                fig, ax = plt.subplots()
                                plt.imshow(masks_to_colorimg(mask[indexx].cpu()))
                                plt.show()
                            # print("mask2",mask.shape)

                            # zero the parameter gradients
                            optimizer.zero_grad()

                            # forward
                            # track history if only in train
                            with torch.set_grad_enabled(phase == 'train'):
                                    outputs = unet(inputs)
                                    _, preds = torch.max(outputs, 1)

                                    if count % 100 == 0:
                                        print(phase)
                                        # print(outputs[0].max(), outputs[0].min())
                                        fig, ax = plt.subplots(figsize = (10,10))
                                        plt.imshow(denormalize(img.permute(1,2,0)))
                                        plt.imshow(preds[indexx].cpu(), alpha = 0.5)
                                        plt.show()
                                        fig, ax = plt.subplots()
                                        plt.imshow(masks_to_colorimg(outputs[indexx].cpu()))
                                        plt.show()
                                    loss =  0.7 * F.binary_cross_entropy_with_logits(outputs.to(device), mask.to(torch.float))\
                                            + 0.3 * dice_loss(torch.sigmoid(outputs.to(device)), mask.to(torch.float))
                                    # print("tsm",0.66 * F.binary_cross_entropy_with_logits(outputs.to(device), mask.to(torch.float)), 0.33 * dice_loss(outputs.to(device), mask.to(torch.float)))
                                    # backward + optimize only if in training phase
                                    if phase == 'train':
                                        loss.backward()
                                        optimizer.step()

                            # statistics
                            running_loss += loss.item() * inputs.size(0)
                            running_corrects += iou(outputs.to(device), mask.to(torch.float)) * inputs.size(0)

                            if count%10 == 0:
                                time_elapsed = time.time() - it_begin
                                print("IIterated over ", count, "LR=", scheduler.get_last_lr(),'Iteration Completed in {:.0f}m {:.0f}s'.format(
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

            torch.save({
                'epoch': epoch,
                'cnn_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'epoch_losses': epoch_losses,
                'epoch_accuracies': epoch_accuracies
            }, PATH)

            time_elapsed = time.time() - epoch_b
            print('epoch completed in {:.0f}m {:.0f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

            print()
            print(epoch_losses)
            print(epoch_accuracies)
            print('-'*30)
            plot_stats(epoch + 1, epoch_losses, epoch_accuracies)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    return unet


def dice_loss(pred, target, smooth=1e-5):
    intersection = (pred * target).sum(dim=[2, 3])
    union = pred.sum(dim=[2, 3]) + target.sum(dim=[2, 3])

    loss = (1 - ((2.0 * intersection + smooth) / (union + smooth)))

    return loss.mean()

def iou(pred, target):
    # print(pred.shape, target.shape)
    pred = pred.cpu().contiguous().view(-1)
    # print(pred.shape)
    target = target.cpu().contiguous().view(-1)
    return jaccard_similarity_score(pred, target)

color = np.array([list(np.random.choice(range(256), size=3)) for _ in range(32)])

def ground_masks_to_colorimg(masks):
    colors = color.cpu().numpy()
    masks = masks.cpu()
    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape
    count = 0
    for y in range(height):
        for x in range(width):
            indices = int(masks[:,y,x])
            selected_colors = colors[indices]
            colorimg[y,x,:] = selected_colors
    print(colorimg.min(), colorimg.max())

    return colorimg

def masks_to_colorimg(masks):
    colors = color
    masks = masks.cpu()

    colorimg = np.ones((masks.shape[1], masks.shape[2], 3), dtype=np.float32) * 255
    channels, height, width = masks.shape

    for y in range(height):
        for x in range(width):
            selected_colors = colors[masks[:,y,x] > 0.2]

            if len(selected_colors) > 0:
                colorimg[y,x,:] = np.mean(selected_colors, axis=0)

    return colorimg.astype(np.uint8)




if __name__ == '__main__':
    main()
