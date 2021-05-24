# Semantic image segmentation
## Model
UNET architecture is implemented with ResNet-34 backbone.
Reason for UNET is because it combines the low level and high level features.
## Dataset
Camvid dataset
## Results

![plot](./Data/Images/download%20(7).png)

Train loss vs validation loss over 200 epochs. 
I used simple BCE loss. Another model with loss function Dice+BCE loss is also
trained, please check colab_bce_dice branch.

![plot](./Data/Images/download%20(1).png)
![plot](./Data/Images/download%20(4).png)

![plot](./Data/Images/download%20(6).png)
![plot](./Data/Images/download%20(5).png)

Original mask (left), generated mask (right)
