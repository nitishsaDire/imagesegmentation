from torch.utils.data.dataset import Dataset
from torchvision import transforms
from .ImageFile import *
import os
from PIL import Image
import torch
from sklearn import preprocessing

class ImageDataset(Dataset):
    def __init__(self, images_path, labels_path, extensions):
        self.images_path = images_path
        self.labels_path = labels_path
        self.extensions = extensions
        imageFiles, labelFiles = ImageFiles(self.images_path, self.extensions), ImageFiles(self.labels_path, self.extensions)
        self.imageFilesPaths = imageFiles.get_files_paths()
        self.labelFilesPaths = labelFiles.get_files_paths()
        self.n = len(self.imageFilesPaths)
        self.le = preprocessing.LabelEncoder()
        self.transforms = transforms.Compose([transforms.Resize((224, 224)),
                                        transforms.ToTensor()
                                        ])
        self.normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                              ])
        # skyscapes dataset
        self.le.fit([
            "Animal", "Archway", "Bicyclist", "Bridge", "Building", "Car", "CartLuggagePram", "Child", "Column_Pole",
            "Fence", "LaneMkgsDriv", "LaneMkgsNonDriv", "Misc_Text", "MotorcycleScooter", "OtherMoving", "ParkingBlock",
            "Pedestrian", "Road", "RoadShoulder", "Sidewalk", "SignSymbol", "Sky", "SUVPickupTruck", "TrafficCone",
            "TrafficLight", "Train", "Tree", "Truck_Bus", "Tunnel", "VegetationMisc", "Void", "Wall"
        ])


    def get_label_image(self, index):
        label = Image.open(self.imageFilesPaths[index])
        label = self.transform(label) * 255. // 1.
        return label

    def get_image(self, index):
        image = Image.open(self.imageFilesPaths[index])
        image = self.normalize(self.transform(image))
        return image

    def __getitem__(self, index):
        image, label = self.get_image(self.imageFilesPaths[index]), self.get_label_image(self.labelFilesPaths[index])
        return image,label

    def __len__(self):
        return len(self.imageFilesPaths)