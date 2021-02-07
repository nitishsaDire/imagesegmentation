from torch.utils.data.dataset import Dataset
from torchvision import transforms
from .ImageFile import *
from PIL import Image
from sklearn import preprocessing

# created by Nitish Sandhu
# date 05/feb/2021

class ImageDataset(Dataset):
    def __init__(self, images_path, extensions):
        # self.path = datasets.untar_data(datasets.URLs.CAMVID)
        self.images_path = images_path
        self.extensions = extensions
        imageFiles = ImageFiles(self.images_path, self.extensions)
        self.imageFilesPaths = imageFiles.get_files_paths()
        self.n = len(self.imageFilesPaths)
        self.le = preprocessing.LabelEncoder()
        self.transforms = transforms.Compose([transforms.Resize((224,224)),
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
        # '/home/nitish/.fastai/data/camvid/images/Seq05VD_f04500.png'
        # '/home/nitish/.fastai/data/camvid/labels/Seq05VD_f04500_P.png'
        label_file = self.imageFilesPaths[index].replace('/images/', '/labels/')
        label = Image.open(label_file[:-4] + '_P' + label_file[-4:])
        label = self.transforms(label).squeeze() * 255. // 1.
        return label

    def get_image(self, index):
        image = Image.open(self.imageFilesPaths[index])
        image = self.normalize(self.transforms(image))
        return image

    def __getitem__(self, index):
        # print(index)
        image, label = self.get_image(index), self.get_label_image(index)
        return image,label

    def __len__(self):
        return len(self.imageFilesPaths)