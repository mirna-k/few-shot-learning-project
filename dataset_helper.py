import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset

class ClassData:
    def __init__(self, support_dir, query_dir, class_name, label, cro_name, description, transform=None):
        self.class_name = class_name
        self.label = label
        self.cro_name = cro_name
        self.description = description
        self.images = []

        support_images_paths = [os.path.join(support_dir, filename) for filename in os.listdir(support_dir)]
        query_images_paths = [os.path.join(query_dir, filename) for filename in os.listdir(query_dir)]

        for image_path in support_images_paths:
            image = Image.open(image_path)
            if transform:
                image = transform(image)
            self.images.append(image)

        for image_path in query_images_paths:
            image = Image.open(image_path)
            if transform:
                image = transform(image)
            self.images.append(image)

        self.images = torch.stack(self.images)
        self.len = len(self.images)

    def __len__(self):
        return self.len

    @classmethod
    def get_class_data(cls, dataset_dir, class_label_map_json, transform=None):
        sets = {}

        with open(class_label_map_json, 'r') as f:
            class_label_map = json.load(f)

        for class_name, attributes in class_label_map.items():
            label = attributes['label']
            cro_name = attributes['cro_name']
            description = attributes['description']

            class_dir = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_dir):
                support_dir = os.path.join(class_dir, "support")
                query_dir = os.path.join(class_dir, "query")
                sets[label] = cls(support_dir, query_dir, class_name, label, cro_name, description, transform)

        return sets


class CustomDataset(Dataset):
    def __init__(self, class_data):
        self.class_data = class_data
        self.images = []
        self.labels = []

        for label, data in class_data.items():
            self.images.extend(data.images)
            self.labels.extend([label] * len(data))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label