from torch.utils.data import Dataset
import cv2

class Micro_OrganismDataset(Dataset):
    def __init__(self, image_paths, class_to_idx, transform=None):
        self.image_paths = image_paths
        self.class_to_idx = class_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        label = image_filepath.split('\\')[-2]
        label = self.class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image)

        return image, label