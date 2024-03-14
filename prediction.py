import numpy as np
import argparse
import os
import glob
import torch
import cv2
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torch.nn import NLLLoss
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import random

ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
	help="path to the trained PyTorch model")
args = vars(ap.parse_args())

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(ROOT_DIR, "Micro_Organism")
class_names = os.listdir(data_dir)

num_class = len(class_names)
image_files = glob.glob(data_dir+'/*/*.jpg',recursive=True)
idx_to_class = {i:j for i, j in enumerate(class_names)}
class_to_idx = {value:key for key,value in idx_to_class.items()}

train_idx,test_idx,val_idx= data.random_split(image_files, [0.8, 0.1, 0.1], generator=torch.Generator().manual_seed(42))

test_list=[image_files[i] for i in test_idx.indices]

class Micro_OrganismDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_filepath = self.image_paths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label = image_filepath.split('\\')[-2]
        label = class_to_idx[label]
        if self.transform is not None:
            image = self.transform(image)

        return image, label

val_transforms=transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((100,100)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.5])
])

test_dataset = Micro_OrganismDataset(test_list, val_transforms)


INIT_LR = 1e-3
BATCH_SIZE = 1
EPOCHS = 25

test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(args["model"]).to(device)
testCorrect = 0
lossFn = NLLLoss()

labels = [label[0].numpy() for _, label in test_loader]
labels = np.array(labels)
predictions = torch.tensor([])

with torch.no_grad():
    # set the model in evaluation mode
    model.eval()
    # loop over the validation set
    for (x, y) in test_loader:
        # send the input to the device
        (x, y) = (x.to(device), y.to(device))
        # make the predictions and calculate the validation loss
        pred = model(x)

        predictions = torch.cat((predictions, pred), dim=0)

        # calculate the number of correct predictions
        testCorrect += (pred.argmax(1) == y).type(
            torch.float).sum().item()

predictions = predictions.argmax(dim=1).numpy()
label_names = ["Amoeba", "Euglena", "Hydra", "Paramecium", "Rod B.", "Spherical B.", "Spiral B.", "Yeast"]

def plot_confusion_matrix(predicted_labels_list, y_test_list, plot_name):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=set(label_names), title='Confusion matrix ' + plot_name)
    plt.show()


def generate_confusion_matrix(cnf_matrix, classes, title='Confusion matrix'):
    plt.imshow(cnf_matrix, interpolation='nearest', cmap=plt.get_cmap('Greens'))
    plt.title(title)
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=30)
    plt.yticks(tick_marks, classes)

    thresh = cnf_matrix.max() / 2.

    for i, j in itertools.product(range(cnf_matrix.shape[0]), range(cnf_matrix.shape[1])):
        plt.text(j, i, format(cnf_matrix[i, j], 'd'), horizontalalignment="center",
                 color="white" if cnf_matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    return cnf_matrix


print("Test Accuracy: ", testCorrect)
plot_confusion_matrix(predictions, labels, "With Residual Con.")


