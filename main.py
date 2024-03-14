import dataset, ConNet, Residual
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import transforms
import torch.utils.data as data
from torch.utils.data import DataLoader
import glob
import os
import copy
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import argparse

data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(ROOT_DIR, "Micro_Organism")
class_names = os.listdir(data_dir)

num_class = len(class_names)
image_files = glob.glob(data_dir + '/*/*.jpg', recursive=True)

idx_to_class = {i: j for i, j in enumerate(class_names)}
class_to_idx = {value: key for key, value in idx_to_class.items()}

train_idx, val_idx, test_idx = data.random_split(image_files, [0.8, 0.1, 0.1],
                                                 generator=torch.Generator().manual_seed(42))

train_list = [image_files[i] for i in train_idx.indices]
val_list = [image_files[i] for i in val_idx.indices]
test_list = [image_files[i] for i in test_idx.indices]
print(test_list)
exit(0)

train_dataset = dataset.Micro_OrganismDataset(train_list, class_to_idx, data_transforms["train"])
val_dataset = dataset.Micro_OrganismDataset(val_list, class_to_idx, data_transforms['val'])
test_dataset = dataset.Micro_OrganismDataset(test_list, class_to_idx, data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

dataloaders = {"train": train_loader, "val": val_loader, "test": test_loader}
sizes = {"train": len(train_dataset), "val": len(val_dataset), "test": len(test_dataset)}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

plots = {"train": [[], []], "val": [[], []]}


def train(model, criterion, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print(10 * '-')

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            loss = 0.0
            totalcorrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                loss += loss.item() * inputs.size(0)
                totalcorrects += torch.sum(preds == labels.data)

            epoch_loss = loss / sizes[phase]
            epoch_acc = totalcorrects.double() / sizes[phase]

            plots[phase][0].append(epoch_loss)
            plots[phase][1].append(epoch_acc)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    print(f'Best val Acc: {best_acc:4f}')

    model.load_state_dict(best_model_wts)
    return model


def plot(plots):
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(plots["train"][0], label="train_loss")
    plt.plot(plots["val"][0], label="val_loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc="lower left")
    plt.savefig("loss.png")

    plt.style.use("ggplot")
    plt.figure()
    plt.plot(plots["train"][1], label="train_acc")
    plt.plot(plots["val"][1], label="val_acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower left")
    plt.savefig("acc.png")


def plot_confusion_matrix(predicted_labels_list, y_test_list, plot_name):
    cnf_matrix = confusion_matrix(y_test_list, predicted_labels_list)
    np.set_printoptions(precision=2)

    plt.figure()
    generate_confusion_matrix(cnf_matrix, classes=set(class_names), title='Confusion matrix ' + plot_name)
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


def predict(model):
    testCorrect = 0
    predictions = torch.tensor([])
    labels = [label[0].numpy() for _, label in test_loader]
    labels = np.array(labels)

    with torch.no_grad():
        model.eval()
        for (x, y) in test_loader:
            (x, y) = (x.to(device), y.to(device))
            pred = model(x)
            predictions = torch.cat((predictions, pred), dim=0)

            testCorrect += (pred.argmax(1) == y).type(
                torch.float).sum().item()

    print("Test Acc: ", testCorrect / sizes["test"])
    predictions = predictions.argmax(dim=1).numpy()
    plot_confusion_matrix(predictions, labels, "ResNet18 FC and Layer 4 Freeze")


def freeze(model, all=True):
    i = 0
    for param in model.parameters():
        if i >= 45 and all == False:
            break
        param.requires_grad = False
        i += 1


ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, required=True,
                help="connet, residualnet, resnet18")
args = vars(ap.parse_args())

model = None
optimizer = None

if (args["model"] == "connet"):
    model = ConNet.ConNet(3)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

elif (args["model"] == "residualnet"):
    model = Residual.ResidualNet(3)
    model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

elif (args["model"] == "resnet18"):
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    freeze(model)

    filters = model.fc.in_features
    model.fc = nn.Linear(filters, 8)
    model = model.to(device)
    optimizer = optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

else:
    print("There is no such model")
    exit(0)



criterion = nn.CrossEntropyLoss()

model_conv = train(model, criterion, optimizer, num_epochs=25)
plot(plots)
predict(model_conv)
