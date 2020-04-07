import argparse

parser = argparse.ArgumentParser(description='Train a neuro-network image classifier.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data_directory', type=str, 
                    default='.',
                    help='The data directory, must contains "train" and "valid" folder')

parser.add_argument('--save_dir', type=str, 
                    default='.',
                    help='The directory for saving checkpoint')

parser.add_argument('-a', '--arch', type=str, 
                    default='vgg11',
                    help='The pre-trained network name, e.g. vgg11. See: https://pytorch.org/docs/master/torchvision/models.html')

parser.add_argument('-hu', '--hidden_units', type=int, nargs='*', 
                    default=[4096, 800],
                    help='Hidden layer units number')

parser.add_argument('-lr', '--learning_rate', type=float, 
                    default=0.001,
                    help='The learning rate')

parser.add_argument('-bs', '--batch_size', type=int, 
                    default=64,
                    help='The SGD batch size')

parser.add_argument('-e', '--epochs', type=int, 
                    default=1,
                    help='The learning epochs')

parser.add_argument('--gpu', help='If training with gpu', action='store_true')
parser.add_argument('--trial', help='Just train with one batch, for debuging', action='store_true')
parser.add_argument('--new_train', help='Do not load previous training checkpoint', action='store_true')

args = parser.parse_args()


data_directory = args.data_directory
save_dir = args.save_dir
pre_trained_arch = args.arch
hidden_units = args.hidden_units
batch_size = args.batch_size
learning_rate = args.learning_rate
epochs = args.epochs
with_gpu = args.gpu
trial = args.trial
new_train = args.new_train
checkpoint_file_name = save_dir + '/checkpoint-' + pre_trained_arch + '-hiddens-' + ('+'.join(map(str, hidden_units))) + '.pth'

# Import

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict


print("--- Defining training data transforms...", flush=True)

data_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(30),
    transforms.RandomResizedCrop((224, 224), scale=(0.4, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print(data_transforms, flush=True)

print("--- Defining validation data transforms...", flush=True)

test_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(1.0, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

print(test_data_transforms, flush=True)

print("--- Creating train/valid data loader...", flush=True)

train_dir = data_directory + '/train'
valid_dir = data_directory + '/valid'

image_datasets = [
    datasets.ImageFolder(train_dir, transform=data_transforms),
    datasets.ImageFolder(valid_dir, transform=test_data_transforms)
]

dataloaders = [
    DataLoader(image_datasets[0], batch_size=batch_size, shuffle=True),
    DataLoader(image_datasets[1], batch_size=batch_size)
]

print(f"--- {len(image_datasets[0].imgs)} training images and {len(image_datasets[1].imgs)} validation images loaded!", flush=True)

if image_datasets[0].class_to_idx != image_datasets[1].class_to_idx:
    raise ValueError("Training and validation labels are different!")
    
class_to_idx = image_datasets[0].class_to_idx
classes = image_datasets[0].classes
print("--- Classes:", flush=True)
print(classes, flush=True)

print("--- Building model, classifier will ajusted to corresponding hidden layer input and final output, only weights on classifier will be trained", flush=True)

model = getattr(models, pre_trained_arch)(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

last_layer_units = model.classifier[0].in_features

layers = []

for idx, hu in enumerate(hidden_units):
    layers.append((f'fc{idx}', nn.Linear(last_layer_units, hu)))
    layers.append((f'relu{idx}', nn.ReLU()))
    layers.append((f'drop{idx}', nn.Dropout(p=0.5)))
    last_layer_units = hu
else:
    layers.append((f'fc{len(hidden_units)}', nn.Linear(last_layer_units, len(class_to_idx))))
    layers.append((f'output', nn.LogSoftmax(dim=1)))

model.classifier = nn.Sequential(OrderedDict(layers))
model.class_to_idx = class_to_idx
model.classes = classes

print(model, flush=True)

if not new_train:
    print("--- Checking previous checkpoint.", flush=True)
    try:
        checkpoint = torch.load(checkpoint_file_name)
        model.classifier.load_state_dict(checkpoint['state_dict'])
        print("--- Checkpoint loaded!", flush=True)
    except:
        print("--- No previous checkpoint found.", flush=True)        

print("--- Start to train the model.", flush=True)

if with_gpu:
    print("--- Move model to CUDA", flush=True)
    model.cuda()

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
steps = 0
running_loss = 0
print_every = 10

def validation(model, testloader, criterion):
    validation_loss = 0
    accuracy = 0
    size = len(dataloaders[1])
    for images, labels in dataloaders[1]:
        
        if with_gpu:
            images = images.cuda()
            labels = labels.cuda()

        output = model.forward(images)
        validation_loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return validation_loss/size, accuracy/size

model.train()
for e in range(epochs):
    for images, labels in dataloaders[0]:
        steps += 1
        
        if with_gpu:
            images = images.cuda()
            labels = labels.cuda()
        
        optimizer.zero_grad()
        
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if steps % print_every == 0 or trial:
            
            model.eval()
            
            with torch.no_grad():
                validation_loss, accuracy = validation(model, dataloaders[1], criterion)

            print("Epoch: {}/{}.. {:.1f}%.. ".format(e+1, epochs, (steps+1) * 64 * 100 / len(image_datasets[0].imgs)),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Validation Loss: {:.3f}.. ".format(validation_loss),
                  "Validation Accuracy: {:.3f}".format(accuracy), flush=True)
            
            running_loss = 0
            
            model.train()
                    
        if trial:
            break
            
    # save after each epoch
    steps = 0
    print("--- Saving model...", flush=True)
    checkpoint = {
        'pre_trained_arch': pre_trained_arch,
        'hidden_units': hidden_units,
        'class_to_idx': model.class_to_idx,
        'classes': model.classes,
        'state_dict': model.classifier.state_dict()
    }
    torch.save(checkpoint, checkpoint_file_name)

print("--- Training finished!", flush=True)




