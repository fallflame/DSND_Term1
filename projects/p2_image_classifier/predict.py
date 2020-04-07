import argparse

parser = argparse.ArgumentParser(description='Classify an image with neuro-network classifier.',
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('image_path', type=str, 
                    help='The path to the image')

parser.add_argument('checkpoint_path', type=str, 
                    help='The path to checkpoint file')


parser.add_argument('--topk', type=int, 
                    default=5,
                    help='Print top probabilities')

parser.add_argument('--category_names', type=str, 
                    help='The category name file, will print the category name instead of label if provided')

parser.add_argument('--gpu', help='If training with gpu', action='store_true')

args = parser.parse_args()

image_path = args.image_path
checkpoint_path = args.checkpoint_path
topk = args.topk
category_names = args.category_names
with_gpu = args.gpu

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

print("--- Loading checkpoint...", flush=True)
checkpoint = torch.load(checkpoint_path)
pre_trained_arch = checkpoint['pre_trained_arch']
hidden_units = checkpoint['hidden_units']
class_to_idx = checkpoint['class_to_idx']
classes = checkpoint['classes']

print("--- Building model...", flush=True)

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

model.classifier.load_state_dict(checkpoint['state_dict'])

print("--- Model built!")

test_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop((224, 224), scale=(1.0, 1.0), ratio=(1.0, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    
def process_image(image):
    image_input = image
    for transform in test_data_transforms.transforms:
        image_input = transform(image_input)
    return image_input
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image = Image.open(image_path)
    image_input = process_image(image)
    batch = image_input.unsqueeze(0)
    
    if with_gpu:
        batch = batch.cuda()
    
    output = model.forward(batch) # unsqueeze make it a batch
    
    ps = torch.exp(output)
    
    top_ps, top_class = ps.topk(topk, dim=1)
    
    top_ps = top_ps.cpu().detach()
    top_class = top_class.cpu().detach()
    
    return top_ps[0], [classes[x] for x in top_class[0]]

if with_gpu:
    model.cuda()

model.eval()
with torch.no_grad():
    top_ps, top_class = predict(image_path, model, topk=topk)

if category_names is not None:
    import json
    with open(category_names, 'r') as f:
        cat_to_name = json.load(f)
        for ps, cls in zip(top_ps, top_class):
            print(f'Class: {cat_to_name[cls]}, Probability: {(ps * 100):.1f}')
else:
    for ps, cls in zip(top_ps, top_class):
        print(f'Class: {cls}, \tProbability: {(ps * 100):.1f}')