import os
import csv
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image

from utils.iou import iou_width_height as iou

class Dataset(data.Dataset):
    def __init__(self, opt, phase, transform=None):
        self.data_dir = opt.data_dir
        self.data_name = opt.data_name
        self.img_size = opt.input_size
        self.transform = transform

        self.ignore_iou_thresh = 0.5
        self.S = opt.S
        self.Anchors = opt.Anchors
        self.C = opt.num_classes

        self.img_names = list()
        self.anno_names = list()
        with open(os.path.join(self.data_dir, self.data_name, '{}.csv'.format(phase))) as f:
            reader = csv.reader(f)
            for line in reader:
                self.img_names.append(phase + '/' + line[0])
                self.anno_names.append(phase + '/' + line[1])

        self.label2num = {}
        with open(os.path.join(self.data_dir, self.data_name, 'label.txt'), 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                self.label2num[line.strip()] = i

    
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.data_dir, self.data_name, self.img_names[index]))
        img = np.array(img.convert("RGB"))
        
        anno_path = os.path.join(self.data_dir, self.data_name, self.anno_names[index])
        bboxes = np.roll(np.loadtxt(fname=anno_path, delimiter=" ", ndmin=2), 4, axis=1).tolist()

        if self.transform is not None:
            augmentations = self.transform(image=img, bboxes=bboxes)
            img = augmentations["image"]
            bboxes = augmentations["bboxes"]

        # 3(num_anchors_per_cell), s,s (per_cell), 6(objectness, x, y, w, h, class)
        anchors = torch.tensor(self.Anchors[0] + self.Anchors[1] + self.Anchors[2])
        targets = [torch.zeros((3, s, s, 6)) for s in self.S]
        for box in bboxes:
            iou_anchors = iou(torch.tensor(box[2:4]), anchors)
            anchor_indices = iou_anchors.argsort(descending=True, dim=0)
            x, y, width, height, class_label = box
            has_anchor = [False] * 3

            for anchor_idx in anchor_indices:
                scale_idx = torch.div(anchor_idx, 3, rounding_mode='floor')
                anchor_on_scale = anchor_idx % 3
                S = self.S[scale_idx]
                i, j = int(S * y), int(S * x) # from (0~1) to integer cell location
                anchor_taken = targets[scale_idx][anchor_on_scale, i, j, 0]

                if not anchor_taken and not has_anchor[scale_idx]:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = 1
                    x_cell, y_cell = S * x - j, S * y - i
                    width_cell, height_cell = (width * S, height * S)
                    box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                    targets[scale_idx][anchor_on_scale, i, j, 1:5] = box_coordinates
                    targets[scale_idx][anchor_on_scale, i, j, 5] = int(class_label)
                    has_anchor[scale_idx] = True
                
                elif not anchor_taken and iou_anchors[anchor_idx] > self.ignore_iou_thresh:
                    targets[scale_idx][anchor_on_scale, i, j, 0] = -1

        return img, tuple(targets)

    def __len__(self):
        return len(self.img_names)