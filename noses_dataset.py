import os
import cv2
import torch
from torch.utils.data import Dataset


class NosesDataset(Dataset):
    def __init__(self, images_dir, labels_dir, img_resize, training=True, transform=None):
        self.transform = transform
        self.img_resize = img_resize
        self.training = training
        self.mode = 'train'
        if self.training == False:
            self.mode = 'test'
        self.labels_dir = labels_dir
        self.images_dir = images_dir
        self.labels = []
        self.images = []

        # save file and label references
        labels_file = os.path.join(labels_dir, self.mode+'_noses.txt')
        with open(labels_file) as label_file:
            labels_string = label_file.readlines()
        for i in range(len(labels_string)):
            line = labels_string[i].replace("\"(", "").replace(")\"", "").replace(" ", "")
            split = line.split(',')
            self.labels.append((float(split[1]), float(split[2])))
            self.images.append(os.path.join(images_dir, split[0]))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        label = self.labels[idx]
        image = cv2.imread(self.images[idx], cv2.IMREAD_COLOR)
        if image is None:
            print(self.images[idx])
            quit()
        image, label = self.resize_data(image, label)
        label = torch.tensor(label)
        image = torch.from_numpy(image).float()
        image = image.permute(2, 0, 1)
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def resize_data(self, image, point):
        scale_percent_x = float(self.img_resize[0] / image.shape[1])
        scale_percent_y = float(self.img_resize[0] / image.shape[0])

        # convert labels to match resized image
        new_point_x = float(point[0] * scale_percent_x)
        new_point_y = float(point[1] * scale_percent_y)

        # normalize point coordinates between 0 and 1
        new_point_x = float(new_point_x / self.img_resize[0])
        new_point_y = float(new_point_y / self.img_resize[0])

        # create and return new objects
        new_point = (new_point_x, new_point_y)
        image_new = cv2.resize(image, self.img_resize)

        # cv2.circle(image, point, 5, (0, 0, 255), -1)
        # cv2.imshow('original image', image)
        # cv2.waitKey(0)
        #
        # cv2.circle(image_new, (new_point_x, new_point_y), 5, (0, 0, 255), -1)
        # cv2.imshow('new image', image_new)
        # cv2.waitKey(0)

        return image_new, new_point
