from torch.utils.data import Dataset


class NosesDataset(Dataset):
    def __init__(self, images_dir, labels_dir, training=True, transform=None):
        self.transform = transform
        self.training = training
        self.images_dir = images_dir
        self.labels_dir = labels_dir

        # define dataset mode
        self.mode = 'train'
        if self.training == False:
            self.mode = 'test'

        # define other parameters


    def __len__(self):
        return len(self.image_files)