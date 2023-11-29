import argparse
from matplotlib import pyplot as plt
import time
from datetime import datetime
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch import nn
from torch.utils.data import DataLoader
from torchsummary import torchsummary

from noses_dataset import NosesDataset
from regression_model import RegressionModel

if __name__ == '__main__':

    image_resize = (128, 128)
    device = torch.device('cpu')

    # ----- set up argument parser for command line inputs ----- #
    parser = argparse.ArgumentParser()
    parser.add_argument('-images_dir', type=str, help='directory for images')
    parser.add_argument('-labels_dir', type=str, help='directory for labels')
    parser.add_argument('-weight_file', type=str, help='file name for saved model')
    parser.add_argument('-b', type=int, default=20, help='Batch size')
    parser.add_argument('-cuda', type=str, default='n', help='[y/N]')

    # ----- get args from the arg parser ----- #
    args = parser.parse_args()
    batch_size = int(args.b)
    use_cuda = str(args.cuda).lower()
    weight_file = Path(args.weight_file)
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    test_set = NosesDataset(images_dir, labels_dir, image_resize, training=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    test_set_length = len(test_set)

    # # ----- initialize model and training parameters ----- #
    model = RegressionModel(image_resize[0])
    print('model loaded OK!')

    print("CudaIsAvailable: {}, UseCuda: {}".format(torch.cuda.is_available(), use_cuda))
    if torch.cuda.is_available() and use_cuda == 'y':
        print('using cuda ...')
        model.cuda()
        device = torch.device('cuda')
    else:
        print('using cpu ...')

    model.to(device=device)

    # ----- begin training the model ----- #
    model.eval()
    torchsummary.summary(model, input_size=(3, 128, 128))
    print("{} training...".format(datetime.now()))

    # ----- Validation ----- #
    with torch.no_grad():
        predicted = []

        for images, labels in test_loader:
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images) * 128

            for x, y in outputs:
                predicted.append((x.item(), y.item()))
            print(predicted)
            quit()

            del images, labels, outputs

    # print("{} Epoch {}, train loss {:.7f}, valid loss {:.7f}".format(
    #     datetime.now(), epoch + 1,
    #                     epoch_loss_train / train_set_length,
    #                     epoch_loss_valid / test_set_length))