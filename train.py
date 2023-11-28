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


def data_transform():
    transform_list = [
        transforms.RandomAutocontrast(0.3),
        # transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ]
    return transforms.Compose(transform_list)


if __name__ == '__main__':

    image_resize = (128, 128)
    device = torch.device('cpu')

    # ----- set up argument parser for command line inputs ----- #
    parser = argparse.ArgumentParser()
    parser.add_argument('-images_dir', type=str, help='directory for images')
    parser.add_argument('-labels_dir', type=str, help='directory for labels')
    parser.add_argument('-weight_file', type=str, help='file name for saved model')
    parser.add_argument('-plot_file', type=str, help='file name for saved plot')
    parser.add_argument('-learn', type=float, default=0.01, help='Learning rate')
    parser.add_argument('-e', type=int, default=20, help='Number of epochs')
    parser.add_argument('-b', type=int, default=20, help='Batch size')
    parser.add_argument('-cuda', type=str, default='n', help='[y/N]')

    # ----- get args from the arg parser ----- #
    args = parser.parse_args()
    learn = float(args.learn)
    n_epochs = int(args.e)
    batch_size = int(args.b)
    use_cuda = str(args.cuda).lower()
    weight_file = Path(args.weight_file)
    plot_file = Path(args.plot_file)
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    train_set = NosesDataset(images_dir, labels_dir, image_resize, training=True, transform=data_transform())
    valid_set = NosesDataset(images_dir, labels_dir, image_resize, training=False, transform=data_transform())
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    train_set_length = len(train_set)
    valid_set_length = len(valid_set)

    # # ----- initialize model and training parameters ----- #
    model = RegressionModel()
    print('model loaded OK!')

    print("CudaIsAvailable: {}, UseCuda: {}".format(torch.cuda.is_available(), use_cuda))
    if torch.cuda.is_available() and use_cuda == 'y':
        print('using cuda ...')
        model.cuda()
        device = torch.device('cuda')
    else:
        print('using cpu ...')

    model.to(device=device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learn, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=5, verbose=True)

    # ----- begin training the model ----- #
    model.train()
    loss_train = []
    loss_valid = []
    torchsummary.summary(model, input_size=(3, 128, 128))
    print("{} training...".format(datetime.now()))
    start_time = time.time()

    for epoch in range(n_epochs):
        epoch_loss_train = 0.0
        epoch_loss_valid = 0.0

        for images, labels in train_loader:
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss_train += loss.item()

        scheduler.step(epoch_loss_train)
        loss_train.append(epoch_loss_train / train_set_length)

        # ----- Validation ----- #
        with torch.no_grad():
            for images, labels in valid_loader:
                images = images.to(device=device)
                labels = labels.to(device=device)
                outputs = model(images)
                loss = loss_fn(outputs, labels)
                epoch_loss_valid += loss.item()
                del images, labels, outputs

        loss_valid.append(epoch_loss_valid / valid_set_length)

        print("{} Epoch {}, train loss {:.7f}, valid loss {:.7f}".format(
            datetime.now(), epoch + 1,
            epoch_loss_train / train_set_length,
            epoch_loss_valid / valid_set_length))

    end_time = time.time()

    # # ----- print final training statistics ----- #
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    total_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    print("Total training time: {}, ".format(total_time))

    # save the model weights
    torch.save(model.state_dict(), weight_file)

    # save loss plot and accuracy plot
    plt.figure(figsize=(12, 7))
    plt.clf()
    plt.plot(loss_train, label='training loss')
    plt.plot(loss_valid, label='validation loss')
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(loc=1)
    plt.savefig(plot_file)
    plt.show()
