import argparse
import math
import random
import time
from datetime import datetime
from pathlib import Path
import cv2
import torch
from torch.utils.data import DataLoader
from torchsummary import torchsummary
from noses_dataset import NosesDataset
from resnet_pet_noses import ResNet_Pet_Noses


def calc_euclidean_distance(labels, predictions, image_size):
    if len(labels) != len(predictions):
        raise ValueError("Arrays must be same length")
    labels = labels * image_size
    predictions = predictions * image_size
    predictions = torch.round(predictions)
    distances = []
    for pt1, pt2 in zip(labels, predictions):
        distance = math.sqrt(sum((x-y)**2 for x, y in zip(pt1, pt2)))
        distances.append(distance)
    return distances


if __name__ == '__main__':

    image_resize = 256
    device = torch.device('cpu')

    # ----- set up argument parser for command line inputs ----- #
    parser = argparse.ArgumentParser()
    parser.add_argument('-images_dir', type=str, help='directory for images')
    parser.add_argument('-labels_dir', type=str, help='directory for labels')
    parser.add_argument('-weight_file', type=str, help='file name for saved model')
    parser.add_argument('-b', type=int, default=20, help='Batch size')
    parser.add_argument('-display', type=str, default='n', help='Display images [y/N]')
    parser.add_argument('-cuda', type=str, default='n', help='[y/N]')

    # ----- get args from the arg parser ----- #
    args = parser.parse_args()
    batch_size = int(args.b)
    display = str(args.display).lower() == 'y'
    use_cuda = str(args.cuda).lower() == 'y'
    weight_file = Path(args.weight_file)
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)

    test_set = NosesDataset(images_dir, labels_dir, image_resize, training=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    test_set_length = len(test_set)

    # # ----- initialize model and training parameters ----- #
    model = ResNet_Pet_Noses()
    model.load_state_dict(torch.load(weight_file, map_location=device))
    print('model loaded OK!')

    print("CudaIsAvailable: {}, UseCuda: {}".format(torch.cuda.is_available(), use_cuda))
    if torch.cuda.is_available() and use_cuda:
        print('using cuda ...')
        model.cuda()
        device = torch.device('cuda')
    else:
        print('using cpu ...')

    model.to(device=device)

    # ----- begin training the model ----- #
    model.eval()
    torchsummary.summary(model, input_size=(3, image_resize, image_resize))
    print("{} testing...".format(datetime.now()))
    start_time = time.time()

    # ----- Testing ----- #
    with torch.no_grad():
        distances = []
        gt_and_pred_noses = [((0, 0), (0, 0))] * test_set_length

        for indexes, images, labels in test_loader:
            images = images.to(device=device)
            labels = labels.to(device=device)
            outputs = model(images)
            distances += calc_euclidean_distance(labels, outputs, image_resize)

            for i in range(len(images)):
                gt = (round(image_resize * labels[i][0].item()), round(image_resize * labels[i][1].item()))
                pred = (round(image_resize * outputs[i][0].item()), round(image_resize * outputs[i][1].item()))
                idx = int(indexes[i].item())
                gt_and_pred_noses[idx] = (gt, pred)

            del images, labels, outputs

    end_time = time.time()

    # print timing statistics
    hours, rem = divmod(end_time - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    total_time = "{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds)
    msec_per_img = 1000 * ((end_time - start_time) / len(test_set))
    print("{} testing complete.".format(datetime.now()))
    print("total test time {}, {:.3f} msec per image".format(total_time, msec_per_img))

    # calculate accuracy statistics
    mean_dist = torch.mean(torch.tensor(distances))
    max_dist = torch.max(torch.tensor(distances))
    min_dist = torch.min(torch.tensor(distances))
    stddev_dist = torch.std(torch.tensor(distances))
    print("Mean Accuracy: {:.2f}%".format(100 * (1 - (mean_dist / image_resize))))
    print("Mean Euclidean Distance: {:.4f}".format(mean_dist))
    print("Maximum Euclidean Distance: {:.4f}".format(max_dist))
    print("Minimum Euclidean Distance: {:.4f}".format(min_dist))
    print("Std Deviation of Euclidean Distance: {:.4f}".format(stddev_dist))

    if display:

        # display images for max and min distance
        for x in range(len(distances)):
            if distances[x] == max_dist:
                gt = gt_and_pred_noses[x][0]
                pred = gt_and_pred_noses[x][1]
                print("Max Eucl Dist: {:.4f}, GT: {}, Pred: {}".format(max_dist, gt, pred))
                img = test_set.get_img_with_noses(x, gt, pred)
                cv2.imshow("Ground Truth: green, Predicted: pink", img)
                cv2.waitKey(0)
            if distances[x] == min_dist:
                gt = gt_and_pred_noses[x][0]
                pred = gt_and_pred_noses[x][1]
                print("Min Eucl Dist: {:.4f}, GT: {}, Pred: {}".format(min_dist, gt, pred))
                img = test_set.get_img_with_noses(x, gt, pred)
                cv2.imshow("Ground Truth: green, Predicted: pink", img)
                cv2.waitKey(0)

        # display 10 randomly selected sample images
        for k in range(10):
            idx = random.randint(0, test_set_length-1)
            gt = gt_and_pred_noses[idx][0]
            pred = gt_and_pred_noses[idx][1]
            print("Random Sample #{}: Mean Eucl Dist: {:.4f}, GT: {}, Pred: {}".format(k+1, distances[idx], gt, pred))
            image_with_noses = test_set.get_img_with_noses(idx, gt, pred)
            cv2.imshow("Ground Truth: green, Predicted: pink", image_with_noses)
            cv2.waitKey(0)
