from __future__ import division
import os
from config import *
from utilities import preprocess_images, preprocess_maps, preprocess_fixmaps, postprocess_predictions
import imageio
import random
from models_pt import acl_net_pt, ACLLoss
import torch
import torch.optim as optim
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


# a generator to get obtain training batch
def generator(video_b_s, phase_gen='train'):
    if phase_gen == 'train':
        videos = [videos_train_path + f for videos_train_path in videos_train_paths for f in
                  os.listdir(videos_train_path) if os.path.isdir(videos_train_path + f)]
        random.shuffle(videos)
        video_counter = 0
        while True:
            Xims = torch.zeros((video_b_s * num_frames, shape_r, shape_c, 3))

            Ymaps = torch.zeros((video_b_s * num_frames, shape_r_out, shape_c_out, 1)) + 0.01
            Yfixs = torch.zeros((video_b_s * num_frames, shape_r_out, shape_c_out, 1)) + 0.01

            for i in range(0, video_b_s):
                video_path = videos[(video_counter + i) % len(videos)]
                images = [video_path + frames_path + f for f in os.listdir(video_path + frames_path) if
                          f.endswith(('.jpg', '.jpeg', '.png'))]
                maps = [video_path + maps_path + f for f in os.listdir(video_path + maps_path) if
                        f.endswith(('.jpg', '.jpeg', '.png'))]
                fixs = [video_path + fixs_path + f for f in os.listdir(video_path + fixs_path) if
                        f.endswith('.mat')]

                start = random.randint(0, max(len(images) - num_frames, 0))
                X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
                Y = preprocess_maps(maps[start:min(start + num_frames, len(images))], shape_r_out, shape_c_out)
                Y_fix = preprocess_fixmaps(fixs[start:min(start + num_frames, len(images))], shape_r_out,
                                           shape_c_out)

                start_batch = i * num_frames
                Xims[start_batch:start_batch + X.shape[0], :] = torch.from_numpy(X.copy())
                Ymaps[start_batch: start_batch + Y.shape[0], :] = torch.from_numpy(Y.copy())
                Yfixs[start_batch:start_batch + Y_fix.shape[0], :] = torch.from_numpy(Y_fix.copy())

            Xims = Xims.permute(0, 3, 1, 2)
            Ymaps = Ymaps.permute(0, 3, 1, 2)
            Yfixs = Yfixs.permute(0, 3, 1, 2)
            yield Xims, Ymaps, Yfixs  #
            video_counter = (video_counter + video_b_s) % len(videos)
    else:
        raise NotImplementedError


# get test path
def get_test(video_test_path):
    images = [video_test_path + frames_path + f for f in os.listdir(video_test_path + frames_path) if
              f.endswith(('.jpg', '.jpeg', '.png'))]
    images.sort()
    start = 0
    while True:
        Xims = torch.zeros((num_frames, shape_r, shape_c, 3))
        X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
        Xims[0:min(len(images)-start, num_frames), :] = torch.from_numpy(X.copy())
        Xims = Xims.permute(0, 3, 1, 2)
        yield Xims  #
        start = min(start + num_frames, len(images))


# predict all test data
def pred_data_all(net):
    videos = [videos_test_path + f for f in os.listdir(videos_test_path) if os.path.isdir(videos_test_path + f)]
    videos.sort()
    nb_videos_test = len(videos)

    for i in range(nb_videos_test):
        output_folder = output_path + 'test/' + videos[i][-4: ] + '/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        images_names = [f for f in os.listdir(videos[i] + frames_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        images_names.sort()

        print("Predicting saliency maps for " + videos[i])
        start = 0
        images = [videos[i] + frames_path + f for f in os.listdir(videos[i] + frames_path) if
                  f.endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()
        while True:
            Xims = torch.zeros((num_frames, shape_r, shape_c, 3))
            X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
            Xims[0:min(len(images) - start, num_frames), :] = torch.from_numpy(X.copy())
            Xims[min(len(images) - start, num_frames):num_frames, :] = torch.from_numpy(X.copy()[-1])
            Xims = Xims.permute(0, 3, 1, 2)

            prediction = net(Xims.cuda()).cpu().detach()
            for j in range(num_frames):
                res = postprocess_predictions(prediction[j, 0, :, :].numpy(), normal_shape_r,normal_shape_c)
                imageio.imsave(output_folder + '%s' % images_names[min(start + j, len(images) - 1)], res.astype(int))
            start = min(start + num_frames, len(images))
            if start == len(images):
                break


# visualize some predictions in the training process
def pred_small_data(net, iter):
    videos = [videos_small_test_path + f for f in os.listdir(videos_small_test_path) if os.path.isdir(videos_small_test_path + f)]
    videos.sort()
    nb_videos_test = len(videos) if len(videos) < nb_small_test_batch else nb_small_test_batch

    for i in range(nb_videos_test):
        output_folder = output_path + 'train/' + videos[i][-4: ] + '/iter_%d'%(iter) + '/'
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        images_names = [f for f in os.listdir(videos[i] + frames_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
        images_names.sort()

        print("Predicting saliency maps for " + videos[i])
        start = 0
        images = [videos[i] + frames_path + f for f in os.listdir(videos[i] + frames_path) if
                  f.endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()
        Xims = torch.zeros((num_frames, shape_r, shape_c, 3))
        X = preprocess_images(images[start:min(start + num_frames, len(images))], shape_r, shape_c)
        Xims[0:min(len(images) - start, num_frames), :] = torch.from_numpy(X.copy())
        Xims = Xims.permute(0, 3, 1, 2)

        prediction = net(Xims.cuda()).cpu().detach()
        for j in range(num_frames):
            res = postprocess_predictions(prediction[j, 0, :, :].numpy(), normal_shape_r, normal_shape_c)
            imageio.imsave(output_folder + '%s' % images_names[j], res.astype(int))


# training the dataset
def train_data(net):
    dataset = generator(video_b_s=video_b_s)
    criterion = ACLLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-5)
    avg_loss, avg_kl, avg_cc, avg_nss = 0, 0, 0, 0
    min_loss = 1e10
    for i in range(nb_train_pt):
        Xims, Ymaps, Yfixs = next(dataset)
        optimizer.zero_grad()
        output = net(Xims.cuda())
        loss, loss_kl, loss_cc, loss_nss = criterion(output, Ymaps.cuda(), Yfixs.cuda())
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
        avg_kl += loss_kl.item()
        avg_cc += loss_cc.item()
        avg_nss += loss_nss.item()
        if i > 0 and i % 100 == 0:
            avg_loss_100 = avg_kl / 10 + avg_cc / 100 + avg_nss / 100
            print("avg loss, kl, cc, nss: %.4f %.4f %.4f %.4f" % (
                avg_loss_100, avg_kl / 100, avg_cc / 100, avg_nss / 100))
            avg_kl = avg_cc = avg_nss = 0
            if avg_loss_100 < min_loss:
                min_loss = avg_loss_100
                net.eval()
                pred_small_data(net, i)
                net.train()
                torch.save({'epoch': i, 'state_dict': net.state_dict(), 'best_loss': avg_loss_100,
                            'optimizer': optimizer.state_dict()}, './m3_' + str("%d_%.4f" % (i, avg_loss_100)) + '.pth.tar')
            if i % 1000 == 0:
                net.eval()
                pred_small_data(net, i)
                net.train()
                avg_loss /= 1000
                torch.save({'epoch': i, 'state_dict': net.state_dict(), 'best_loss': avg_loss,
                            'optimizer': optimizer.state_dict()}, './m3_' + str("%d_%.4f" % (i, avg_loss)) + '.pth.tar')
                avg_loss = 0


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    # settings
    parser.add_argument('--mode', help='train or test', default='train')
    parser.add_argument('--weight', help='load pretrained network', default='')
    args = parser.parse_args()
    net = acl_net_pt().cuda()
    if args.mode == 'test':
        iter_nb = 0
        if not args.weight:
            raise Exception("No weight find!")
        cp = torch.load(args.weight)
        net.load_state_dict(cp['state_dict'])
        net.eval()
        pred_data_all(net)
    elif args.mode == 'train':
        if args.weight:
            cp = torch.load(args.weight)
            net.load_state_dict(cp['state_dict'])
        print('Start training!')
        net.train()
        pred_small_data(net, 0)
        train_data(net)



