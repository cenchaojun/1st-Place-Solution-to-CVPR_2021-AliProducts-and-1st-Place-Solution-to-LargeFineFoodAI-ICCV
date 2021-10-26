#coding=utf8
from __future__ import print_function, division

import os, time, datetime
import numpy as np
from math import ceil
import datetime

import torch
from torch import nn
from torch.autograd import Variable
#from torchvision.utils import make_grid, save_image

from utils.utils import LossRecord, clip_gradient
from models.focal_loss import FocalLoss
from utils.eval_model import eval_turn
from utils.Asoftmax_loss import AngleLoss

import pdb


def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def train(Config,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          exp_lr_scheduler,
          data_loader,
          save_dir,
          data_size=448,
          savepoint=500,
          checkpoint=1000):
    # savepoint: save without evalution
    # checkpoint: save with evaluation

    step = 0
    eval_train_flag = False
    rec_loss = []
    checkpoint_list = []

    date_suffix = dt()
    log_file = open(
        os.path.join(
            Config.log_folder,
            'formal_log_r50_dcl_%s_%s.log' % (str(data_size), date_suffix)),
        'a')

    add_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()
    get_focal_loss = FocalLoss()
    get_angle_loss = AngleLoss()

    for epoch in range(start_epoch, epoch_num - 1):
        exp_lr_scheduler.step(epoch)
        model.train(True)

        save_grad = []
        data_name = 'train'

        train_batch_size = data_loader[data_name].batch_size
        train_epoch_step = data_loader[data_name].__len__()
        train_loss_recorder = LossRecord(train_batch_size)

        if savepoint > train_epoch_step:
            savepoint = 1 * train_epoch_step
            checkpoint = savepoint

        for batch_cnt, data in enumerate(data_loader[data_name]):
            step += 1
            loss = 0
            model.train(True)
            if Config.use_backbone:
                inputs, labels, img_names = data
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(
                        torch.from_numpy(np.array(labels)).cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(torch.from_numpy(np.array(labels)))

            if Config.use_dcl:
                inputs, labels, labels_swap, swap_law, img_names = data
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda())
                    labels = Variable(
                        torch.from_numpy(np.array(labels)).cuda())
                    labels_swap = Variable(
                        torch.from_numpy(np.array(labels_swap)).cuda())
                    swap_law = Variable(
                        torch.from_numpy(np.array(swap_law)).float().cuda())
                else:
                    inputs = Variable(inputs)
                    labels = Variable(torch.from_numpy(np.array(labels)))
                    labels_swap = Variable(
                        torch.from_numpy(np.array(labels_swap)))
                    swap_law = Variable(
                        torch.from_numpy(np.array(swap_law)).float())

            optimizer.zero_grad()

            if inputs.size(0) < 2 * train_batch_size:
                outputs = model(inputs, inputs[0:-1:2])
            else:
                outputs = model(inputs, None)

            if Config.use_focal_loss:
                ce_loss = get_focal_loss(outputs[0], labels)
            else:
                if isinstance(outputs[0], tuple):
                    loss_list = []
                    for o in outputs[0]:
                        loss_list.append(
                            get_ce_loss(o, labels) / len(outputs[0]))
                    #loss_list = [
                    #    get_ce_loss(o, labels) / len(outputs[0])
                    #    for o in outputs[0]
                    #]
                    ce_loss = sum(loss_list)
                else:
                    ce_loss = get_ce_loss(outputs[0], labels)

                #ce_loss = get_ce_loss(outputs[0], labels)

            if Config.use_Asoftmax:
                fetch_batch = labels.size(0)
                if batch_cnt % (train_epoch_step // 5) == 0:
                    angle_loss = get_angle_loss(outputs[3],
                                                labels[0:fetch_batch:2],
                                                decay=0.9)
                else:
                    angle_loss = get_angle_loss(outputs[3],
                                                labels[0:fetch_batch:2])
                loss += angle_loss

            loss += ce_loss

            alpha_ = 1
            beta_ = 1
            gamma_ = 0.01 if Config.dataset == 'STCAR' or Config.dataset == 'AIR' else 1
            if Config.use_dcl:
                swap_loss = get_ce_loss(outputs[1], labels_swap) * beta_
                loss += swap_loss
                law_loss = add_loss(outputs[2], swap_law) * gamma_
                loss += law_loss

            loss.backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            if Config.use_dcl:
                print(
                    'step: {:-8d} / {:d} lr:{:6.5f} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} + {:6.4f} + {:6.4f} '
                    .format(step, train_epoch_step,
                            optimizer.param_groups[0]["lr"],
                            loss.detach().item(),
                            ce_loss.detach().item(),
                            swap_loss.detach().item(),
                            law_loss.detach().item()),
                    flush=True)
            if Config.use_backbone:
                print(
                    'step: {:-8d} / {:d} lr:{:6.5f} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} '
                    .format(step, train_epoch_step,
                            optimizer.param_groups[0]["lr"],
                            loss.detach().item(),
                            ce_loss.detach().item()),
                    flush=True)
            rec_loss.append(loss.detach().item())

            train_loss_recorder.update(loss.detach().item())

            # evaluation & save
            if step % checkpoint == 0:
                rec_loss = []
                print(32 * '-', flush=True)
                print(
                    'step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'
                    .format(step, train_epoch_step,
                            1.0 * step / train_epoch_step, epoch,
                            train_loss_recorder.get_val()),
                    flush=True)
                print('current lr:%s' % exp_lr_scheduler.get_lr(), flush=True)
                if eval_train_flag:
                    trainval_acc1, trainval_acc2, trainval_acc3 = eval_turn(
                        Config, model, data_loader['trainval'], 'trainval',
                        epoch, log_file)
                    if abs(trainval_acc1 - trainval_acc3) < 0.01:
                        eval_train_flag = False

                val_acc1, val_acc2, val_acc3 = eval_turn(
                    Config, model, data_loader['val'], 'val', epoch, log_file)

                save_path = os.path.join(
                    save_dir, 'weights_%d_%d_%.4f_%.4f.pth' %
                    (epoch, batch_cnt, val_acc1, val_acc3))
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                torch.save(model.state_dict(), save_path)
                print('saved model to %s' % (save_path), flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # save only
            elif step % savepoint == 0:
                train_loss_recorder.update(rec_loss)
                rec_loss = []
                save_path = os.path.join(
                    save_dir, 'savepoint_weights-%d-%s.pth' % (step, dt()))

                checkpoint_list.append(save_path)
                if len(checkpoint_list) == 6:
                    os.remove(checkpoint_list[0])
                    del checkpoint_list[0]
                torch.save(model.state_dict(), save_path)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    log_file.close()
