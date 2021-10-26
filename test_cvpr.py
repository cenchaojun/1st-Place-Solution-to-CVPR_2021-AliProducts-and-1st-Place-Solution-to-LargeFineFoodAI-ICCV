#coding=utf-8
import os
import sys

os.environ['CUDA_DEVICE_ORDRE'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'

import json
import csv
import argparse
import pandas as pd
import numpy as np
from math import ceil
from tqdm import tqdm
import pickle
import shutil
import collections
from sklearn.metrics.pairwise import cosine_similarity, paired_distances

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss
from torchvision import datasets, models
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from transforms import transforms
from models.LoadModel import MainModel
from utils.dataset_DCL import collate_fn4cvpr, dataset
from config import LoadConfig, load_data_transformers
from utils.test_tool import set_text, save_multi_img, cls_base_acc


def parse_args():
    parser = argparse.ArgumentParser(description='dcl parameters')
    parser.add_argument('--data', dest='dataset', default='food', type=str)
    parser.add_argument('--backbone',
                        dest='backbone',
                        default='dm_nfnet_f3',
                        type=str)
    parser.add_argument('--b', dest='batch_size', default=104, type=int)
    parser.add_argument('--nw', dest='num_workers', default=32, type=int)
    parser.add_argument('--ver', dest='version', default='test', type=str)
    parser.add_argument(
        '--save',
        dest='resume',
        default='./net_model/_83113_food/weights_5_2316_0.9885_0.9964.pth',
        type=str)
    parser.add_argument('--size',
                        dest='resize_resolution',
                        default=438,
                        type=int)
    parser.add_argument('--crop',
                        dest='crop_resolution',
                        default=384,
                        type=int)
    parser.add_argument('--ss', dest='save_suffix', default=None, type=str)
    parser.add_argument('--acc_report',
                        dest='acc_report',
                        default=True,
                        type=bool)
    parser.add_argument('--swap_num',
                        default=[7, 7],
                        nargs=2,
                        metavar=('swap1', 'swap2'),
                        type=int,
                        help='specify a range')
    args = parser.parse_args()
    return args


def merge_crop_files(file_list):
    for ind, file in enumerate(file_list):
        out_name = file.replace('crop_', '')
        if os.path.exists(out_name):
            continue
        probs = {}
        headers = ['id', 'probs']
        rows = []
        if 'crop_merge' in file:
            data = pd.read_csv(file)
            for index, row in data.iterrows():
                sys.stdout.write('\r%d:%d-%d' % (ind, len(file_list), index))
                img = row.id
                values = eval(row.probs)
                if img not in probs:
                    probs[img] = {
                        key: [value]
                        for key, value in values.items()
                    }
                else:
                    for key, value in values.items():
                        if key not in probs[img]:
                            probs[img][key] = [value]
                        else:
                            probs[img][key].append(value)

        for img, values in probs.items():
            for key, value in values.items():
                probs[img][key] = sum(value) / len(value)
            probs[img] = sorted(probs[img].items(),
                                key=lambda a: a[1],
                                reverse=True)[:10]
            rows.append((img, {k: round(v, 8) for k, v in probs[img]}))
        with open(out_name, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)


def merge_models_results(file_list):
    headers = ['id', 'predicted']
    rows = []
    probs = collections.OrderedDict()
    for ind, file in enumerate(file_list):
        if 'vit' in file:
            rate = 1
        else:
            rate = 1

        if "submission_" in file:
            data = pd.read_csv(file)
            for index, row in data.iterrows():
                sys.stdout.write('\r%d:%d-%d' % (ind, len(file_list), index))
                img = row.id
                if img not in probs:
                    probs[img] = {
                        row.predict_1: [row.confid_1 * rate],
                        row.predict_2: [row.confid_2 * rate],
                        row.predict_3: [row.confid_3 * rate]
                    }

                else:
                    for key, value in [(row.predict_1, row.confid_1),
                                       (row.predict_2, row.confid_2),
                                       (row.predict_3, row.confid_3)]:
                        if key not in probs[img]:
                            probs[img][key] = [value * rate]
                        else:
                            probs[img][key].append(value * rate)

        elif "swin_" in file:
            data = pd.read_csv(file)
            for index, row in data.iterrows():
                sys.stdout.write('\r%d:%d-%d' % (ind, len(file_list), index))
                img = row.id
                if img not in probs:
                    probs[img] = {
                        row.top1_cls: [row.top1_prob * rate],
                        row.top2_cls: [row.top2_prob * rate],
                        row.top3_cls: [row.top3_prob * rate],
                        row.top4_cls: [row.top4_prob * rate],
                        row.top5_cls: [row.top5_prob * rate]
                    }

                else:
                    for key, value in [(row.top1_cls, row.top1_prob),
                                       (row.top2_cls, row.top2_prob),
                                       (row.top3_cls, row.top3_prob),
                                       (row.top4_cls, row.top4_prob),
                                       (row.top5_cls, row.top5_prob)]:

                        if key not in probs[img]:
                            probs[img][key] = [value * rate]
                        else:
                            probs[img][key].append(value * rate)
        else:
            data = pd.read_csv(file)
            for index, row in data.iterrows():
                sys.stdout.write('\r%d:%d-%d' % (ind, len(file_list), index))
                img = row.id
                values = eval(row.probs)
                if img not in probs:
                    probs[img] = {
                        key: [value * rate]
                        for key, value in values.items()
                    }
                else:
                    for key, value in values.items():
                        if key not in probs[img]:
                            probs[img][key] = [value * rate]
                        else:
                            probs[img][key].append(value * rate)
    for img, values in probs.items():
        for key, value in values.items():
            probs[img][key] = sum(value)  #/ len(value)
        probs[img] = sorted(probs[img].items(), key=lambda a: a[1])[-1][0]
        rows.append((img, probs[img]))
    with open('output/final.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def get_emb_avg_result(name, sub_cat, embed, dis='euclidean'):
    distance = []
    embeds = []
    for cat in sub_cat:
        class_dir = os.path.join('output/val', str(cat).zfill(5))
        class_embeds = np.array([
            np.load(os.path.join(class_dir, file))
            for file in os.listdir(class_dir)
        ])
        embeds.append(class_embeds.mean(axis=0))

    distance = paired_distances(np.array([embed] * len(sub_cat)),
                                np.array(embeds),
                                metric=dis)
    label = sub_cat[distance.argmin()]
    return label


def get_test_labeled_data(
        csv_file, test_dir='/data3/zengpeng/products/food/classify/Test_new'):
    data = pd.read_csv(csv_file)
    for index, row in data.iterrows():
        img = row.id
        label = row.predicted
        dst_dir = os.path.join(os.path.dirname(test_dir), 'Test', str(label))
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)
        shutil.copy(os.path.join(test_dir, img), dst_dir)


def get_clean_label_by_mul_model(file_list,
                                 source_dir='/data/zengpeng/products',
                                 image_save=''):
    rows = []
    probs = collections.OrderedDict()
    number = len(file_list)
    for ind, file in enumerate(file_list):
        data = pd.read_csv(file)
        for index, row in data.iterrows():
            sys.stdout.write('\r%d:%d' % (ind, index))
            img = row.id
            label = row.predicted
            if img not in probs:
                probs[img] = [label]
            else:
                probs[img].append(label)
    ret = []
    for img, labels in probs.items():
        image_id = os.path.basename(img)
        class_id = img.split('/')[1]
        if len(set(labels)) == 1 and class_id == str(labels[0]):
            ret.append({"class_id": str(labels[0]), "image_id": image_id})
            if image_save:
                save_dir = os.path.join(image_save, class_id)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                shutil.copy(os.path.join(source_dir, img), save_dir)
    ret = json.dumps({'images': ret}, indent=4)
    with open('output/train_clean_v1.json', 'w', encoding='utf-8') as f:
        f.write(ret)


if __name__ == '__main__':
    get_test_data = False
    merge_result = True
    ten_crop = True
    if get_test_data:
        csv_file = 'output/final.csv'
        get_test_labeled_data(csv_file)
        exit(0)
    if merge_result:
        file_list = [
            os.path.join('./output', file) for file in os.listdir('./output')
            if file.endswith('merge.csv')
        ]
        merge_crop_files(file_list)
        file_list = set([file.replace('crop_', '') for file in file_list])
        merge_models_results(file_list)
        exit(0)
    args = parse_args()
    use_gpu = True if torch.cuda.is_available() else False
    print(args)
    Config = LoadConfig(args, args.version)
    transformers = load_data_transformers(args.resize_resolution,
                                          args.crop_resolution, args.swap_num)
    data_set = dataset(Config,
                       anno=Config.test_anno,
                       swap=transformers["None"],
                       totensor=transformers['test_totensor']
                       if ten_crop else transformers['val_totensor'],
                       test=True)

    #for data in data_set:
    #    print(data)

    dataloader = torch.utils.data.DataLoader(data_set,\
                                             batch_size=args.batch_size,\
                                             shuffle=False,\
                                             num_workers=args.num_workers,\
                                             collate_fn=collate_fn4cvpr)

    setattr(dataloader, 'total_item_len', len(data_set))

    cudnn.benchmark = True

    model = MainModel(Config)
    model_dict = model.state_dict()
    if use_gpu:
        pretrained_dict = torch.load(args.resume)
    else:
        pretrained_dict = torch.load(args.resume, map_location='cpu')
    pretrained_dict = {
        k[7:]: v
        for k, v in pretrained_dict.items() if k[7:] in model_dict
    }
    print(
        f'model_dict: {len(model_dict)}, pretrained_dict: {len(pretrained_dict)}'
    )
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    if use_gpu:
        model.cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    headers = ['id', 'predicted']
    headers_merge = ['id', 'probs']
    rows = []
    rows_merge = []
    if not ten_crop:
        csv_name = "output/" + args.backbone + "_" + args.resume.split(
            '/')[-1][:-4] + '_' + str(args.crop_resolution) + ".csv"
        csv_name_merge = "output/" + args.backbone + "_" + args.resume.split(
            '/')[-1][:-4] + '_' + str(args.crop_resolution) + "_merge.csv"
    else:
        csv_name = "output/" + args.backbone + "_" + args.resume.split(
            '/')[-1][:-4] + '_' + str(args.crop_resolution) + "_crop.csv"
        csv_name_merge = "output/" + args.backbone + "_" + args.resume.split(
            '/')[-1][:-4] + '_' + str(args.crop_resolution) + "_crop_merge.csv"

    model.train(False)
    softmax = nn.Softmax(dim=1)

    with torch.no_grad():
        count_bar = tqdm(total=dataloader.__len__())
        for batch_cnt_val, data_val in enumerate(dataloader):
            count_bar.update(1)
            inputs, img_name = data_val
            if use_gpu:
                inputs = Variable(inputs.cuda())
            else:
                inputs = Variable(inputs)

            if ten_crop:
                bs, ncrops, c, h, w = inputs.size()
                inputs = inputs.view(-1, c, h, w)
                img_name = [val for val in img_name for i in range(5)]

            outputs = model(inputs)
            if Config.use_dcl:
                outputs_pred = outputs[0] + outputs[
                    1][:, 0:Config.numcls] + outputs[1][:, Config.numcls:2 *
                                                        Config.numcls]
            else:
                outputs_pred = outputs[0]

            top10_val, top10_pos = torch.topk(softmax(outputs_pred), 10)
            for sub_name, sub_cat, value, pred in zip(img_name,
                                                      top10_pos.tolist(),
                                                      top10_val.tolist(),
                                                      outputs_pred):
                ####使用验证集样本embeding平均后与pred的距离判断类别
                #sub_cat_avg = get_emb_avg_result(sub_name, sub_cat, pred.cpu().numpy())
                #rows.append((os.path.basename(sub_name), sub_cat_avg))

                ####softmax分类
                rows.append((os.path.basename(sub_name), sub_cat[0]))

                rows_merge.append(
                    (os.path.basename(sub_name),
                     {k: round(v, 8)
                      for k, v in zip(sub_cat, value)}))
        with open(csv_name, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
        with open(csv_name_merge, 'w', encoding='utf-8',
                  newline='') as f_merge:
            writer_merge = csv.writer(f_merge)
            writer_merge.writerow(headers_merge)
            writer_merge.writerows(rows_merge)
    count_bar.close()
