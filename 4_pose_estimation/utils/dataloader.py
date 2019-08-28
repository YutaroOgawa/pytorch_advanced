# 第4章姿勢推定のデータオーギュメンテーション
# 実装の一部参考に使用
# https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/
# Released under the MIT license

import os
import os.path as osp
import json
from PIL import Image
import cv2
import numpy as np
from scipy import misc, ndimage
import torch
import torch.utils.data as data

from utils.data_augumentation import Compose, get_anno, add_neck, aug_scale, aug_rotate, aug_croppad, aug_flip, remove_illegal_joint, Normalize_Tensor, no_Normalize_Tensor


def putGaussianMaps(center, accumulate_confid_map, params_transform):
    '''ガウスマップに変換する'''
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    stride = params_transform['stride']
    sigma = params_transform['sigma']

    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    start = stride / 2.0 - 0.5
    y_range = [i for i in range(int(grid_y))]
    x_range = [i for i in range(int(grid_x))]
    xx, yy = np.meshgrid(x_range, y_range)
    xx = xx * stride + start
    yy = yy * stride + start
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    mask = exponent <= 4.6052
    cofid_map = np.exp(-exponent)
    cofid_map = np.multiply(mask, cofid_map)
    accumulate_confid_map += cofid_map
    accumulate_confid_map[accumulate_confid_map > 1.0] = 1.0

    return accumulate_confid_map


def putVecMaps(centerA, centerB, accumulate_vec_map, count, params_transform):
    '''Parts A Fieldのベクトルを求める'''

    centerA = centerA.astype(float)
    centerB = centerB.astype(float)

    stride = params_transform['stride']
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    thre = params_transform['limb_width']   # limb width
    centerB = centerB / stride
    centerA = centerA / stride

    limb_vec = centerB - centerA
    norm = np.linalg.norm(limb_vec)
    if (norm == 0.0):
        # print 'limb is too short, ignore it...'
        return accumulate_vec_map, count
    limb_vec_unit = limb_vec / norm
    # print 'limb unit vector: {}'.format(limb_vec_unit)

    # To make sure not beyond the border of this two points
    min_x = max(int(round(min(centerA[0], centerB[0]) - thre)), 0)
    max_x = min(int(round(max(centerA[0], centerB[0]) + thre)), grid_x)
    min_y = max(int(round(min(centerA[1], centerB[1]) - thre)), 0)
    max_y = min(int(round(max(centerA[1], centerB[1]) + thre)), grid_y)

    range_x = list(range(int(min_x), int(max_x), 1))
    range_y = list(range(int(min_y), int(max_y), 1))
    xx, yy = np.meshgrid(range_x, range_y)
    ba_x = xx - centerA[0]  # the vector from (x,y) to centerA
    ba_y = yy - centerA[1]
    limb_width = np.abs(ba_x * limb_vec_unit[1] - ba_y * limb_vec_unit[0])
    mask = limb_width < thre  # mask is 2D

    vec_map = np.copy(accumulate_vec_map) * 0.0
    vec_map[yy, xx] = np.repeat(mask[:, :, np.newaxis], 2, axis=2)
    vec_map[yy, xx] *= limb_vec_unit[np.newaxis, np.newaxis, :]

    mask = np.logical_or.reduce(
        (np.abs(vec_map[:, :, 0]) > 0, np.abs(vec_map[:, :, 1]) > 0))

    accumulate_vec_map = np.multiply(
        accumulate_vec_map, count[:, :, np.newaxis])
    accumulate_vec_map += vec_map
    count[mask == True] += 1

    mask = count == 0

    count[mask == True] = 1

    accumulate_vec_map = np.divide(accumulate_vec_map, count[:, :, np.newaxis])
    count[mask == True] = 0

    return accumulate_vec_map, count


def get_ground_truth(meta, mask_miss):
    """アノテーションとマスクデータから正しい答えを求める"""

    # 初期設定
    params_transform = dict()
    params_transform['stride'] = 8  # 画像サイズを変更したくない場合は1にする
    params_transform['mode'] = 5
    params_transform['crop_size_x'] = 368
    params_transform['crop_size_y'] = 368
    params_transform['np'] = 56
    params_transform['sigma'] = 7.0
    params_transform['limb_width'] = 1.0

    stride = params_transform['stride']
    mode = params_transform['mode']
    crop_size_y = params_transform['crop_size_y']
    crop_size_x = params_transform['crop_size_x']
    num_parts = params_transform['np']
    nop = meta['numOtherPeople']

    # 画像サイズ
    grid_y = crop_size_y / stride
    grid_x = crop_size_x / stride
    channels = (num_parts + 1) * 2

    # 格納する変数
    heatmaps = np.zeros((int(grid_y), int(grid_x), 19))
    pafs = np.zeros((int(grid_y), int(grid_x), 38))

    mask_miss = cv2.resize(mask_miss, (0, 0), fx=1.0 / stride, fy=1.0 /
                           stride, interpolation=cv2.INTER_CUBIC).astype(
        np.float32)
    mask_miss = mask_miss / 255.
    mask_miss = np.expand_dims(mask_miss, axis=2)

    # マスク変数
    heat_mask = np.repeat(mask_miss, 19, axis=2)
    paf_mask = np.repeat(mask_miss, 38, axis=2)

    # ピンポイントの座標情報をガウス分布にぼやっとさせる
    for i in range(18):
        if (meta['joint_self'][i, 2] <= 1):
            center = meta['joint_self'][i, :2]
            gaussian_map = heatmaps[:, :, i]
            heatmaps[:, :, i] = putGaussianMaps(
                center, gaussian_map, params_transform)
        for j in range(nop):
            if (meta['joint_others'][j, i, 2] <= 1):
                center = meta['joint_others'][j, i, :2]
                gaussian_map = heatmaps[:, :, i]
                heatmaps[:, :, i] = putGaussianMaps(
                    center, gaussian_map, params_transform)
    # pafs
    mid_1 = [2, 9, 10, 2, 12, 13, 2, 3, 4,
             3, 2, 6, 7, 6, 2, 1, 1, 15, 16]

    mid_2 = [9, 10, 11, 12, 13, 14, 3, 4, 5,
             17, 6, 7, 8, 18, 1, 15, 16, 17, 18]

    thre = 1
    for i in range(19):
        # limb

        count = np.zeros((int(grid_y), int(grid_x)), dtype=np.uint32)
        if (meta['joint_self'][mid_1[i] - 1, 2] <= 1 and meta['joint_self'][mid_2[i] - 1, 2] <= 1):
            centerA = meta['joint_self'][mid_1[i] - 1, :2]
            centerB = meta['joint_self'][mid_2[i] - 1, :2]
            vec_map = pafs[:, :, 2 * i:2 * i + 2]
            #                    print vec_map.shape

            pafs[:, :, 2 * i:2 * i + 2], count = putVecMaps(centerA=centerA,
                                                            centerB=centerB,
                                                            accumulate_vec_map=vec_map,
                                                            count=count, params_transform=params_transform)
        for j in range(nop):
            if (meta['joint_others'][j, mid_1[i] - 1, 2] <= 1 and meta['joint_others'][j, mid_2[i] - 1, 2] <= 1):
                centerA = meta['joint_others'][j, mid_1[i] - 1, :2]
                centerB = meta['joint_others'][j, mid_2[i] - 1, :2]
                vec_map = pafs[:, :, 2 * i:2 * i + 2]
                pafs[:, :, 2 * i:2 * i + 2], count = putVecMaps(centerA=centerA,
                                                                centerB=centerB,
                                                                accumulate_vec_map=vec_map,
                                                                count=count, params_transform=params_transform)
    # background
    heatmaps[:, :, -
             1] = np.maximum(1 - np.max(heatmaps[:, :, :18], axis=2), 0.)

    # Tensorに
    heat_mask = torch.from_numpy(heat_mask)
    heatmaps = torch.from_numpy(heatmaps)
    paf_mask = torch.from_numpy(paf_mask)
    pafs = torch.from_numpy(pafs)

    return heat_mask, heatmaps, paf_mask, pafs


def make_datapath_list(rootpath):
    """
    学習、検証の画像データとアノテーションデータ、マスクデータへのファイルパスリストを作成する。
    """

    # アノテーションのJSONファイルを読み込む
    json_path = osp.join(rootpath, 'COCO.json')
    with open(json_path) as data_file:
        data_this = json.load(data_file)
        data_json = data_this['root']

    # indexを格納
    num_samples = len(data_json)
    train_indexes = []
    val_indexes = []
    for count in range(num_samples):
        if data_json[count]['isValidation'] != 0.:
            val_indexes.append(count)
        else:
            train_indexes.append(count)

    # 画像ファイルパスを格納
    train_img_list = list()
    val_img_list = list()

    for idx in train_indexes:
        img_path = os.path.join(rootpath, data_json[idx]['img_paths'])
        train_img_list.append(img_path)

    for idx in val_indexes:
        img_path = os.path.join(rootpath, data_json[idx]['img_paths'])
        val_img_list.append(img_path)

    # マスクデータのパスを格納
    train_mask_list = []
    val_mask_list = []

    for idx in train_indexes:
        img_idx = data_json[idx]['img_paths'][-16:-4]
        anno_path = "./data/mask/train2014/mask_COCO_train2014_" + img_idx+'.jpg'
        train_mask_list.append(anno_path)

    for idx in val_indexes:
        img_idx = data_json[idx]['img_paths'][-16:-4]
        anno_path = "./data/mask/val2014/mask_COCO_val2014_" + img_idx+'.jpg'
        val_mask_list.append(anno_path)

    # アノテーションデータを格納
    train_meta_list = list()
    val_meta_list = list()

    for idx in train_indexes:
        train_meta_list.append(data_json[idx])

    for idx in val_indexes:
        val_meta_list.append(data_json[idx])

    return train_img_list, train_mask_list, val_img_list, val_mask_list, train_meta_list, val_meta_list


class DataTransform():
    """
    画像とマスク、アノテーションの前処理クラス。
    学習時と推論時で異なる動作をする。
    学習時はデータオーギュメンテーションする。
    """

    def __init__(self):

        self.data_transform = {
            'train': Compose([
                get_anno(),  # JSONからアノテーションを辞書に格納
                add_neck(),  # アノテーションデータの順番を変更し、さらに首のアノテーションデータを追加
                aug_scale(),  # 拡大縮小
                aug_rotate(),  # 回転
                aug_croppad(),  # 切り出し
                aug_flip(),  # 左右反転
                remove_illegal_joint(),  # 画像からはみ出たアノテーションを除去
                Normalize_Tensor()  # 色情報の標準化とテンソル化
                # no_Normalize_Tensor()  # 本節のみ、色情報の標準化をなくす
            ]),
            'val': Compose([
                # 本書では検証は省略
            ])
        }

    def __call__(self, phase, meta_data, img, mask_miss):
        """
        Parameters
        ----------
        phase : 'train' or 'val'
            前処理のモードを指定。
        """
        meta_data, img, mask_miss = self.data_transform[phase](
            meta_data, img, mask_miss)

        return meta_data, img, mask_miss


class COCOkeypointsDataset(data.Dataset):
    """
    MSCOCOのCocokeypointsのDatasetを作成するクラス。PyTorchのDatasetクラスを継承。

    Attributes
    ----------
    img_list : リスト
        画像のパスを格納したリスト
    anno_list : リスト
        アノテーションへのパスを格納したリスト
    phase : 'train' or 'test'
        学習か訓練かを設定する。
    transform : object
        前処理クラスのインスタンス

    """

    def __init__(self, img_list, mask_list, meta_list, phase, transform):
        self.img_list = img_list
        self.mask_list = mask_list
        self.meta_list = meta_list
        self.phase = phase
        self.transform = transform

    def __len__(self):
        '''画像の枚数を返す'''
        return len(self.img_list)

    def __getitem__(self, index):
        img, heatmaps, heat_mask, pafs, paf_mask = self.pull_item(index)
        return img, heatmaps, heat_mask, pafs, paf_mask

    def pull_item(self, index):
        '''画像のTensor形式のデータ、アノテーション、マスクを取得する'''

        # 1. 画像読み込み
        image_file_path = self.img_list[index]
        img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]

        # 2. マスクとアノテーション読み込み
        mask_miss = cv2.imread(self.mask_list[index])
        meat_data = self.meta_list[index]

        # 3. 画像前処理
        meta_data, img, mask_miss = self.transform(
            self.phase, meat_data, img, mask_miss)

        # 4. 正解アノテーションデータの取得
        mask_miss_numpy = mask_miss.numpy().transpose((1, 2, 0))
        heat_mask, heatmaps, paf_mask, pafs = get_ground_truth(
            meta_data, mask_miss_numpy)

        # 5. マスクデータはRGBが(1,1,1)か(0,0,0)なので、次元を落とす
        heat_mask = heat_mask[:, :, :, 0]
        paf_mask = paf_mask[:, :, :, 0]

        # 6. チャネルが最後尾にあるので順番を変える
        # 例：paf_mask：torch.Size([46, 46, 38])
        # → torch.Size([38, 46, 46])
        paf_mask = paf_mask.permute(2, 0, 1)
        heat_mask = heat_mask.permute(2, 0, 1)
        pafs = pafs.permute(2, 0, 1)
        heatmaps = heatmaps.permute(2, 0, 1)

        return img, heatmaps, heat_mask, pafs, paf_mask
