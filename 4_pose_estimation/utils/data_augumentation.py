# 第4章姿勢推定のデータオーギュメンテーション
# 実装の一部参考に使用
# https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation/
# のImageAugmentation.py
# Released under the MIT license


# パッケージのimport
import cv2
import numpy as np
import random

import torch
from torchvision import transforms


class Compose(object):
    """引数transformに格納された変形を順番に実行するクラス
       対象画像、マスク画像、アノテーション画像を同時に変換させます。 
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, meta_data, img, mask_miss):
        for t in self.transforms:
            meta_data, img, mask_miss = t(meta_data, img, mask_miss)

        return meta_data, img, mask_miss


class get_anno(object):
    """JSON形式のアノテーションデータを辞書オブジェクトに格納"""

    def __call__(self, meta_data, img, mask_miss):
        anno = dict()
        anno['dataset'] = meta_data['dataset']
        anno['img_height'] = int(meta_data['img_height'])
        anno['img_width'] = int(meta_data['img_width'])

        anno['isValidation'] = meta_data['isValidation']
        anno['people_index'] = int(meta_data['people_index'])
        anno['annolist_index'] = int(meta_data['annolist_index'])

        # (b) objpos_x (float), objpos_y (float)
        anno['objpos'] = np.array(meta_data['objpos'])
        anno['scale_provided'] = meta_data['scale_provided']
        anno['joint_self'] = np.array(meta_data['joint_self'])

        anno['numOtherPeople'] = int(meta_data['numOtherPeople'])
        anno['num_keypoints_other'] = np.array(
            meta_data['num_keypoints_other'])
        anno['joint_others'] = np.array(meta_data['joint_others'])
        anno['objpos_other'] = np.array(meta_data['objpos_other'])
        anno['scale_provided_other'] = meta_data['scale_provided_other']
        anno['bbox_other'] = meta_data['bbox_other']
        anno['segment_area_other'] = meta_data['segment_area_other']

        if anno['numOtherPeople'] == 1:
            anno['joint_others'] = np.expand_dims(anno['joint_others'], 0)
            anno['objpos_other'] = np.expand_dims(anno['objpos_other'], 0)

        meta_data = anno

        return meta_data, img, mask_miss


class add_neck(object):
    '''
    アノテーションデータの順番を変更し、さらに首のアノテーションデータを追加します。
    首ポジションは両肩の位置から計算します。

    MS COCO annotation order:
    0: nose	   		1: l eye		2: r eye	3: l ear	4: r ear
    5: l shoulder	6: r shoulder	7: l elbow	8: r elbow
    9: l wrist		10: r wrist		11: l hip	12: r hip	13: l knee
    14: r knee		15: l ankle		16: r ankle
    The order in this work:
    (0-'nose'	1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
    5-'left_shoulder' 6-'left_elbow'	    7-'left_wrist'  8-'right_hip'
    9-'right_knee'	 10-'right_ankle'	11-'left_hip'   12-'left_knee'
    13-'left_ankle'	 14-'right_eye'	    15-'left_eye'   16-'right_ear'
    17-'left_ear' )
    '''

    def __call__(self, meta_data, img, mask_miss):
        meta = meta_data
        our_order = [0, 17, 6, 8, 10, 5, 7, 9,
                     12, 14, 16, 11, 13, 15, 2, 1, 4, 3]
        # Index 6 is right shoulder and Index 5 is left shoulder
        right_shoulder = meta['joint_self'][6, :]
        left_shoulder = meta['joint_self'][5, :]
        neck = (right_shoulder + left_shoulder) / 2

        # right_shoulder[2]が値1のときはアノテーションがあり画像内に部位も見えている
        # 値0のときはアノテーションの座標情報はあるが、画像内に部位は映っていない
        # 値が2のときは画像内に写っておらず、アノテーション付けもない
        # ※注意　元のMSCOCOの定義と値の意味が変わっている
        # v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible.
        if right_shoulder[2] == 2 or left_shoulder[2] == 2:
            neck[2] = 2
        elif right_shoulder[2] == 1 or left_shoulder[2] == 1:
            neck[2] = 1
        else:
            neck[2] = right_shoulder[2] * left_shoulder[2]

        neck = neck.reshape(1, len(neck))
        neck = np.round(neck)
        meta['joint_self'] = np.vstack((meta['joint_self'], neck))
        meta['joint_self'] = meta['joint_self'][our_order, :]
        temp = []

        for i in range(meta['numOtherPeople']):
            right_shoulder = meta['joint_others'][i, 6, :]
            left_shoulder = meta['joint_others'][i, 5, :]
            neck = (right_shoulder + left_shoulder) / 2
            if (right_shoulder[2] == 2 or left_shoulder[2] == 2):
                neck[2] = 2
            elif (right_shoulder[2] == 1 or left_shoulder[2] == 1):
                neck[2] = 1
            else:
                neck[2] = right_shoulder[2] * left_shoulder[2]
            neck = neck.reshape(1, len(neck))
            neck = np.round(neck)
            single_p = np.vstack((meta['joint_others'][i], neck))
            single_p = single_p[our_order, :]
            temp.append(single_p)
        meta['joint_others'] = np.array(temp)

        meta_data = meta

        return meta_data, img, mask_miss


class aug_scale(object):
    def __init__(self):
        self.params_transform = dict()
        self.params_transform['scale_min'] = 0.5
        self.params_transform['scale_max'] = 1.1
        self.params_transform['target_dist'] = 0.6

    def __call__(self, meta_data, img, mask_miss):

        # ランダムに0.5倍～1.1倍する
        dice = random.random()  # (0,1)
        scale_multiplier = (
            self.params_transform['scale_max'] - self.params_transform['scale_min']) * dice + self.params_transform['scale_min']

        scale_abs = self.params_transform['target_dist'] / \
            meta_data['scale_provided']
        scale = scale_abs * scale_multiplier
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale,
                         interpolation=cv2.INTER_CUBIC)

        mask_miss = cv2.resize(mask_miss, (0, 0), fx=scale,
                               fy=scale, interpolation=cv2.INTER_CUBIC)

        # modify meta data
        meta_data['objpos'] *= scale
        meta_data['joint_self'][:, :2] *= scale
        if (meta_data['numOtherPeople'] != 0):
            meta_data['objpos_other'] *= scale
            meta_data['joint_others'][:, :, :2] *= scale
        return meta_data, img, mask_miss


class aug_rotate(object):
    def __init__(self):
        self.params_transform = dict()
        self.params_transform['max_rotate_degree'] = 40

    def __call__(self, meta_data, img, mask_miss):

        # ランダムに-40～40度回転
        dice = random.random()  # (0,1)
        degree = (dice - 0.5) * 2 * \
            self.params_transform['max_rotate_degree']  # degree [-40,40]

        def rotate_bound(image, angle, bordervalue):
            # grab the dimensions of the image and then determine the
            # center
            (h, w) = image.shape[:2]
            (cX, cY) = (w // 2, h // 2)

            # grab the rotation matrix (applying the negative of the
            # angle to rotate clockwise), then grab the sine and cosine
            # (i.e., the rotation components of the matrix)
            M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])

            # compute the new bounding dimensions of the image
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY

            # perform the actual rotation and return the image
            return cv2.warpAffine(image, M, (nW, nH), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=bordervalue), M

        def rotatepoint(p, R):
            point = np.zeros((3, 1))
            point[0] = p[0]
            point[1] = p[1]
            point[2] = 1

            new_point = R.dot(point)

            p[0] = new_point[0]

            p[1] = new_point[1]
            return p

        # 画像とマスク画像の回転
        img_rot, R = rotate_bound(img, np.copy(
            degree), (128, 128, 128))  # 回転でできた隙間は青色
        mask_miss_rot, _ = rotate_bound(
            mask_miss, np.copy(degree), (255, 255, 255))

        # アノテーションデータの回転
        meta_data['objpos'] = rotatepoint(meta_data['objpos'], R)

        for i in range(18):
            meta_data['joint_self'][i, :] = rotatepoint(
                meta_data['joint_self'][i, :], R)

        for j in range(meta_data['numOtherPeople']):

            meta_data['objpos_other'][j, :] = rotatepoint(
                meta_data['objpos_other'][j, :], R)

            for i in range(18):
                meta_data['joint_others'][j, i, :] = rotatepoint(
                    meta_data['joint_others'][j, i, :], R)

        return meta_data, img_rot, mask_miss_rot


class aug_croppad(object):
    def __init__(self):
        self.params_transform = dict()
        self.params_transform['center_perterb_max'] = 40
        self.params_transform['crop_size_x'] = 368
        self.params_transform['crop_size_y'] = 368

    def __call__(self, meta_data, img, mask_miss):

        # ランダムにオフセットを用意 -40から40
        dice_x = random.random()  # (0,1)
        dice_y = random.random()  # (0,1)
        crop_x = int(self.params_transform['crop_size_x'])
        crop_y = int(self.params_transform['crop_size_y'])
        x_offset = int((dice_x - 0.5) * 2 *
                       self.params_transform['center_perterb_max'])
        y_offset = int((dice_y - 0.5) * 2 *
                       self.params_transform['center_perterb_max'])

        center = meta_data['objpos'] + np.array([x_offset, y_offset])
        center = center.astype(int)

        # pad up and down
        # img.shape=（幅、高さ）
        pad_v = np.ones((crop_y, img.shape[1], 3), dtype=np.uint8) * 128
        pad_v_mask_miss = np.ones(
            (crop_y, mask_miss.shape[1], 3), dtype=np.uint8) * 255
        img = np.concatenate((pad_v, img, pad_v), axis=0)

        mask_miss = np.concatenate(
            (pad_v_mask_miss, mask_miss, pad_v_mask_miss), axis=0)

        # pad right and left
        # img.shape=（幅、高さ）
        pad_h = np.ones((img.shape[0], crop_x, 3), dtype=np.uint8) * 128
        pad_h_mask_miss = np.ones(
            (mask_miss.shape[0], crop_x, 3), dtype=np.uint8) * 255

        img = np.concatenate((pad_h, img, pad_h), axis=1)
        mask_miss = np.concatenate(
            (pad_h_mask_miss, mask_miss, pad_h_mask_miss), axis=1)

        # 切り出す
        img = img[int(center[1] + crop_y / 2):int(center[1] + crop_y / 2 + crop_y),
                  int(center[0] + crop_x / 2):int(center[0] + crop_x / 2 + crop_x), :]

        mask_miss = mask_miss[int(center[1] + crop_y / 2):int(center[1] + crop_y / 2 +
                                                              crop_y + 1*0), int(center[0] + crop_x / 2):int(center[0] + crop_x / 2 + crop_x + 1*0)]

        offset_left = crop_x / 2 - center[0]
        offset_up = crop_y / 2 - center[1]

        offset = np.array([offset_left, offset_up])
        meta_data['objpos'] += offset
        meta_data['joint_self'][:, :2] += offset

        # 画像からはみ出ていないかチェック
        # 条件式4つのORを計算する
        mask = np.logical_or.reduce((meta_data['joint_self'][:, 0] >= crop_x,
                                     meta_data['joint_self'][:, 0] < 0,
                                     meta_data['joint_self'][:, 1] >= crop_y,
                                     meta_data['joint_self'][:, 1] < 0))

        meta_data['joint_self'][mask == True, 2] = 2
        if (meta_data['numOtherPeople'] != 0):
            meta_data['objpos_other'] += offset
            meta_data['joint_others'][:, :, :2] += offset

            # 条件式4つのORを計算する
            mask = np.logical_or.reduce((meta_data['joint_others'][:, :, 0] >= crop_x,
                                         meta_data['joint_others'][:,
                                                                   :, 0] < 0,
                                         meta_data['joint_others'][:,
                                                                   :, 1] >= crop_y,
                                         meta_data['joint_others'][:, :, 1] < 0))

            meta_data['joint_others'][mask == True, 2] = 2

        return meta_data, img, mask_miss


class aug_flip(object):
    def __init__(self):
        self.params_transform = dict()
        self.params_transform['flip_prob'] = 0.5

    def __call__(self, meta_data, img, mask_miss):

        # ランダムにオフセットを用意 -40から40

        dice = random.random()  # (0,1)
        doflip = dice <= self.params_transform['flip_prob']

        if doflip:
            img = img.copy()
            cv2.flip(src=img, flipCode=1, dst=img)
            w = img.shape[1]  # img.shape=（幅、高さ）

            mask_miss = mask_miss.copy()
            cv2.flip(src=mask_miss, flipCode=1, dst=mask_miss)

            '''
            The order in this work:
                (0-'nose'   1-'neck' 2-'right_shoulder' 3-'right_elbow' 4-'right_wrist'
                5-'left_shoulder' 6-'left_elbow'        7-'left_wrist'  8-'right_hip'  
                9-'right_knee'   10-'right_ankle'   11-'left_hip'   12-'left_knee' 
                13-'left_ankle'  14-'right_eye'     15-'left_eye'   16-'right_ear' 
                17-'left_ear' )
            '''
            meta_data['objpos'][0] = w - 1 - meta_data['objpos'][0]
            meta_data['joint_self'][:, 0] = w - \
                1 - meta_data['joint_self'][:, 0]
            # print meta['joint_self']
            meta_data['joint_self'] = meta_data['joint_self'][[0, 1, 5, 6,
                                                               7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]]

            num_other_people = meta_data['numOtherPeople']
            if (num_other_people != 0):
                meta_data['objpos_other'][:, 0] = w - \
                    1 - meta_data['objpos_other'][:, 0]
                meta_data['joint_others'][:, :, 0] = w - \
                    1 - meta_data['joint_others'][:, :, 0]
                for i in range(num_other_people):
                    meta_data['joint_others'][i] = meta_data['joint_others'][i][[
                        0, 1, 5, 6, 7, 2, 3, 4, 11, 12, 13, 8, 9, 10, 15, 14, 17, 16]]

        return meta_data, img, mask_miss


class remove_illegal_joint(object):
    """データオーギュメンテーションの結果、画像内から外れたパーツの位置情報を変更する"""

    def __init__(self):
        self.params_transform = dict()
        self.params_transform['crop_size_x'] = 368
        self.params_transform['crop_size_y'] = 368

    def __call__(self, meta_data, img, mask_miss):
        crop_x = int(self.params_transform['crop_size_x'])
        crop_y = int(self.params_transform['crop_size_y'])

        # 条件式4つのORを計算する
        mask = np.logical_or.reduce((meta_data['joint_self'][:, 0] >= crop_x,
                                     meta_data['joint_self'][:, 0] < 0,
                                     meta_data['joint_self'][:, 1] >= crop_y,
                                     meta_data['joint_self'][:, 1] < 0))

        # 画像内の枠からはみ出たパーツの位置情報は(1,1,2)にする
        meta_data['joint_self'][mask == True, :] = (1, 1, 2)

        if (meta_data['numOtherPeople'] != 0):
            mask = np.logical_or.reduce((meta_data['joint_others'][:, :, 0] >= crop_x,
                                         meta_data['joint_others'][:,
                                                                   :, 0] < 0,
                                         meta_data['joint_others'][:,
                                                                   :, 1] >= crop_y,
                                         meta_data['joint_others'][:, :, 1] < 0))
            meta_data['joint_others'][mask == True, :] = (1, 1, 2)

        return meta_data, img, mask_miss


class Normalize_Tensor(object):
    def __init__(self):
        self.color_mean = [0.485, 0.456, 0.406]
        self.color_std = [0.229, 0.224, 0.225]

    def __call__(self, meta_data, img, mask_miss):

        # 画像の大きさは最大1に規格化される
        img = img.astype(np.float32) / 255.

        # 色情報の標準化
        preprocessed_img = img.copy()[:, :, ::-1]  # BGR→RGB

        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:,
                                                         :, i] - self.color_mean[i]
            preprocessed_img[:, :, i] = preprocessed_img[:,
                                                         :, i] / self.color_std[i]

        # （幅、高さ、色）→（色、幅、高さ）
        img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
        mask_miss = mask_miss.transpose((2, 0, 1)).astype(np.float32)

        # 画像をTensorに
        img = torch.from_numpy(img)
        mask_miss = torch.from_numpy(mask_miss)

        return meta_data, img, mask_miss


class no_Normalize_Tensor(object):
    def __init__(self):
        self.color_mean = [0, 0, 0]
        self.color_std = [1, 1, 1]

    def __call__(self, meta_data, img, mask_miss):

        # 画像の大きさは最大1に規格化される
        img = img.astype(np.float32) / 255.

        # 色情報の標準化
        preprocessed_img = img.copy()[:, :, ::-1]  # BGR→RGB

        for i in range(3):
            preprocessed_img[:, :, i] = preprocessed_img[:,
                                                         :, i] - self.color_mean[i]
            preprocessed_img[:, :, i] = preprocessed_img[:,
                                                         :, i] / self.color_std[i]

        # （幅、高さ、色）→（色、幅、高さ）
        img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)
        mask_miss = mask_miss.transpose((2, 0, 1)).astype(np.float32)

        # 画像をTensorに
        img = torch.from_numpy(img)
        mask_miss = torch.from_numpy(mask_miss)

        return meta_data, img, mask_miss
