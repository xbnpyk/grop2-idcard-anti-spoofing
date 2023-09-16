# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np
import torch

from utils.general import (Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.torch_utils import select_device

def check_points_on_rectangle_edges(points, x1, y1, x2, y2):
    # 计算矩形的边界
    left = x1
    right = x2
    top = y1
    bottom = y2
    forgw = (x2-x1) * 0.05
    forgh = (y2-y1) * 0.05
    # 遍历每个点，检查是否在矩形的边界上
    for point in points:
        px, py = point  # 三个点中的一个点
        # 检查点是否在矩形的边界上
        if (
                (((left-forgw <= px <= left + forgw) or (right - forgw <= px <= right+forgw)) and (
                top-forgh <= py <= top + forgh)) or (bottom-forgh<=py<=bottom+forgh and (left <= px <= left + forgw or right - forgw <= px <= right+forgw))
        ):
            continue  # 点在边界上，继续检查下一个点
        else:
            return False  # 有一个点不在边界上，返回False
    # 所有点都在矩形的边界上
    return True


def pixel_area(pts):
    # 将坐标值转换为整数
    # pts=order_points1(pts)
    pts = np.round(pts).astype(int)

    # 计算四边形的边长
    side_lengths = []
    for i in range(4):
        side_lengths.append(np.linalg.norm(pts[(i + 1) % 4] - pts[i]))

    # 计算半周长
    s = sum(side_lengths) / 2.0

    # 使用海伦公式计算四边形的面积
    area = np.sqrt((s - side_lengths[0]) * (s - side_lengths[1]) * (s - side_lengths[2]) * (s - side_lengths[3]))

    return area


def calculate_angle(pt1, pt2, pt3):
    # 计算两个向量的夹角，角度范围在[0, 180]
    vector1 = pt1 - pt2
    vector2 = pt3 - pt2
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    cos_theta = dot_product / (norm1 * norm2)
    angle = np.arccos(cos_theta) * 180 / np.pi
    return angle


def is_document_distorted(pts):
    # 检测四个角点之间的夹角是否接近90度
    angle_threshold = 20  # 夹角阈值，可以根据具体情况调整
    # pts=order_points1(pts)

    angles = []
    for i in range(4):
        pt1, pt2, pt3 = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        angle = calculate_angle(pt1, pt2, pt3)
        angles.append(angle)

    # 如果任意一个角度不在阈值范围内，认为证件倾斜或畸变
    if any(abs(angle - 90) > angle_threshold for angle in angles) or qingxie(pts):
        return True
    else:
        return False


def qingxie(pts):
    # 计算角点坐标与水平直线之间的夹角
    horizontal_line_angle = np.arctan2(pts[1][1] - pts[0][1], pts[1][0] - pts[0][0])
    horizontal_line_angle_deg = np.degrees(horizontal_line_angle)

    # 计算角点坐标与垂直直线之间的夹角
    vertical_line_angle = np.arctan2(pts[3][1] - pts[0][1], pts[3][0] - pts[0][0])
    vertical_line_angle_deg = np.degrees(vertical_line_angle)

    # 设置阈值来判断图像是否倾斜，可以根据实际情况调整
    angle_threshold = 5.0  # 阈值，表示图像倾斜的最大角度

    # 如果水平和垂直夹角超过阈值，则判断图像倾斜
    if abs(horizontal_line_angle_deg) > angle_threshold or abs(vertical_line_angle_deg) > angle_threshold:
        return True
    else:
        return False


import math

def find_longest_line_segment(points):
    # 初始化最长线段长度和对应的点
    max_distance = 0
    longest_segment = ()

    # 遍历所有点的组合，计算距离并找到最长线段
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            point1 = points[i]
            point2 = points[j]
            # 计算两点之间的距离
            distance = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
            # 如果找到更长的线段，更新最长线段信息
            if distance > max_distance:
                max_distance = distance
                longest_segment = (point1, point2)
                re = (point2+point1)/2
    return re

def addpointfor3(pts, imshape):
    pts = np.array(pts)
    imshape = np.array([imshape[0], imshape[1]])
    diff = []
    ptlist = [0, 0, 0, 0]
    diff.append([np.sum(abs(pts[0] - pts[1])), [0, 1]])
    diff.append([np.sum(abs(pts[0] - pts[2])), [0, 2]])
    diff.append([np.sum(abs(pts[2] - pts[1])), [2, 1]])
    ptlist[0],ptlist[2]=diff[np.argmax([ff[0] for ff in diff])][1]
    # # 计算重心
    # center = [0,0]
    # center = np.array(center)
    # # center[0] = imshape[1]/2
    # # center[1] = imshape[0]/2
    # for i in range(3):
    #     current_point = i
    #     min_angle = 181  # 初始化最小夹角为181度
    #     temp=[]
    #     # 计算重心到当前点的向量
    #     vector1 = center - pts[current_point]
    #     for other_point in range(3):
    #         if other_point != current_point:
    #             # 计算重心到已选择点的向量
    #             vector2 = center - pts[other_point]
    #             # 计算两向量之间的夹角
    #             angle = calculate_angle_between_vectors(vector1, vector2)
    #             temp.append(angle)
    #     if (temp[0] > 0) ^ (temp[1] > 0) or (temp[0] < 0) ^ (temp[1] < 0):
    #         ptlist[1]=i
    #         ptlist[2]=(i+1)%3
    #         break
    for i in range(3):
        if i != ptlist[0] and i != ptlist[2]:
            ptlist[1]= i
    # ptlist = [0, 0, 0, 0]
    # # 计算点1与点2的距离
    # distance1_2 = np.linalg.norm(pts[0] - pts[1])
    # # 计算点1与点3的距离
    # distance1_3 = np.linalg.norm(pts[0] - pts[2])
    # # 计算点2与点3的距离
    # distance2_3 = np.linalg.norm(pts[1] - pts[2])
    # # 计算每个点与其他两个点构成的两条线段长度之和
    # length_sum = []
    # length_sum.append(distance1_2 + distance1_3)
    # length_sum.append(distance1_2 + distance2_3)
    # length_sum.append(distance1_3 + distance2_3)
    # ptlist[1] = np.argmin(np.array(length_sum))
    # ptlist[2] = np.argmax(np.array(length_sum))
    # for i in range(3):
    #     if i != ptlist[1] and i != ptlist[2]:
    #         ptlist[0] = i

    dist_12 = np.linalg.norm(pts[ptlist[0]] - pts[ptlist[1]])
    dist_23 = np.linalg.norm(pts[ptlist[1]] - pts[ptlist[2]])
    if dist_12 > dist_23:
        long_edge = dist_12
        short_edge = dist_23
        long_point = ptlist[0]
        short_point = ptlist[2]
    else:
        long_edge = dist_23
        short_edge = dist_12
        long_point = ptlist[2]
        short_point = ptlist[0]
    sig = 1
    stop_flag=False
    while True:
        if sig*(long_edge / 1.5852) / short_edge > sig*(short_edge * 1.5852) / long_edge:
            # p4 = pts[short_point]+(pts[1]-pts[long_point])
            good = long_point
            bad = short_point
        else:
            # p4 = pts[long_point]+(pts[1]-pts[short_point])
            good = short_point
            bad = long_point
        good_angle = abs(calculate_angle_between_vectors(pts[bad] - pts[ptlist[1]], pts[good] - pts[ptlist[1]]))
        di = np.linalg.norm(pts[ptlist[1]] - pts[good])
        xiebian = np.linalg.norm(pts[ptlist[1]] - pts[bad])
        unit_vector = (pts[good] - pts[ptlist[1]]) / di
        if good_angle > 90:
            di_angle = 180 - good_angle
            p4 = pts[bad] + unit_vector * (di + 2 * xiebian * math.cos(di_angle * (math.pi / 180)))
        else:
            di_angle = good_angle
            p4 = pts[bad] + unit_vector * (di - 2 * xiebian * math.cos(di_angle * (math.pi / 180)))
        if (p4[0]>0 and p4[1]>0) or stop_flag:
            break
        else:
            sig = -1
            stop_flag = True
    forgive = imshape * 0.035
    forgive_re = ((p4[0] <= forgive[0] or p4[0] >= imshape[1] - forgive[0]) and abs((p4[0]-pts[bad][0])/(p4[1]-pts[bad][1]))>0.03) or ((p4[1] <= forgive[1] or p4[1] >= imshape[0] - forgive[1]) and abs((p4[1]-pts[bad][1])/(p4[0]-pts[bad][0]))>0.03)
    if forgive_re:
        p4 = jisuanintersection_points(unit_vector, pts[bad], imshape[0], imshape[1])
        p4 = p4[0].tolist()
    return p4


def jisuanintersection_points(unit_v, point_A, height, width):
    # 计算直线与矩形四条边的交点
    intersection_points = []
    # 计算与上边界的交点
    if unit_v[1] < 0:  # 如果直线向上
        t = (0 - point_A[1]) / unit_v[1]
        intersection_points.append(point_A + t * unit_v)
    # 计算与下边界的交点
    if unit_v[1] > 0:  # 如果直线向下
        t = (height - point_A[1]) / unit_v[1]
        intersection_points.append(point_A + t * unit_v)
    # 计算与左边界的交点
    if unit_v[0] < 0:  # 如果直线向左
        t = (0 - point_A[0]) / unit_v[0]
        intersection_points.append(point_A + t * unit_v)
    # 计算与右边界的交点
    if unit_v[0] > 0:  # 如果直线向右
        t = (width - point_A[0]) / unit_v[0]
        intersection_points.append(point_A + t * unit_v)
    # 检查每个交点是否在矩形的范围内
    valid_intersection_points = []
    for point in intersection_points:
        if 0 <= point[0] <= width and 0 <= point[1] <= height:
            valid_intersection_points.append(point)
    del intersection_points
    return valid_intersection_points


def tensor_to_pts_for3(jiao, imshape, boxxyxy):
    # n, m = jiao.shape
    # 提取前四列，分别是左上角 (x1, y1) 和右下角 (x2, y2) 坐标值
    left_top = jiao[:, :2]
    right_bottom = jiao[:, 2:4]

    # 计算左上角和右下角坐标的平均值，得到角点坐标
    pts = (left_top + right_bottom) / 2

    # 将坐标值转换为列表
    pts = pts.tolist()
    if check_points_on_rectangle_edges(pts, boxxyxy[0][0].item(), boxxyxy[0][1].item(), boxxyxy[0][2].item(), boxxyxy[0][3].item()):
        return pts
    pts.append(addpointfor3(pts, imshape))
    pts = order_points1(pts)

    return pts


def tensor_to_pts(jiao):
    n, m = jiao.shape
    assert n == 4 and m >= 4, "Input tensor shape should be (4, m) with m >= 4."

    # 提取前四列，分别是左上角 (x1, y1) 和右下角 (x2, y2) 坐标值
    left_top = jiao[:, :2]
    right_bottom = jiao[:, 2:4]

    # 计算左上角和右下角坐标的平均值，得到角点坐标
    pts = (left_top + right_bottom) / 2

    # 将坐标值转换为列表
    pts = pts.tolist()

    pts = order_points1(pts)

    return pts


def calculate_angle_between_vectors(vector_a, vector_b):
    # 将列表转换为NumPy数组
    vector_a = np.array(vector_a)
    vector_b = np.array(vector_b)

    # 计算点积
    dot_product = np.dot(vector_a, vector_b)

    # 计算向量的模
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # 计算角度（以弧度为单位）
    angle_radians = np.arccos(dot_product / (norm_a * norm_b))

    # 根据向量的方向来确定角度的正负性
    cross_product = np.cross(vector_a, vector_b)
    if cross_product < 0:
        angle_radians = -angle_radians

    # 将弧度转换为度
    angle_degrees = np.degrees(angle_radians)

    return angle_degrees


def order_points1(pts):
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect = np.zeros((4, 2), dtype="float32")
    # 计算重心
    center = np.mean(pts, axis=0)
    result = [np.argmin(s)]

    while len(result) < 4:
        current_point = result[-1]  # 当前的参考点
        min_angle = 181  # 初始化最小夹角为181度
        next_point = None
        # 计算重心到当前点的向量
        vector1 = center - pts[current_point]
        for other_point in range(4):
            if other_point not in result and other_point != current_point:
                # 计算重心到已选择点的向量
                vector2 = center - pts[other_point]
                # 计算两向量之间的夹角
                angle = calculate_angle_between_vectors(vector1, vector2)
                if angle > 0:
                    if angle < min_angle:
                        min_angle = angle
                        next_point = other_point
        # 将下一个点添加到结果列表中
        result.append(next_point)
    for i in range(4):
        rect[i] = pts[result[i]]
    return rect


def order_points(pts):
    pts = np.array(pts)
    rect = np.zeros((4, 2), dtype="float32")
    rememb = [0, 1, 2, 3]
    # 获取左上角和右下角坐标点
    s = pts.sum(axis=1)
    sorted_indices = np.argsort(s)
    rect[0] = pts[sorted_indices[0]]
    len1 = np.linalg.norm(pts[sorted_indices[0]] - pts[sorted_indices[3]])
    len2 = np.linalg.norm(pts[sorted_indices[0]] - pts[sorted_indices[2]])
    if len1 > len2:
        rect[2] = pts[sorted_indices[3]]
        indices_to_remove = [np.argmin(s), sorted_indices[3]]
    else:
        rect[2] = pts[sorted_indices[2]]
        indices_to_remove = [np.argmin(s), sorted_indices[2]]

    rememb = [rememb[i] for i in range(4) if i not in indices_to_remove]

    # 分别计算左上角和右下角的离散差值，无需担心重复选择已选择的点
    diff = np.diff(pts, axis=1)
    if diff[rememb[0]] > diff[rememb[1]]:
        rect[1] = pts[rememb[1]]
        rect[3] = pts[rememb[0]]
    else:
        rect[1] = pts[rememb[0]]
        rect[3] = pts[rememb[1]]

    return rect


def find_optimal_dimension(pts):
    # 指定所需的长宽比例
    desired_aspect_ratio = 1.5852  # 身份证长宽比例
    pts = np.array(pts)
    # 计算四条边的长度
    side_lengths = [np.linalg.norm(pts[(i + 1) % 4] - pts[i]) for i in range(4)]
    # 计算相邻边与对边之间的夹角
    angles = []
    for i in range(2):
        edge1 = pts[i + 1] - pts[i]
        edge2 = pts[(i + 2) % 4] - pts[(i + 3) % 4]
        cos_angle = np.dot(edge1, edge2) / (np.linalg.norm(edge1) * np.linalg.norm(edge2))
        if cos_angle > 1:
            cos_angle = 1
        elif cos_angle < -1:
            cos_angle = -1
        angle = np.arccos(cos_angle)
        angles.append(np.degrees(angle))
    # 找到夹角最小的边索引
    optimal_edge_index = np.argmin(angles)
    # 找到夹角最小的边的对边索引
    opposite_edge_index = (optimal_edge_index + 2) % 4
    # 找到四条边中最长的边的索引
    longest_edge_index = np.argmax(side_lengths)

    # 判断图像是否为正
    zheng = 1
    if not (longest_edge_index == 0 or longest_edge_index == 2):
        zheng = 0
    # 根据条件选择新的宽度和高度
    if optimal_edge_index == longest_edge_index or opposite_edge_index == longest_edge_index:
        new_width = int(side_lengths[longest_edge_index])
        return new_width, int(new_width / desired_aspect_ratio), zheng
    else:
        if side_lengths[optimal_edge_index] > side_lengths[opposite_edge_index]:
            new_height = int(side_lengths[optimal_edge_index])
        else:
            new_height = int(side_lengths[opposite_edge_index])
        return int(new_height * desired_aspect_ratio), new_height, zheng


def four_point_transform(image, pts):
    # 获取坐标点，并将它们分离开来
    # rect = order_points1(pts)
    rect = pts
    rect = np.array(rect, dtype="float32")
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    neww, newh, iszheng = find_optimal_dimension(rect)
    if not iszheng:
        neww, newh = newh, neww

    # 构建新图片的4个坐标点
    dst = np.array([
        [0, 0],
        [neww - 1, 0],
        [neww - 1, newh - 1],
        [0, newh - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)

    warped = cv2.warpPerspective(image, M, (neww, newh))
    if iszheng == 0:
        warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)
    # cv2.imshow('m', cv2.resize(warped, (800, 600)))
    # cv2.waitKey(0)
    # 返回变换后的结果
    return warped

def diandao(img, det, xy):
    shape=[img.shape[1],img.shape[0],img.shape[1],img.shape[0]]
    shape=np.array(shape)
    if 0.95*img.shape[0]>img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    for c in det[:, -1].unique():
        Back = (det[:, -1] == 1).sum()
        Front = (det[:, -1] == 4).sum()
    if xy is not None:
        xywh = (xyxy2xywh(torch.tensor(xy).view(1, 4)) / shape).view(-1).tolist()
    else:
        return img
    for *xyxy, conf, cls in reversed(det):
        #人脸
        if cls == 2 :
            if Back==1 and xywh[0]<0.5:
                img = cv2.rotate(img, cv2.ROTATE_180)
                break
        elif cls == 3:
            if Front==1 and xywh[0]>0.5:
                img = cv2.rotate(img, cv2.ROTATE_180)
                break
    return img