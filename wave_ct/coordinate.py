"""
WaveCT统一坐标转换模块

约定：
反演模型坐标：
    X_model
    Y_model

CAD显示坐标：
    X_cad
    Y_cad

关系：

X_cad = X_model - x_offset
Y_cad = Y_model - y_offset

"""

from __future__ import annotations

import numpy as np


def model_to_cad(
    x,
    y,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
):
    """
    反演坐标 -> CAD绘图坐标
    """

    return (
        np.asarray(x) - x_offset,
        np.asarray(y) - y_offset,
    )


def cad_to_model(
    x,
    y,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
):
    """
    CAD坐标 -> 反演坐标
    """

    return (
        np.asarray(x) + x_offset,
        np.asarray(y) + y_offset,
    )


def transform_points(
    points,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
):
    """
    点集转换

    输入:
        [[x,y],
         [x,y]]

    输出:
        CAD坐标
    """

    pts = np.asarray(points, dtype=float)

    result = pts.copy()

    result[:,0] -= x_offset
    result[:,1] -= y_offset

    return result