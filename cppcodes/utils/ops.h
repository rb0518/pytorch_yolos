#pragma once

#include <torch/torch.h>
#include <tuple>

torch::Tensor scale_boxes(std::tuple<int, int> img1_shape, torch::Tensor boxes, std::tuple<int, int> img0_shape, 
	std::tuple<std::tuple<float, float>, std::tuple<int, int>> ratio_pad,
	bool padding, bool xywh);

torch::Tensor clip_boxes(torch::Tensor boxes, std::tuple<int, int> shape);

// 函数名定义说明： 函数名中带n，对应的是在图中的坐标
//Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height) format where (x1, y1) is the
//top-left corner and (x2, y2) is the bottom-right corner.
//Args:
//    x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
//Returns:
//    y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height) format.
torch::Tensor xyxy2xywh(const torch::Tensor& x);


//Convert bounding box coordinates from (x, y, width, height) format to (x1, y1, x2, y2) format where (x1, y1) is the
//top-left corner and (x2, y2) is the bottom-right corner. Note: ops per 2 channels faster than per channel.
//Args:
//    x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x, y, width, height) format.
//Returns:
//    y (np.ndarray | torch.Tensor): The bounding box coordinates in (x1, y1, x2, y2) format.
torch::Tensor xywh2xyxy(const torch::Tensor& x);


//Convert normalized bounding box coordinates to pixel coordinates.
//Args:
//    x (np.ndarray | torch.Tensor): The bounding box coordinates.
//    w (int): Width of the image. Defaults to 640
//    h (int): Height of the image. Defaults to 640
//    padw (int): Padding width. Defaults to 0
//    padh (int): Padding height. Defaults to 0
//Returns:
//    y (np.ndarray | torch.Tensor): The coordinates of the bounding box in the format [x1, y1, x2, y2] where
//        x1,y1 is the top-left corner, x2,y2 is the bottom-right corner of the bounding box.
torch::Tensor xywhn2xyxy(const torch::Tensor& x, int w = 640, int h = 640, int padw = 0, int padh = 0);


//Convert bounding box coordinates from (x1, y1, x2, y2) format to (x, y, width, height, normalized) format. x, y,
//width and height are normalized to image dimensions.
//Args:
//    x (np.ndarray | torch.Tensor): The input bounding box coordinates in (x1, y1, x2, y2) format.
//    w (int): The width of the image. Defaults to 640
//    h (int): The height of the image. Defaults to 640
//    clip (bool): If True, the boxes will be clipped to the image boundaries. Defaults to False
//    eps (float): The minimum value of the box0x00s width and height. Defaults to 0.0
//Returns:
//    y (np.ndarray | torch.Tensor): The bounding box coordinates in (x, y, width, height, normalized) format
torch::Tensor xyxyn2xywh(const torch::Tensor& x, int w = 640, int h = 640, bool clip = false, float eps = 0.0);

torch::Tensor xyn2xy(const torch::Tensor& x, int w = 640, int h = 640, int padw = 0, int padh = 0);
