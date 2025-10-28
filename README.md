# 学习YOLOs模型用

## 目标
目标是消化Github上的[YOLO air](https://github.com/iscyy/yoloair)和[YOLOv5](https://github.com/ultralytics/yolov5)中的代码，裁剪掉不是必要的代码，简化代码文件，并用libtorch重构代码完成一步步迭代YOLO版本，跟着`YOLO air`收集的文档，慢慢掌握相关知识的掌握，有空再通过libtorch+OpenCV完成对应代码功能

** 个人游戏之作，不建议学习的人用libtorch来进行代码重构，一是学习的效率较python低很多，每一次代码修改均需要进行编译，二是libtorch中缺少很多pytorch的库，如`LambdaLR`, `Amp.GradeScaler`,还有Numpy等工具，三是网上资源很少。

## 测试数据
`COCO`数据集和自建数据集，自行下载，约定`pytorch_yolos`为根目录，并修改data中的`coco128.yaml`和`coco.yaml`中与其相对路径

## 目录说明
所有的目录布局如下
```
 directory_root | pytorch_yolos
 - cfgs  从yolov5转存的opt.yaml和hyp.scratch.yaml
 - cppsegmentor libtorch c++代码根目录
  -external/yaml-cpp
 -models 从yolov5转存的 model cfgs文件如yolov5s.yaml
 -data 自行拷贝yolov5下data目录内容到这个目录下
 -pycodes 
 -weigths 可以将jit.script.pt文件放在这个目录下
 ```
## Requirements
 * OpenCV 4.11.1
 * Pytorch 2.7.1
 * Libtorch 2.7.1
 * CUDA 12.6
 * Visual studio 2022
 * Windows 11 & WSL Ubuntu24.04
 * Python 3.12.3 

## Architectures
测试硬件环境为
* CPU Intel(R) Xeon(R) CPU E5-2670 0 @ 2.60GHz
* NVIDIA Geforece RTX 2080 8G
  
## Current Support CMake
 - [x] Windows
 - [x] Linux

## 编译

- linux，安装好cuda12.6、opencv和libtorch
```
cd pytorch_yolos\cppcodes
mkdir build && cd build
cmake ..
make -j4
./yolos
```
- Windows,修改cppcodes/CMakeLists.txt文件，确保libtorch和OpenCV库指向自己的环境
```
cd pytorch_yolos\cppcodes
mkdir build_win && cd build_win
cmake ..
```
用Visual Studio 2022打开build_win\LibtorchCPP.sln，编译后执行或命令行执行，注意所有程序是以pytorch_yolos为要目录，Windows下执行文件是在Release下，相对路径要多加一级

## 训练标准流程
以下是YOLOv5训练的完整流程，结合模型架构、数据准备、训练优化等关键环节：

* Scratch训练 
`./yolos`
或
`./yolos --epochs=300 --data=data/coco128.yaml`

* pretrain
```
./yolos --epochs=300 --resume=runs/train/exp2/weigths/last.pt
```
或者`--resume`不为空，自己去找最新的`.pt`文件
```
./yolos --epochs=300 --resume=X
```

程序提供了从jit.script文件中读取named_parameters的方法，需要利用yolov5中的expert.py文件将pt文件转换为torchscript输出
```
./yolos --jit_weights=weights/yolov5s.script.pt
```

* 运行结果
以调用jit_weights为例，epochs 30次后输出结果如下
<img src =".\readme_images\run_sample.jpg">

## Segment
训练，目前mosaic方式可能还因为丢失box与mask不一致，程序出错，先用--rect参数来
```bash
./yolos --batch_size=8 --epochs=20 --jit_weights=weights/yolov5s-seg.script.pt  --img_size=640  --rect --notest=false
```

验证
```bash
./yolos --runtype=predict --is_segment=true --cfg=models/segment/yolov5s-seg.yaml --weights=runs/train_seg/exp1/weights/last.pt
```

* 2025-10-28 
  根据`ultralytics`代码，增加代码，能够解释v5, v11, v12版本的yaml文件，但v8DetectionLoss中cls损失函数异常，比pytorch原代码大了约2000倍，训练不收敛，检查了代码，暂时未发现问题所在。尝试调用yolov11n.pt(自己在原代码中添加，在model.train()后调用jit.trace()函数转换，不要在eval下转换)，模型训练收敛，同时预测输出正确。要不在init_weight等地方还有丢失步骤，要不v8Detection中还有错误未发现
