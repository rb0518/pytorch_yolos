# 网络训练 cls loss异常问题排查记录

## 问题现象
在完成按ultralytics版本修改后，发现原始的ultralytics无预训参数情况下，box, cls, dfl三项损失输出如下
```
Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances
2/30      2.69G      3.006      5.745      4.259        164      
```
而C++重构网络训练输出与其有差异，可以看到其中的cls数据异常，其它两项基本在一个数量级
```
  1-30        7        3.62089     8748.63379        4.27526   229
```

## 问题排查步骤记录

### 1 程序异常分析
cls数据异常，而box与dfl正常，训练不收敛，第一步是整个网络错误，首先要排查是否网络有问题
通过引入预训练weight，查看网络predict是否正确。
在ultralytics原代码engine.train._do_train函数中添加如下代码，输出model.train()状态下的weights
```python
        #self.model.export(format='torchscript')
        export_script = False
        if export_script:
            self.model.train()
            test = torch.zeros(1, 3, 640, 640).float()
            test=test.to(self.device)
            trace_module = torch.jit.trace(self.model, test)
            torch.jit.save(trace_module, 'yolo11n.script.pt')
```
可以发现网络训练的三项loss数据与pytorch原代码保持一致
```
  10-30        6        1.24768        1.41472        1.28747   255  
```
对`bus.jpt`预测结果输出正确
```
0 box size: [4, 6] type: float
  0   3 box: [   477.00,   234.50 -   560.00 ,   520.50 ] cls: 0   score: 0.6538
  0   2 box: [   212.75,   240.88 -   284.75 ,   510.00 ] cls: 0   score: 0.6792
  0   1 box: [   109.25,   236.50 -   222.75 ,   536.50 ] cls: 0   score: 0.7808
  0   0 box: [    91.50,   134.25 -   551.00 ,   434.75 ] cls: 5   score: 0.8032
```
 - 初步结论：整个网络代码应该没有问题，不然预测结果应该有偏差，异常最大可能是在`v8DetectionLoss`中计算有问题，或者是网络初始化过程有异常

 ### 2 异常定位

- 思路与准备工作
通过代码通读排查，不容易查找到问题点，最简单的办法是pytorch输出与C++代码输出进行比对。
准备通过中间层数据Tensor替换，查看最终cls loss是否有变化。

- v8DetectionLoss问题排查
在ultralyitcs pytorch代码中在调用v8DetectionLoss计算loss的数据导出：
    - preds, batch['cls']、batch['batch_idx']、batch['bboxes']
    - C++代码在计算前导入上面4个Tensor数据，查看各步输出结果，可以看loss正常，其中的cls 输出为11.5699, 乘上其加权的0.5，最终输出约为5.7，与pytorch的计算基本一致，排除v8DetectionLoss部分代码
```
3 -- bbox_decode over... ret pred_bboxes: [16, 8400, 4]
4 -- assigner over...
cls loss: 11.5699
[ CUDAFloatType{} ]
5 -- bce over...
6 -- box loss over...
```

通过上述检测，可以确认网络部分还有bug，为了定位是那层代码有问题，准备是通过同一个batch数据，分步替换各层的输入，判定那层数据影响了最终cls loss的问题。根据前面分析，网络整体代码是没有问题的，那么就可能是出现在初始化部分，还要用到前面调用torchscript weights的方法，这次只替换测试层的Parameter

- 准备数据
    - 在engine.trainer代码_do_train前添加如下代码
    ```python
        only_onetime = debug_state.get_one_time_state()
        if only_onetime:
            export_test_tensor_tofile("test_img", batch["img"])
            torch.save(batch["img"], "img.pt")

            export_test_tensor_tofile("test_batch_idx", batch["batch_idx"])
            torch.save(batch["batch_idx"], "batch_idx.pt")

            export_test_tensor_tofile("test_cls", batch["cls"])
            torch.save(batch["cls"], "cls.pt")

            export_test_tensor_tofile("test_bboxes", batch["bboxes"])
            torch.save(batch["bboxes"], "bboxes.pt")
        else:
            batch["img"] = torch.load("img.pt")
            batch["batch_idx"] = torch.load("batch_idx.pt")
            cls=batch["cls"] = torch.load("cls.pt")
            bboxes=batch["bboxes"] = torch.load("bboxes.pt")
            
        batch = self.preprocess_batch(batch)
    ```
    得到统一的训练`batch data`,在task._predict_once函数中添加如下代码

    ```python
        x = m(x)  # run

        # 2025-10-30 添加记录每层输出
        debug_state = GlobalDebugManger()
        record_flag = debug_state.get_start_record_state()
        onetime_flag = debug_state.get_one_time_state()
        if m != self.model[-1] and record_flag and onetime_flag:
            save_name = 'ly{}'.format(layerid)
            print(f'save layer {layerid} x as {save_name}.pt {x.shape} {x.dtype} {x.device}')
            export_test_tensor_tofile(save_name, x)
    ```
    得到pytorch代码网络各层的输出 `ly0.pt`......`ly22.pt`。

- 倒序排查
Detec层是重点怀疑对象，这里代码中包含有init_bias的原因。在yolo.cpp forward函数中，将前22层Tesnor调入后，输出结果没有改变
```
   1-30        0        3.69149     7527.13867        4.21041   247 
```
修改`utils.cpp`中`LoadWeightFromJitScript`代码，只调入`Model.23.`的各`Parameter`后，再测试代码
```
  2-30        5        3.76363        6.77920        3.27327   210
```
可以看到，cls loss降低到pytorch输出一个数量级，问题基本上定位到Detection中了。

通过查看比对，torchscript中读入的Parameters与C++代码的Parameters完全一致
```
trans ok : model.23.cv2.0.0.conv.weight
trans ok : model.23.cv2.0.0.bn.weight
trans ok : model.23.cv2.0.0.bn.bias
trans ok : model.23.cv2.0.1.conv.weight
trans ok : model.23.cv2.0.1.bn.weight
trans ok : model.23.cv2.0.1.bn.bias
trans ok : model.23.cv2.0.2.weight
trans ok : model.23.cv2.0.2.bias
trans ok : model.23.cv2.1.0.conv.weight
trans ok : model.23.cv2.1.0.bn.weight
trans ok : model.23.cv2.1.0.bn.bias
trans ok : model.23.cv2.1.1.conv.weight
trans ok : model.23.cv2.1.1.bn.weight
trans ok : model.23.cv2.1.1.bn.bias
trans ok : model.23.cv2.1.2.weight
trans ok : model.23.cv2.1.2.bias
trans ok : model.23.cv2.2.0.conv.weight
trans ok : model.23.cv2.2.0.bn.weight
trans ok : model.23.cv2.2.0.bn.bias
trans ok : model.23.cv2.2.1.conv.weight
trans ok : model.23.cv2.2.1.bn.weight
trans ok : model.23.cv2.2.1.bn.bias
trans ok : model.23.cv2.2.2.weight
trans ok : model.23.cv2.2.2.bias
trans ok : model.23.cv3.0.0.0.conv.weight
trans ok : model.23.cv3.0.0.0.bn.weight
trans ok : model.23.cv3.0.0.0.bn.bias
trans ok : model.23.cv3.0.0.1.conv.weight
trans ok : model.23.cv3.0.0.1.bn.weight
trans ok : model.23.cv3.0.0.1.bn.bias
trans ok : model.23.cv3.0.1.0.conv.weight
trans ok : model.23.cv3.0.1.0.bn.weight
trans ok : model.23.cv3.0.1.0.bn.bias
trans ok : model.23.cv3.0.1.1.conv.weight
trans ok : model.23.cv3.0.1.1.bn.weight
trans ok : model.23.cv3.0.1.1.bn.bias
trans ok : model.23.cv3.0.2.weight
trans ok : model.23.cv3.0.2.bias
trans ok : model.23.cv3.1.0.0.conv.weight
trans ok : model.23.cv3.1.0.0.bn.weight
trans ok : model.23.cv3.1.0.0.bn.bias
trans ok : model.23.cv3.1.0.1.conv.weight
trans ok : model.23.cv3.1.0.1.bn.weight
trans ok : model.23.cv3.1.0.1.bn.bias
trans ok : model.23.cv3.1.1.0.conv.weight
trans ok : model.23.cv3.1.1.0.bn.weight
trans ok : model.23.cv3.1.1.0.bn.bias
trans ok : model.23.cv3.1.1.1.conv.weight
trans ok : model.23.cv3.1.1.1.bn.weight
trans ok : model.23.cv3.1.1.1.bn.bias
trans ok : model.23.cv3.1.2.weight
trans ok : model.23.cv3.1.2.bias
trans ok : model.23.cv3.2.0.0.conv.weight
trans ok : model.23.cv3.2.0.0.bn.weight
trans ok : model.23.cv3.2.0.0.bn.bias
trans ok : model.23.cv3.2.0.1.conv.weight
trans ok : model.23.cv3.2.0.1.bn.weight
trans ok : model.23.cv3.2.0.1.bn.bias
trans ok : model.23.cv3.2.1.0.conv.weight
trans ok : model.23.cv3.2.1.0.bn.weight
trans ok : model.23.cv3.2.1.0.bn.bias
trans ok : model.23.cv3.2.1.1.conv.weight
trans ok : model.23.cv3.2.1.1.bn.weight
trans ok : model.23.cv3.2.1.1.bn.bias
trans ok : model.23.cv3.2.2.weight
trans ok : model.23.cv3.2.2.bias
trans ok : model.23.dfl.conv.weight
```
    - 判定是否是初始化计算有差异，打印查看pytorch,初始化时传入的三次数值如下：
    ```
    8.0 80 fill_v: -11.536642013939286
    16.0 80 fill_v: -10.150347652819397
    32.0 80 fill_v: -8.764053291699506
    ```
原代码中a对应的cv2, b对应cv3，其中cv3对应着的是cls
```python
    a[-1].bias.data[:] = 1.0  # box
    b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640 / s) ** 2)  # cls (.01 objects, 80 classes, 640 img)
    fill_v = math.log(5 / m.nc / (640 / s) ** 2)
```
同时排查出cv2的初始化值是0.f，但对应着的是box，对结果影响不大，cv3计算结果为
```
s: 8 nc: 80 fill_val: -11.5366
s: 16 nc: 80 fill_val: -10.1503
s: 32 nc: 80 fill_val: -8.76405
```
可以看到，pytorch中这里是double，而C++用了float，修改函数参数类型与log_val计算相关变量，全部更换为double类型

```
   1-30        0        3.59484        5.82185        4.20135   256 
```
将类型修改回float，问题复现，证明Bug点排查正确。