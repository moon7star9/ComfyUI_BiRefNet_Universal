## 简介 | Introduction  

ComfyUI BiRefNet Cosmos 为您提供了一套完整的BiRefNet模型调用解决方案。该项目将所有BiRefNet系列模型封装为易用的ComfyUI节点，支持一键调用各类模型进行抠图处理，方便一键下载模型或手动下载模型。

## 特性 | Features  

- 🚀 支持全系列BiRefNet模型，包括通用模型、高分辨率模型和轻量级模型等，支持所有BiRefNet的huggingface模型库

- 💫 自动模型管理，支持本地加载与在线下载

  可以直接下载到models\BiRefNet\下，如图所示：
  
  ![image](https://github.com/user-attachments/assets/db8f1992-12f6-4d8a-8887-076226e417f5)

  也可以选择运行的时候自动下载

- 🎯 针对不同场景优化的模型选项，自动选择模型推理时的最佳抠图尺寸 
  - BiRefNet (通用场景 | General Purpose)  
  - BiRefNet_HR (高分辨率 | High Resolution)  
  - BiRefNet_lite (轻量级 | Lightweight)  
  - BiRefNet-matting (抠图 | Image Matting)  
  - BiRefNet-portrait (人像 | Portrait)  
  - 稍后详细补充

模型库见：https://huggingface.co/ZhengPeng7

## 感谢

感谢BiRefNet仓库的所有作者开源的代码和模型 [ZhengPeng7/BiRefNet](https://github.com/zhengpeng7/birefnet)
