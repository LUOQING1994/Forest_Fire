# Forest_Fire
森林火灾项目分为两个部分，本部分为模型效果展示。既实用flask搭建一个简易的服务器，web端访问相应接口，程序将读取本地视频和模型，并对每一帧进行检测。分析完毕后，保存检测结果为视频
# 环境：pytorch 1.6.0，python 3.6，Python-opencv 3.x，flask 1.12
# 简介
本部分功能为森林火灾模型测试结果展示，前端访问flask服务器------> 程序读取本地视频和模型 -----> 分析结果实时存到视频中
### 备注：读者可使用flask写一个前端视频文件上传接口，替代程序读取本地的视频，把分析后的视频地址返回前端进行视频结果展示
