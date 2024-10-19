# AiRemote：Atlas200DK Inference Remote 昇腾AI应用开发：如此简单！

## 1. 项目动机与特性
昇腾Atlas200DK为AI开发者带来了全新的选择，但对于新手存在上手慢、展现差，对教学存在资源短缺等困难。本项目基于将模型推理与模型应用相分离的机制，将Atlas200DK开发者套件封装为AI推理的黑盒服务，构建了分布式远程推理框架，并提供了多种输入模态、多种输出方式以及多线程支持的高度复用框架，解决了开发板资源稀缺课程受限、环境配置复杂上手困难、缺乏可视化体验不好等痛点问题。141人选课的课程实践表明，该远程推理平台可明显提高深度学习模型推理应用的教学实践效果。

该项目可极大提升了开发板利用效率和推理应用的用户体验，主要特点有：

1. 极大简化开发环境，支持跨平台开发，上手非常简单，可快速专注于深度神经网络模型的推理应用实践；
2. 分布式运行：输入前处理、后处理及GUI界面等通用计算运行于本地的CPU，模型推理神经网络计算运行于远程的NPU，支持复杂创新应用实践；
3. 框架基于抽象和封装设计，高度复用易于扩展，可基于此示例快速开发其他远程模型和推理应用；
4. 单块开发板可提供单点远程推理服务，实现一板对多人的复用；多块开发板可提供集群服务，实现多板对多人的复用；不受时空限制，充分发挥昇腾算力。

### 远程推理应用的测试DEMO

[实时视频风格迁移](https://v.youku.com/v_show/id_XNTg2MTk0OTY0OA==.html)

[Yolov4交通场景物体检测与识别](https://v.youku.com/v_show/id_XNTg2MTI4Njk3Mg==.html)

### 重要升级特性
当前版本为 2.0.2，结合新版开发者套件测试版已完成重要升级，新版统一框架主要特性有：

1. AirLoader模型加载器启动om模型服务，同时支持Python和C++远程推理应用，开发者不再必须关注ACL接口；
2. AtlasApplet组件支持灵活的插件机制，C++与Python可混合运行，支持弹性部署，运行性能更高；
3. AiremoteTest组件为Python和C++提供统一的测试方式，更为简洁、灵活；
4. Python版本案例更为简洁，不再必须提供 _model.py 文件来启动模型服务，进一步简化入门理解难度;
5. C++与Python应用开发风格高度一致，开发效率大幅提升。


![Atlas200DK远程推理集群](airemote.jpg)


### 相关课程
- [“智能基座”2022年第四期:人工智能-昇腾CANN应用的远程推理与异构开发实践](https://edu.hicomputing.huawei.com/education/training/1546385869698068482/course)

- [基于Atlas200DK的远程推理集群平台搭建实验](https://www.hiascend.com/developer/courses/detail/1587270014125752321)

- [基于昇腾AI的远程推理应用与开发 （Atlas200I DK A2)](https://www.hiascend.com/developer/courses/detail/1640887402266980354)


## 2. 推理应用开发流程简介

### 使用方式

  - 开发板上用airloader工具一键启动OM模型的推理服务

  - PC端用Python或C++进行推理应用开发、测试和运行

  - 推理应用可灵活部署于PC端或开发板

### 极简环境准备

  分为两端，PC端为应用开发环境，开发板端为离线模型（OM）的ACL运行环境，

  - PC端：用于推理应用的开发、测试、运行，与ACL完全解耦，特点：仅依赖
    Python，配置极为简单。

  - 开发板端：用于OM离线模型的推理，封装ACL开发全部要素，特点：将模型封
    装为黑盒，通过网络提供推理服务。

  本文档仅针对PC端中的推理应用运行环境配置，是学生开发推理应用仅需依赖的环境。开发板端可由教师或助教来维护，构建方式见[Atlas200DK合设迷你镜像1.6GB](https://gitee.com/haojiash/airemote/tree/master/sdcard).

#### PC端环境准备

  pip 一键安装即完成，支持Windows/MacOS/Linux 跨平台开发，学生只需准备此环境即可。

  演示一种平台：下载、安装，python 命令打印包版本号验证安装成功。

  `python -c 'import inferemote;print(inferemote.__version__)'`


#### 开发板端环境准备

  超小镜像制卡即完成，支持两代开发板，可由教师或助教准备。

  AiRemote超小镜像系统：
  - 已配置AiRemote软件包
  - 自动启动示例模型和Applet服务
  - 支持OpenCV的C++开发环境。

  演示流程：下载、写卡、网线连接、登陆、查看已启动服务。

  此外，亦可在现有系统中自行配置，主要任务：安装AiRemote C++软件包；安装
  AiRemote Python软件包；启动示例模型和Applet服务，另见手册文档。


## 3. PC端开发（运行）环境配置

学生只需配置PC端环境，即可开展推理应用开发、测试和运行。本工具包主要内容有：

- 提供一个远程推理模型的封装包，通过pip install安装即可运行应用示例；
- 提供若干推理应用的示例，包括 Python（含jupyter-notebook）和 C++ 示例，代码简洁，专注模型前后处理和应用，无需分别处理图片和视频等不同输入数据；
- 通过命令行参数支持多源测试方式，包括支持图片文件、目录、视频文件或摄像头采集测试，无需修改代码；
- 通过命令行参数支持多种结果展现方式，包括图片对比显示、在线web视频流显示，以及多线程快速测试，无需修改代码。

当前支持 Windows（64位）, GNU/Linux（64位）和 MacOS 10.15 (Intel)系统的 Python 3.9.x 环境，可在通用PC和树莓派等嵌入式系统上使用Atlas200DK的远程推理服务。

### 3.1. Windows 安装配置说明

#### (1) 安装 Python
以下两种方式二选一即可，推荐第一种。

##### 通过 MiniConda 安装

参考网址（https://docs.conda.io/en/latest/miniconda.html）。

Python version	Name	Size	SHA256 hash

Python 3.9	[点此下载](https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Windows-x86_64.exe) Windows 64-bit	58.1 MiB	b33797064593ab2229a0135dc69001bea05cb56a20c2f243b1231213642e260a

##### 直接通过安装包安装
Windows 安装包下载地址：https://www.python.org/ftp/python/ 下载。
执行安装文件，注意第一步中选择 注册环境变量。

#### (2) 安装 AiRemote 项目软件包
将本项目下载到本地目录，如：
    D:\airemote

启动 cmd.exe，执行以下命令：

python -V

确认python版本为3.9.x，执行以下相应命令：

`pip3.9 install D:\airemote\lib\python3\inferemote-2.0.2-py39-none-win_amd64.whl`

安装成功后，执行如下命令应输出 “2.0.2”。

`python3.9 -c 'import inferemote;print(inferemote.__version__)`


### 3.2. GNU/Linux 安装配置说明
适用于 Ubuntu/CentOS/openEuler 等常见操作系统，以及树莓派平台。

#### (1) 安装 Python

以下两种方式二选一即可，推荐第一种。

##### 通过 MiniConda 安装
Python version	Name	Size	SHA256 hash

Python 3.9	[点此下载](https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh) Linux 64-bit	63.6 MiB	1ea2f885b4dbc3098662845560bc64271eb17085387a70c2ba3f29fff6f8d52f

Python 3.7 	[点此下载](https://repo.anaconda.com/miniconda/Miniconda3-py37_4.10.3-Linux-aarch64.sh) Linux-aarch64 64-bit	89.2 MiB	65f400a906e3132ddbba35a38d619478be77d32210a2acab05133d92ba08f111

##### 直接通过安装包安装
Linux 安装包下载地址：https://www.python.org/ftp/python/ 下载安装。
按文档说明进行编译安装，添加路径环境变量。


#### (2) 安装AIRemote 项目软件包

将本项目下载到本地目录，如：
    ~/airemote

进入shell，执行以下命令：

python -V

根据版本，执行以下相应命令：

`pip3.9 install ~/airemote/lib/python3/inferemote-2.0.2-py39-none-linux_x86_64.whl`

`python3.9 -c 'import inferemote;print(inferemote.__version__)`

### 3.3. MacOS 安装配置说明
请参照上述 GNU/Linux 的安装方式。

`pip3.9 install ~/airemote/lib/python3/inferemote-2.0.1-py39-none-macosx_10_15_universal.whl`

`python3.9 -c 'import inferemote;print(inferemote.__version__)`

## 4. 使用说明
1. 务必确认 Python 的版本与安装包是对应的；
2. 目前仅限课堂实践使用，所涉及 开发板或集群服务器的 IP 根据教学需要提供。

以下为 Windows 平台运行示例。

#### 进入本项目主目录
cd D:\airemote\python

#### 转换摄像头采集的图片，可执行以下命令：
`python styletransfer\test.py -r tcp://IP:port -m show -w 3 -I camera`

#### 运行采用毕加索模型的图像风格转换，使用指定图片：
`python styletransfer\test.py -r tcp://IP:port -m show -w 3 -i D:\test.jpg`

#### 转换指定目录下的全部图片，如：D:\test_pictures，可执行以下命令：
`python styletransfer\test.py -r tcp://IP:port -m show -w 3 -i D:\test_pictures`

#### 执行如下命令，可显示参数帮助信息，请自行探索更多功能！
`python styletransfer\test.py -h`

### 4. 开发板和集群系统构建
如需制作支持以上远程推理的开发板系统，请参考：
https://gitee.com/haojiash/airemote/tree/master/sdcard

### 5. 反馈建议
欢迎提交issue，并参与更多实践案例移植和分享。
