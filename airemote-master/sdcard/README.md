# AIRemote 系统

系统已完成了 Atlas200 DK 以及 Atlas200I DK A2 两代开发板端全部环境安装部署，主要特点是系统映像体积较小，除支持开发板运行官方推理案例外，还直接支持远程推理服务和分布式集群接入，无需额外配置即可满足基本实践要求。

本映像系统的固件驱动版本和 CANN 版本分别为  5.0.5（Atlas200 DK） 和 6.2.RC1（Atlas200I DK A2），经过 5 次课程实践优化，累计超过 280 人成功使用。

### TF卡制作

通过映像文件压缩包直接写卡即可。下载映像文件压缩文件包，写卡过程约为8分钟。

#### 下载文件包

下载后无需解压缩，直接写卡即可，百度网盘下载地址：

https://pan.baidu.com/s/1P9WmPw1z8SaApA0QHl1hzg?pwd=1sjk

其中 **Atlas200DK** 文件夹下为第一代开发板的镜像文件及对应校验文件。推荐下载：**air-A1-v2.img.xz**

其中 **AtlasI_200_DK_A2** 文件夹下为第二代开发板正式版和测试版的镜像文件及对应校验文件。

- 正式版：**AIR-CANN6-v2.0.2.img.xz**
- 测试版：**AIR-CANN6-v2.0.1-测试板.img.xz**

下载后请计算对应 img.xz 文件的 MD5sum，确认文件未受损。

#### 用工具写卡

推荐写卡工具 Raspberry Pi Imager或Etcher，支持 GNU Linux/Windows/MacOS 操作系统，下载链接附后。

运行该工具，选择下载的 img.xz 文件，写入对应的 TF 卡设备即可。

【提示】自动扩容：在开发板上启动后会自动扩展文件系统至 TF 卡的全部空间。

### 访问开发板

#### 第一代开发板（Atlas 200 DK）

可参考昇腾官方发布的启动和访问方式，通过 USB-C 或 网线 连接开发板，IP地址分别为 192.168.2.123 和 192.168.1.111

登录用户名有 root, HwHiAiUser，

密码均为 Mind@123，建议使用 HwHiAiUser 用户登录测试使用。如需 root 权限，请登录 HuHiAiUser 并通过 sudo 取得。

#### 第二代开发板（Atlas 200I DK A2）

可参考昇腾官方发布的启动和访问方式，通过 USB-C 或 网线 连接开发板，IP地址分别为 192.168.0.2 和 192.168.137.100（eth1）

登录用户名有 root 和 HwHiAiUser，

密码均为 Mind@123，建议使用 HwHiAiUser 用户登录测试使用。如需root权限，请登录 HuHiAiUser 并通过 sudo 取得。

### 远程推理应用实践

只需安装远程推理Python工具包，即可在自己的电脑上直接进行开发板本机测试和远程测试，测试demo如百度网盘中的mp4，具体请参考:
https://gitee.com/haojiash/airemote

### Raspberry Pi Imager 下载

Linux: https://downloads.raspberrypi.org/imager/imager_latest_amd64.deb

Windows:https://downloads.raspberrypi.org/imager/imager_latest.exe

MacOS: https://downloads.raspberrypi.org/imager/imager_latest.dmg

### Etcher工具下载

官方链接：https://www.balena.io/etcher/

### 建议反馈

请提交issue。
