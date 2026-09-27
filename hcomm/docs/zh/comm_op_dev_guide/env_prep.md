# 环境准备

1. 安装驱动固件与CANN软件包。

    HCCL通信算子的开发使用依赖CANN软件包，CANN软件包的详细安装步骤请参考《[CANN 软件安装指南](https://hiascend.com/document/redirect/CannCommunityInstSoftware)》。

2. 设置环境变量。

    进行程序的编译运行前，需要设置CANN软件环境变量。

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

    “/usr/local/Ascend”为CANN软件root用户的默认安装路径，如果使用普通用户安装，或指定路径安装，请自行替换。
