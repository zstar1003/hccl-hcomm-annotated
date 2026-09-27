# 编译部署

开发者完成通信算子开发之后，需部署到运行环境上进行功能验证。

## 了解自定义算子编译打包工程

HCCL代码仓提供了自定义算子编译打包工程，开发者可参考[自定义Send/Receive通信算子样例](https://gitcode.com/cann/hccl/tree/master/examples/04_custom_ops_p2p)进行开发，其目录结构如下：

```text
├── build.sh                                # HCCL代码仓根目录编译工程入口
├── CMakeLists.txt                          # HCCL代码仓根目录编译/构建配置文件
├── cmake/                                  # CMake函数
│   ├── config.cmake
│   ├── func.cmake
│   ├── package.cmake
│   └── makeself_custom.cmake
├── scripts/
│   ├── custom/
│   │   └── install.sh                     # 自定义算子包安装脚本
│   └── sign/
│       └── add_header_sign.py              # AI CPU算子包签名脚本
└── examples/04_custom_ops_p2p               # 自定义算子工程目录
    ├── CMakeLists.txt                       # 自定义算子编译/构建配置文件
    ├── op_host/
    │   ├── send.cc                         # HcclSendCustom算子实现源文件
    │   ├── recv.cc                         # HcclRecvCustom算子实现源文件
    │   ├── load_kernel.cc                  # AI CPU Kernel在Host侧的加载逻辑
    │   └── launch_kernel.cc                # AI CPU Kernel在Host侧的下发逻辑
    ├── op_kernel_aicpu/
    │   ├── libp2p_aicpu_kernel.json        # AI CPU Kernel算子描述文件
    │   ├── aicpu_kernel.cc                 # AI CPU Kernel实现逻辑
    │   └── exec_op.cc                      # AI CPU算子编排逻辑
    ├── inc/
    │   ├── hccl_custom_p2p.h               # 自定义Send/Receive算子接口头文件
    │   ├── common.h                        # 公共类型头文件
    │   └── log.h                           # 日志宏定义
    ├── scripts/
    │   └── hccl_custom_p2p_check_cfg.xml   # 签名配置文件
    └── testcase/
        ├── main.cc                          # 测试样例源文件
        └── Makefile                         # 编译/构建配置文件
```

建议开发者基于上述目录结构存放代码文件，需重点关注如下内容：

1. 自定义算子头文件放在inc文件夹。
2. Host侧实现代码放在op_host文件夹，AI CPU侧实现代码放在op_kernel_aicpu文件夹。
3. CMakeLists.txt内容根据实际需要进行调整。

## 编译自定义算子包

1. 设置CANN软件环境变量。

    ```bash
    source /usr/local/Ascend/cann/set_env.sh
    ```

    “/usr/local/Ascend”为CANN软件root用户的默认安装路径，如果使用普通用户安装，或指定路径安装，请自行替换。

2. 下载HCCL代码仓。

    ```bash
    git clone https://gitcode.com/cann/hccl.git
    ```

3. 编译自定义算子包。

    ```bash
    bash build.sh --vendor=<vendor> --ops=<ops> --custom_ops_path=<ops_project_path>
    ```

    其中：

    - <vendor\>：自定义算子包标识信息，用户自定义，保持唯一。
    - <ops\>：自定义算子名称，用户自定义，保持唯一。
    - <ops_project_path\>：自定义算子工程根目录，例如[了解自定义算子编译打包工程](#了解自定义算子编译打包工程)中的./examples/04_custom_ops_p2p目录，可配置为绝对路径或相对路径。

    编译完成后，会在当前目录的build_out目录下生成自定义算子安装包cann-hccl_custom_<ops\>_linux-<arch\>.run，其中：

    - <ops\>：表示编译自定义算子包时通过--ops参数指定的算子名称。
    - <arch\>：表示当前编译环境的系统架构。

## 部署自定义算子包

执行下列命令进行安装：

```bash
./build_out/cann-hccl_custom_<ops>_linux-<arch>.run --install
```

自定义算子包安装信息如下：

- 头文件：$\{ASCEND_HOME_PATH\}/opp/vendors/<vendor\>/include
- 动态库：$\{ASCEND_HOME_PATH\}/opp/vendors/<vendor\>/lib64
- AI CPU算子信息库文件：$\{ASCEND_HOME_PATH\}/opp/vendors/<vendor\>/aicpu/config
- AI CPU算子包：$\{ASCEND_HOME_PATH\}/opp/vendors/<vendor\>/aicpu/kernel
- 安装脚本：$\{ASCEND_HOME_PATH\}/opp/vendors/<vendor\>/scripts/install.sh

其中<vendor\>为编译自定义算子包时通过--vendor参数指定的算子标识。
