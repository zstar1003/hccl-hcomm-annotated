# 基于NPU环境测试

开发者需基于自定义通信算子的功能，自行编写测试用例进行验证，主要包括输入数据的构造和语义逻辑的检查。下面简要介绍通信算子的测试方法。

## 编写测试程序

测试程序主要包含以下六个步骤：

1. 申请集合通信操作所需的内存。
2. 构造输入数据。
3. 初始化通信域。
4. 调用自定义通信算子。
5. 校验通信结果。
6. 释放资源。

测试程序代码样例如下所示（开发者须按需修改相关逻辑，下列样例代码不可直接执行）：

```c
#include "acl/acl_rt.h"
#include "hccl/hccl_types.h"
#include "<custom_ops_header>"    // 开发者定义的通信算子接口头文件

#define ACLCHECK(ret)                                                                          \
    do {                                                                                       \
        if (ret != ACL_SUCCESS) {                                                              \
            printf("acl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                        \
        }                                                                                      \
    } while (0)

#define HCCLCHECK(ret)                                                                          \
    do {                                                                                        \
        if (ret != HCCL_SUCCESS) {                                                              \
            printf("hccl interface return err %s:%d, retcode: %d \n", __FILE__, __LINE__, ret); \
            return ret;                                                                         \
        }                                                                                       \
    } while (0)

struct ThreadContext {
    HcclRootInfo *rootInfo;
    uint32_t device;
    uint32_t devCount;
};

int Sample(void *arg)
{
    ThreadContext *ctx = (ThreadContext *)arg;
    void *sendBuf = nullptr;
    void *recvBuf = nullptr;
    uint32_t device = ctx->device;
    uint64_t count = ctx->devCount;
    size_t mallocSize = count * sizeof(float);

    // 设置当前线程操作的设备
    ACLCHECK(aclrtSetDevice(static_cast<int32_t>(device)));

    // 初始化集合通信域
    HcclComm hcclComm;
    HCCLCHECK(HcclCommInitRootInfo(ctx->devCount, ctx->rootInfo, device, &hcclComm));

    // 创建任务流
    aclrtStream stream;
    ACLCHECK(aclrtCreateStream(&stream));

    // 构造输入数据
    void *hostBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&hostBuf, mallocSize));
    // TODO：将输入数据写入hostBuf中

    // 将输入数据拷贝至Device侧
    ACLCHECK(aclrtMalloc(&sendBuf, mallocSize, ACL_MEM_MALLOC_HUGE_ONLY));
    ACLCHECK(aclrtMemcpy(sendBuf, mallocSize, hostBuf, mallocSize, ACL_MEMCPY_HOST_TO_DEVICE));
    ACLCHECK(aclrtFreeHost(hostBuf));

    // TODO：调用自定义集合通信算子
    HCCLCHECK(HcclOpsCustom(sendBuf, count, HCCL_DATA_TYPE_FP32, hcclComm, stream));
 
    // 阻塞等待任务流中的集合通信任务执行完成
    ACLCHECK(aclrtSynchronizeStream(stream));

    // 将Device侧集合通信任务结果拷贝到Host
    void *resultHostBuf;
    ACLCHECK(aclrtMallocHost(&resultHostBuf, mallocSize));
    ACLCHECK(aclrtMemcpy(resultHostBuf, mallocSize, recvBuf, mallocSize, ACL_MEMCPY_DEVICE_TO_HOST));
    // TODO：校验resultHostBuf中的结果数据
    ACLCHECK(aclrtFreeHost(resultHostBuf));

    // 释放资源
    HCCLCHECK(HcclCommDestroy(hcclComm));  // 销毁通信域
    if (sendBuf) {
        ACLCHECK(aclrtFree(sendBuf));      // 释放Device侧内存
    }
    if (recvBuf) {
        ACLCHECK(aclrtFree(recvBuf));      // 释放Device侧内存
    }
    ACLCHECK(aclrtDestroyStream(stream));  // 销毁任务流
    ACLCHECK(aclrtResetDevice(device));    // 重置设备
    return 0;
}

int main()
{
    // 设备资源初始化
    ACLCHECK(aclInit(NULL));
    // 查询设备数量
    uint32_t devCount;
    ACLCHECK(aclrtGetDeviceCount(&devCount));
    std::cout << "Found " << devCount << " NPU device(s) available" << std::endl;

    int32_t rootRank = 0;
    ACLCHECK(aclrtSetDevice(rootRank));
    // 生成Root节点信息，各线程使用同一份RootInfo
    void *rootInfoBuf = nullptr;
    ACLCHECK(aclrtMallocHost(&rootInfoBuf, sizeof(HcclRootInfo)));
    HcclRootInfo *rootInfo = (HcclRootInfo *)rootInfoBuf;
    HCCLCHECK(HcclGetRootInfo(rootInfo));

    // 启动线程执行集合通信操作
    std::vector<std::thread> threads(devCount);
    std::vector<ThreadContext> args(devCount);
    for (uint32_t i = 0; i < devCount; i++) {
        args[i].rootInfo = rootInfo;
        args[i].device = i;
        args[i].devCount = devCount;
        threads[i] = std::thread(Sample, (void *)&args[i]);
    }
    for (uint32_t i = 0; i < devCount; i++) {
        threads[i].join();
    }

    // 释放资源
    ACLCHECK(aclrtFreeHost(rootInfoBuf));  // 释放Host内存
    ACLCHECK(aclFinalize());               // 设备去初始化
    return 0;
}
```

## 编写Makefile

Makefile文件示例如下，开发者须按需修改。

```text
ifndef ASCEND_HOME_PATH
    $(error "ASCEND_HOME_PATH is not set, please ensure CANN is properly installed and \
             source environment variables by running `source /path/to/Ascend/cann/set_env.sh`")
endif

CXXFLAGS := -std=c++17 \
        -Werror \
        -fstack-protector-strong \
        -fPIE -pie \
        -O2 \
        -s \
        -Wl,-z,relro \
        -Wl,-z,now \
        -Wl,-z,noexecstack \
        -Wl,--copy-dt-needed-entries

SOURCES = main.cc
ASCEND_INC_DIR = ${ASCEND_HOME_PATH}/include
ASCEND_LIB_DIR = ${ASCEND_HOME_PATH}/lib64
CUSTOM_OPS_INC_DIR = ${ASCEND_HOME_PATH}/opp/vendors/<vendor>/include
CUSTOM_OPS_LIB_DIR = ${ASCEND_HOME_PATH}/opp/vendors/<vendor>/lib64
LIBS = -L$(ASCEND_LIB_DIR) -lacl_rt -L${CUSTOM_OPS_LIB_DIR} -l<custom_ops_so>
INCS = -I$(ASCEND_INC_DIR) -I${CUSTOM_OPS_INC_DIR}
TARGET = test_custom_ops

# Default target
all:
    g++ $(CXXFLAGS) $(SOURCES) $(INCS) $(LIBS) -o ${TARGET}
    @echo "${TARGET} compile completed"

# Test target
test:
    export LD_LIBRARY_PATH=${CUSTOM_OPS_LIB_DIR}:${LD_LIBRARY_PATH}; \
    ./$(TARGET)

# Clean build artifacts
clean:
    rm ${TARGET}

.PHONY: all test clean
```

## 执行测试程序

1. 关闭AI CPU算子验签功能。

    AI CPU算子包会在业务启动时加载至Device，加载过程中驱动默认会执行安全验签，以确保包的可信性。但用户自行编译生成的AI CPU算子包不包含签名头，因此需要手工关闭驱动的验签机制，才可以正常加载。

    参考如下命令，使用root用户在物理机上执行，以device 0为例：

    ```bash
    npu-smi set -t custom-op-secverify-enable -i 0 -d 1    # 开启验签配置
    npu-smi set -t custom-op-secverify-mode -i 0 -d 0      # 关闭自定义验签
    ```

    > [!NOTE]说明
    > 关闭驱动安全验签机制存在一定的安全风险，需要用户自行确保自定义通信算子的安全可靠，防止恶意攻击行为。

2. 修改AI CPU白名单。

    若用户新增AI CPU算子包，需同步将该AI CPU算子包配置到AI CPU白名单中。以root用户默认安装路径为例，编辑ascend_package_load.ini文件：

    ```bash
    vim /usr/local/Ascend/cann/conf/ascend_package_load.ini
    ```

    将下列内容追加到ascend_package_load.ini中：

    ```text
    name:<aicpu_kernel_file_name>
    install_path:2
    optional:true
    package_path:<aicpu_kernel_file_path>
    ```

    其中：

    - <aicpu_kernel_file_name\>：表示AI CPU算子包文件名，文件格式为tar.gz，例如：aicpu_hccl_custom_p2p.tar.gz。
    - <aicpu_kernel_file_path\>：表示AI CPU算子包在CANN包下的相对路径，例如：opp/vendors/cust/aicpu/kernel。

3. 编译并执行测试样例。

    ```bash
    # 编译
    make
    # 执行测试用例
    make test
    ```
