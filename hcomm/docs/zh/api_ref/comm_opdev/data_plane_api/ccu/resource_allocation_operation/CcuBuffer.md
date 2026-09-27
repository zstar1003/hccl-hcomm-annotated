# CcuBuffer

## 产品支持情况

<!-- npu="950" id1 -->
- Ascend 950PR/Ascend 950DT：支持
<!-- end id1 -->
<!-- npu="A3" id2 -->
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：不支持
<!-- end id2 -->
<!-- npu="910b" id3 -->
- Atlas A2 训练系列产品/Atlas A2 推理系列产品：不支持
<!-- end id3 -->
<!-- npu="910" id4 -->
- Atlas 训练系列产品：不支持
<!-- end id4 -->
<!-- npu="310p" id5 -->
- Atlas 推理系列产品：不支持
<!-- end id5 -->

## 功能说明

`ccu::CcuBuffer`是CCU kernel内片上Memory Slice（MS）的C++包装类，每个MS切片固定大小为4096字节。

- 构造即分配：默认构造函数自动申请1块MS虚拟句柄。
- 析构不释放：析构函数不释放硬件资源；虚拟句柄在翻译完成后失效，物理资源随CCU实例生命周期统一管理、回收。

MS切片是CCU die内的片上高速暂存区，用于在HBM与对端之间中转数据，或作为多Buffer归约（一次最多8 路MS）的操作数。

> [!NOTE]说明
> C++类名保留`Ccu`前缀（类名为`CcuBuffer`而非`Buffer`），与命名空间合写为`ccu::CcuBuffer`。

## 类声明

```cpp
namespace AscendC {
namespace ccu {
class CcuBuffer final {
public:
    CcuBuffer();                      // 构造即Alloc
    CcuBufferHandle handle{0};       // 虚拟句柄
};
} // namespace ccu
} // namespace AscendC
```

## 构造函数说明

| 构造形式 | 说明 |
| --- | --- |
| `CcuBuffer buf;` | 申请1块4KB MS虚拟句柄。只能在kernel注册阶段（`HcommCcuKernelRegister`执行的kernel函数体内）调用。 |

> [!CAUTION]注意
> `CcuBuffer` 类有copy / move 构造函数，它们只拷贝 `handle` 字段、不申请新的MS 切片——`CcuBuffer b2 = b1;` 后 `b1` 和 `b2` 指向同一块MS 切片。如需独立的CcuBuffer，必须显式 `CcuBuffer b2;` 默认构造（或用 `Array<CcuBuffer>` 申请物理连续的多块）。

构造失败时抛出异常（携带[CcuResult](../../../datatype_definition/CcuResult.md)错误码）。

## 约束说明

- 只能在kernel注册阶段构造CcuBuffer。
- 析构不释放硬件资源，不应在kernel之外保存或比较`handle`值，翻译完成后句柄即失效。
- 每个CcuBuffer固定代表4096字节的片上切片，大小不可配置。
- 操作CcuBuffer的接口（如`LocalCopy`、`LocalReduce`、`Read`、`Write`）传入的`len`不可超过4096字节。该上限须由调用方自行保证，超出会导致运行期硬件行为未定义（数据截断/越界等）。
- C++ 构造只申请虚拟句柄，恒成功不抛异常；`MS`物理资源不足时，由`HcommCcuKernelRegister`阶段返回`CCU_E_UNAVAIL`，不是在构造时抛出。如需多Buffer归约，可使用[`Array<CcuBuffer>`](Array.md)批量申请物理连续的多个Buffer。

## 调用示例

```cpp
using namespace AscendC::ccu;

CcuResult MyKernel(CcuKernelArg arg) {
    CcuBuffer buf;    // 申请1块4KB MS切片
    LocalAddr src;
    Variable len;
    Event evt;

    // 将HBM数据拷贝到MS Buffer
    LocalCopy(buf, src, len, evt);
    EventWait(evt);

    return CCU_SUCCESS;
}
```
