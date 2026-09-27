# 算子下发

开发者完成通信算子Kernel注册后，需要将Kernel函数下发给具体的通信引擎执行。

对于CCU通信引擎，开发者需调用HcommCcuKernelLaunch接口进行Kernel下发。

## 代码示例

以自定义AllGather算子为例，使用CCU通信引擎时，Host侧下发CCU Kernel函数的代码片段如下：

```c
uint64_t currentRankSliceInputOffset = 0; // 卡间输入地址偏移量
uint64_t currentRankSliceOutputOffset = sliceSize * myRank; // 卡间目标地址偏移量
std::vector<uint64_t> taskArgs = {
    inputAddr,
    outputAddr,
    token,
    currentRankSliceInputOffset,
    currentRankSliceOutputOffset,
    sliceSize
};
CcuResult launchRet = HcommCcuKernelLaunch(thread, ccuKernel, taskArgs.data(), taskArgs.size()); // 下发CCU任务
if (launchRet != CCU_SUCCESS) {
    HCCL_ERROR("[CcuTempAllGatherMesh1DMem2Mem::ExecOp] kernel launch failed, ccuRet -> %d", launchRet);
    return ConvertCcuToHccl(launchRet);
}
```
