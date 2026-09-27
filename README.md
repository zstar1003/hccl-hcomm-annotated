# HCCL / HCOMM 中文源码导读

面向昇腾/CANN通信软件开发的**非官方中文注释副本**。保留HCCL、HCOMM两个完整上游快照，在实际源码中补充可检索的`// [中文导读]`注释，解释调用职责、参数单位、资源生命周期、条件分支与同步关系。

**当前为首批核心链路注释，不是全仓每个函数均已注释。** 共导入10,048个上游文件；新增注释覆盖17个核心源码文件、95个注释块、196行。重点是Ascend 950、新流程、AICPU_TS单算子路径，以及公共调度中的CCU/AIV资源分支。AllGather、AllReduce目前补充入口说明，未逐个注释其全部算法。

## 从哪里开始

1. [阅读路线与跨仓调用说明](docs/READING_GUIDE.zh-CN.md)：从算子下发，读到资源申请、建链、设备编排和数据原语。
2. [精确覆盖清单](docs/ANNOTATION_COVERAGE.md)：哪些文件补过、哪些内容尚未覆盖。
3. [验证说明](docs/VERIFICATION.md)：如何确认去除新增注释后恢复原始代码文本。
4. [HCCL原始文档](hccl/README.md) / [HCOMM原始文档](hcomm/README.md)：构建、接口和平台要求以各仓文档为准。

```bash
git lfs install
git clone https://github.com/zstar1003/hccl-hcomm-annotated.git
cd hccl-hcomm-annotated
rg -n '\[中文导读\]' hccl/src hcomm/src
```

上游文档中的PNG/GIF保留Git LFS管理，克隆完整文档图片需要Git LFS。该仓不包含私人PPT、截图或参考照片。

## 来源与版本

用户提供的官方项目入口在GitCode。源码实际从GitHub第三方镜像克隆；选取的提交及Git树哈希已与此前取得的GitCode官方源码副本核对一致。**镜像不标为官方，固定快照不宣称是最新版本。**

| 项目 | 官方入口 | GitHub克隆来源 | 固定提交 |
|---|---|---|---|
| HCCL | [cann/hccl](https://gitcode.com/cann/hccl) | [hicann/hccl](https://github.com/hicann/hccl) | `170ddeec539b4d693028ce6e0cf5c58933e4d46d` |
| HCOMM | [cann/hcomm](https://gitcode.com/cann/hcomm) | [hicann/hcomm](https://github.com/hicann/hcomm) | `87ce550f8f7c584e0ed89b0ec56699553d9c332d` |

完整锁定信息见[sources.lock.json](sources.lock.json)。仓库保存两个源码快照，不合并两个上游的全部提交历史。基线提交`efbe1cd23c64f4c62525c81d39812834113f4aa9`中的两棵子树与锁定上游树完全一致。

## 注释与验证原则

- 保留原代码、原注释、版权头、许可证和目录结构；不改API签名、算法逻辑或构建配置。
- 新增内容统一打标，不用“初始化变量”之类重复代码的注释填充数量。
- 明确新建/复用、控制面/数据面、Host/Device、任务提交/实际完成的区别。
- 说明只针对锁定版本；不同设备、Engine、算法、图模式、缓存命中及回退条件会改变路径。
- 已做源码注释复核与文本增量验证；**未在Ascend/CANN环境编译、运行UT/ST或上板验证**。插入注释会移动源文件行号，不声称二进制或`__LINE__`完全相同。

本地检查需要Node.js 20或更新版本和Git，无额外npm依赖：

```bash
node --test scripts/verify-annotations.test.mjs
node scripts/verify-annotations.mjs
git diff efbe1cd23c64f4c62525c81d39812834113f4aa9 -- hccl hcomm
```

## 许可证

保留[HCCL许可证](hccl/LICENSE)与[HCOMM许可证](hcomm/LICENSE)，主体为CANN Open Software License Agreement Version 2.0，包含处理器/系统用途限制。公开仓库不等于无限制使用，也不代表官方认证或背书。第三方文件按各自许可声明执行，详见[许可证与来源](LICENSES.md)。
