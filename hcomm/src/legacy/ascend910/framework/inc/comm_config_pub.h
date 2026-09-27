/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_CONFIG_PUB_H
#define HCCL_COMM_CONFIG_PUB_H

#include "hccl_common.h"
#include "common.h"

constexpr u32 COMM_CONFIG_MAGIC_WORD = 0xf0f0f0f0; // Magic word值，用于校验传入的配置结构体是否已经被初始化
constexpr uint32_t COMM_ALGO_MAX_LENGTH = 1600;        // hccl algo max length
constexpr uint32_t COMM_RETRY_ENABLE_MAX_LENGTH = 50;  // hccl_retry_enable max length
constexpr uint32_t COMM_RETRY_PARAMS_MAX_LENGTH = 128; // hccl_retry_params max length
constexpr int32_t COMM_EXECTIMEOUT_CONFIG_NOT_SET = 0xffffffff;

enum CommConfigVersion {
    COMM_CONFIG_VERSION_ONE = 1,
    COMM_CONFIG_VERSION_TWO = 2,
    COMM_CONFIG_VERSION_THREE = 3,
    COMM_CONFIG_VERSION_FOUR = 4,
    COMM_CONFIG_VERSION_FIVE = 5,
    COMM_CONFIG_VERSION_SIX = 6,
    COMM_CONFIG_VERSION_SEVEN = 7,
    COMM_CONFIG_VERSION_EIGHT = 8,
    COMM_CONFIG_VERSION_NINE = 9,
    COMM_CONFIG_VERSION_TEN = 10,
    COMM_CONFIG_VERSION_ELEVEN = 11 // 当前支持的最高版本
};

enum CommConfigOpExpansion {
    COMM_CONFIG_OPEXPANSION_DEFAULT = 0, // Config配置默认模式
    COMM_CONFIG_OPEXPANSION_HOST = 1,    // Config配置HOST模式
    COMM_CONFIG_OPEXPANSION_AICPU = 2,   // Config配置AICPU模式
    COMM_CONFIG_OPEXPANSION_AIV = 3,     // Config配置AIV模式
    COMM_CONFIG_OPEXPANSION_ONLY_AIV = 4 // Config配置only aiv模式
};

// 通信域级别配置参数结构体 - 内部信息
using CommConfigInfo = struct CommConfigInfoDef {
    size_t configSize; // 配置结构体大小
    u32 magicWord;     // Magic word
    u32 version;       // HCCL版本
    char reserved[8];  // 8 byte 保留字段
};

// 通信域级别配置参数结构体 - 外部配置项
using CommConfigHandle = struct CommConfigHandleDef {
    CommConfigInfo info;
    u32 bufferSize;                      // ccl buffer 大小配置
    u32 deterministic;                   // 确定性计算配置
    char commName[COMM_NAME_MAX_LENGTH]; // 通信域名称
    char udi[UDI_MAX_LENGTH];            // user define information
    u32 opExpansionMode;                 // 0：默认值  1：host  2：aicpu  3:aiv 4:aiv only
    u32 trafficClass;
    u32 serviceLevel;
    u32 worldRankID;
    u64 jobID;
    u8 aclGraphZeroCopyEnable; ///< 0:关闭aclgraph零拷贝 1:开启aclgraph零拷贝
    s32 execTimeOut;
    char hcclAlgo[COMM_ALGO_MAX_LENGTH];
    char hcclRetryEnable[COMM_RETRY_ENABLE_MAX_LENGTH]; // hccl_retry_enable
    char hcclRetryParams[COMM_RETRY_PARAMS_MAX_LENGTH]; // hccl_retry_params
    char bufferName[BUFFER_NAME_MAX_LENGTH];            // cclbuffer名称
    u32 hcclQos = HCCL_COMM_QOS_CONFIG_NOT_SET;
    uint64_t symmetricMemoryStride; // 对称内存预留VA大小
    u32 sqDepth = HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET;
};

namespace hccl {
class CommConfig {
public:
    CommConfig(const std::string& commName); // 构造函数需传入默认的通信域ID
    CommConfig();
    ~CommConfig();

    HcclResult Load(const HcclCommConfig* userConfig); // 读取通信域配置
    u64 GetConfigBufferSize() const;                   // 获取CCL buffer大小配置
    u8 GetConfigDeterministic() const;                 // 获取确定性计算配置
    const std::string& GetConfigCommName() const;      // 获取通信域名称
    const std::string& GetConfigUdi() const;           // 获取UDI
    bool GetConfigAivMode() const;                     // 获取AIV配置
    bool GetConfigIsOnlyAivMode() const;               // 获取aiv only配置
    bool GetConfigAicpuUnfold() const;                 // 获取AICPU配置, 在310P和A3中AICPU展开
    u32 GetConfigTrafficClass() const;
    u32 GetConfigServiceLevel() const;
    u32 GetConfigWorldRankID() const;
    u64 GetConfigJobID() const;
    u8 GetConfigAclGraphZeroCopyEnable() const; // 获取aclGraphZeroCopyEnable 的配置值，在ExecOp Zerocopy准备流程中使用
    s32 GetConfigExecTimeOut() const;
    bool GetConfigExecTimeOutSet() const;
    std::vector<HcclAlgoType> GetConfigHcclAlgo(HcclCMDType opType = HcclCMDType::HCCL_CMD_ALL);
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& GetConfigHcclAlgoMap() const;
    bool GetConfigIntraServerRetryEnable() const;
    bool GetConfigInterServerRetryEnable() const;
    bool GetConfigInterSuperPodRetryEnable() const;
    u32 GetConfigRetryMaxCnt() const;
    u32 GetConfigRetryHoldTime() const;
    u32 GetConfigRetryIntervalTime() const;
    HcclResult SetConfigExecTimeOut(s32 execTimeOut);
    const std::string& GetConfigBufferName() const;
    u32 GetConfigHcclQos() const;
    u64 GetConfigSymmetricMemoryStride() const;
    u32 GetConfigSqDepth() const;
    HcclResult SetConfigTrafficClass(u32 trafficClass);
    HcclResult SetConfigServiceLevel(u32 serviceLevel);
    HcclResult SetConfigHcclQos(u32 hcclQos);
    HcclResult SetConfigHcclAlgoStr(const std::string& hcclAlgo);
    const std::string& GetConfigHcclAlgoStr() const;
    HcclResult SetConfigSqDepth(u32 sqDepth);

private:
    void InitAlgoConfig();
    void InitRetryEnable();
    HcclResult CheckMagicWord(const CommConfigHandle& config);     // 检查Magic Word是否合法
    HcclResult SetConfigByVersion(const CommConfigHandle& config); // 根据版本号读取配置，保证兼容性

    HcclResult SetConfigBufferSize(const CommConfigHandle& config);      // 设置通信Buffer配置
    HcclResult SetConfigDeterministic(const CommConfigHandle& config);   // 设置确定性计算配置
    HcclResult SetConfigCommName(const CommConfigHandle& config);        // 设置通信域名称
    HcclResult SetConfigUdi(const CommConfigHandle& config);             // 设置UDI
    HcclResult SetConfigOpExpansionMode(const CommConfigHandle& config); // 设置AIV和AICPU, 在310P和A3中AICPU展开
    HcclResult SetConfigExecTimeout(const CommConfigHandle& config);     // 设置HCCL执行超时时间
    HcclResult SetConfigHcclAlgo(const CommConfigHandle& config);        // 设置HCCL_ALGO
    HcclResult SetConfigHcclRetryEnable(const CommConfigHandle& config); // 设置retry_enable
    HcclResult SetConfigHcclRetryParams(const CommConfigHandle& config); // 设置retry_params
    HcclResult SetSpecificAlgTypeConfig(std::vector<std::string>& algos);
    HcclResult SplitRetryEnable(const std::string& retryConfig, std::vector<std::string>& retryEnables);
    HcclResult SetConfigRetryEnable(const std::vector<std::string>& retryEnables);
    HcclResult SetConfigBufferName(const CommConfigHandle& config); // 设置通信Buffer名称

    u64 bufferSize_;   // CCL buffer大小配置，单位B
    u8 deterministic_; // 确定性计算配置：0-关闭，1-开启确定性（不支持规约保序），2-开启确定性&规约保序，其他数字暂时保留
    std::string commName_;
    std::string udi_; // user define information，用于在报错日志中定位错误通信域
    bool aivMode_;
    bool aicpuUnfold_;
    u32 trafficClass_;
    u32 serviceLevel_;
    u32 worldRankID_;
    u64 jobID_;
    u8 aclGraphZeroCopyEnable_ = 0; // 0:关闭aclgraph零拷贝 1:开启aclgraph零拷贝
    bool onlyAivMode_;
    s32 execTimeOut_;
    bool execTimeOutSetByConfig_;
    std::map<HcclCMDType, std::vector<HcclAlgoType>> algoConfig_;
    bool retryEnable_[HCCL_RETRY_ENABLE_LEVEL_NUM];
    u32 retryMaxCnt_; // 重执行最大尝试次数，配置范围[0,60000]，默认值为3
    u32 retryHoldTime_;
    u32 retryIntervalTime_;  // 重执行间隔时间，配置范围[0,3600000]，默认值1000
    std::string bufferName_; // CCL buffer名称
    u32 hcclQos_;
    u64 symmetricMemoryStride_; // 对称内存预留VA大小，单位GB
    std::string hcclAlgoStr_;   // hcclAlgo原始字符串
    u32 sqDepth_;
};
} // namespace hccl
#endif /* HCCL_COMM_CONFIG_PUB_H */
