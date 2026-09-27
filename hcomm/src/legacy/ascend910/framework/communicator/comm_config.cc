/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "externalinput_pub.h"
#include "externalinput.h"
#include "comm_config_pub.h"
#include "adapter_error_manager_pub.h"
#include "adapter_rts_common.h"
#include "alg_env_config.h"

namespace hccl {
CommConfig::CommConfig(const std::string& commName)
    : bufferSize_(GetExternalInputCCLBuffSize()),
      deterministic_(GetExternalInputHcclDeterministicV2()),
      commName_(commName),
      aivMode_(GetExternalInputHcclAivMode()),
      aicpuUnfold_(GetExternalInputHcclAicpuUnfold()),
      trafficClass_(HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET),
      serviceLevel_(HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET),
      worldRankID_(0),
      jobID_(0),
      aclGraphZeroCopyEnable_(0),
      onlyAivMode_(false),
      execTimeOut_(GetInternalExecTimeOut()),
      execTimeOutSetByConfig_(false),
      retryMaxCnt_(GetExternalInputRetryMaxCnt()),
      retryHoldTime_(GetExternalInputRetryHoldTime()),
      retryIntervalTime_(GetExternalInputRetryIntervalTime()),
      bufferName_(""),
      hcclQos_(HCCL_COMM_QOS_CONFIG_NOT_SET),
      symmetricMemoryStride_(HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE),
      sqDepth_(HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET)
{
    InitAlgoConfig();
    InitRetryEnable();
}

CommConfig::CommConfig()
    : bufferSize_(GetExternalInputCCLBuffSize()),
      deterministic_(GetExternalInputHcclDeterministicV2()),
      aivMode_(GetExternalInputHcclAivMode()),
      aicpuUnfold_(GetExternalInputHcclAicpuUnfold()),
      trafficClass_(HCCL_COMM_TRAFFIC_CLASS_CONFIG_NOT_SET),
      serviceLevel_(HCCL_COMM_SERVICE_LEVEL_CONFIG_NOT_SET),
      worldRankID_(0),
      jobID_(0),
      aclGraphZeroCopyEnable_(0),
      onlyAivMode_(false),
      execTimeOut_(GetInternalExecTimeOut()),
      execTimeOutSetByConfig_(false),
      retryMaxCnt_(GetExternalInputRetryMaxCnt()),
      retryHoldTime_(GetExternalInputRetryHoldTime()),
      retryIntervalTime_(GetExternalInputRetryIntervalTime()),
      bufferName_(""),
      hcclQos_(HCCL_COMM_QOS_CONFIG_NOT_SET),
      symmetricMemoryStride_(HCCL_DEFAULT_SYMMETRIC_MEMORY_STRIDE),
      sqDepth_(HCCL_COMM_SQ_DEPTH_CONFIG_NOT_SET)
{
    InitAlgoConfig();
    InitRetryEnable();
}

CommConfig::~CommConfig() {}

void CommConfig::InitAlgoConfig()
{
    for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
        algoConfig_[static_cast<HcclCMDType>(opType)]
            = GetExternalInputHcclAlgoConfig(static_cast<HcclCMDType>(opType));
    }
}

void CommConfig::InitRetryEnable()
{
    retryEnable_[HCCL_RETRY_ENABLE_LEVEL_0] = GetExternalInputIntraServerRetryEnable();
    retryEnable_[HCCL_RETRY_ENABLE_LEVEL_1] = GetExternalInputInterServerRetryEnable();
    retryEnable_[HCCL_RETRY_ENABLE_LEVEL_2] = GetExternalInputInterSuperPodRetryEnable();
}

HcclResult CommConfig::Load(const HcclCommConfig* userConfig)
{
    // 检查是否为空
    CHK_PTR_NULL(userConfig);

    // 读取结构体的size
    size_t configSize = *(reinterpret_cast<const size_t*>(userConfig));
    HCCL_INFO("[Load] config size[%llu]", configSize);

    const size_t maxConfigSize = sizeof(CommConfigHandle);
    if (configSize > maxConfigSize) {
        HCCL_WARNING(
            "[Load] configSize[%llu] is larger than sizeof(CommConfigHandle)[%llu]", configSize, maxConfigSize);
        configSize = maxConfigSize;
    } else if (configSize < maxConfigSize) {
        HCCL_WARNING("[Load] configSize[%llu] is less than sizeof(CommConfigHandle)[%llu]", configSize, maxConfigSize);
    }

    // 根据size读取结构体
    CommConfigHandle configHandle;
    s32 sRet = memcpy_s(&configHandle, maxConfigSize, userConfig, configSize);
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Load] memcpy comm config fail. errorno[%d] "
            "params:destMaxSize[%u], count[%u]",
            sRet, maxConfigSize, configSize),
        HCCL_E_MEMORY);

    // 检查Magic word是否合法
    CHK_RET(CheckMagicWord(configHandle));

    // 根据版本号读取配置，检查配置参数合法性
    CHK_RET(SetConfigByVersion(configHandle));

    HCCL_RUN_INFO(
        "[Load] comm config info of [%s]: configSize[%llu], version[%u], opExpansionMode[%u]", commName_.c_str(),
        configHandle.info.configSize, configHandle.info.version, configHandle.opExpansionMode);
    HCCL_RUN_INFO(
        "[Load] comm config of [%s]: bufferSize[%llu], deterministic[%u], trafficClass[%u], serviceLevel[%u]"
        ", execTimeOut[%u]s, bufferName[%s], hcclQos[%u], symmetricMemoryStride[%llu], aclGraphZeroCopyEnable[%u]",
        commName_.c_str(), bufferSize_, deterministic_, trafficClass_, serviceLevel_, execTimeOut_, bufferName_.c_str(),
        hcclQos_, symmetricMemoryStride_, aclGraphZeroCopyEnable_);
    return HCCL_SUCCESS;
}

HcclResult CommConfig::CheckMagicWord(const CommConfigHandle& config)
{
    if (config.info.magicWord != COMM_CONFIG_MAGIC_WORD) {
        RPT_INPUT_ERR(
            true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
            std::vector<std::string>(
                {"HcclCommInitRootInfoConfig", std::to_string(config.info.magicWord), "magic word",
                 "The magic word must be initialized with the result of HcclCommConfigInit()"}));
        HCCL_ERROR(
            "[CheckMagicWord] Invalid magic word[0x%x]. Please make sure the config has been initialized by "
            "HcclCommConfigInit().",
            config.info.magicWord);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigByVersion(const CommConfigHandle& config)
{
    if (config.info.version > CommConfigVersion::COMM_CONFIG_VERSION_ELEVEN) {
        // 传入的config的版本高于当前版本，警告不支持的配置项将被忽略
        HCCL_WARNING(
            "[SetConfigByVersion] The version of provided config[%u] is higher than the current version[%u], "
            "unsupported configuration will be ignored.",
            config.info.version, CommConfigVersion::COMM_CONFIG_VERSION_ELEVEN);
    } else if (config.info.version < CommConfigVersion::COMM_CONFIG_VERSION_ELEVEN) {
        // 传入的config的版本低于当前版本，警告高版本支持的配置项将被忽略
        HCCL_WARNING(
            "[SetConfigByVersion] The version of provided config[%u] is lower than the current version[%u], "
            "configurations supported by later versions will be ignored.",
            config.info.version, CommConfigVersion::COMM_CONFIG_VERSION_ELEVEN);
    }

    if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_ONE) {
        // 版本大于等于1，设置CCL buffer、确定性计算配置
        CHK_RET(SetConfigBufferSize(config));
        CHK_RET(SetConfigDeterministic(config));
    }

    if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_TWO) {
        // 版本大于等于2，设置通信域名称
        CHK_RET(SetConfigCommName(config));
    }

    if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_THREE) {
        // 版本大于等于3，设置Udi
        CHK_RET(SetConfigUdi(config));
    }

    if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_FOUR) {
        // 版本大于等于4，设置Aiv、Aicpu
        CHK_RET(SetConfigOpExpansionMode(config));
    }

    if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_FIVE) {
        // 版本大于等于5，支持配置TC，SL
        trafficClass_ = config.trafficClass;
        serviceLevel_ = config.serviceLevel;
    }

    if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_SIX) {
        // 版本大于等于6
        worldRankID_ = config.worldRankID;
        jobID_ = config.jobID;
    }

    if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_SEVEN) {
        // 版本大于等于7，支持配置AclGraph使能/去使能
        RPT_INPUT_ERR(
            config.aclGraphZeroCopyEnable > 1, "EI0003",
            std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
            std::vector<std::string>(
                {"HcclCommInitRootInfoConfig", std::to_string(config.aclGraphZeroCopyEnable), "aclGraphZeroCopy",
                 "0 or 1."}));
        CHK_PRT_RET(
            config.aclGraphZeroCopyEnable > 1,
            HCCL_ERROR(
                "[CommConfig][SetConfigByVersion] aclGraphZeroCopyEnable value=[%u] invalid. support 0 or 1",
                config.aclGraphZeroCopyEnable),
            HCCL_E_PARA);
        aclGraphZeroCopyEnable_ = config.aclGraphZeroCopyEnable;
    }

    if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_EIGHT) {
        // 版本大于等于8，支持配置execTimeOut
        CHK_RET(SetConfigExecTimeout(config));
        // 支持配置HcclAlgo
        CHK_RET(SetConfigHcclAlgo(config));
        // 解析重执行设置
        CHK_RET(SetConfigHcclRetryEnable(config));
        CHK_RET(SetConfigHcclRetryParams(config));
    }

    if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_NINE) {
        // 版本大于等于9
        CHK_RET(SetConfigBufferName(config));
    }

    if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_TEN) {
        // 版本大于等于10,支持配置通信域级别的AI CPU SDMA QOS
        hcclQos_ = config.hcclQos;
        // 版本大于等于10，支持配置对称内存每个rank的预留VA大小
        symmetricMemoryStride_ = config.symmetricMemoryStride;
    }

    if (config.info.version >= CommConfigVersion::COMM_CONFIG_VERSION_ELEVEN) {
        // 版本大于等于11，支持配置通信域级别的sqDepth
        sqDepth_ = config.sqDepth;
    }
    HCCL_INFO("CommConfig version: config.info.version = [%u].", config.info.version);
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigBufferSize(const CommConfigHandle& config)
{
    if (config.bufferSize == HCCL_COMM_BUFFSIZE_CONFIG_NOT_SET) {
        // 默认跟随环境变量配置
        HCCL_INFO(
            "[SetConfigByVersion] The hcclBufferSize is not configured, use the env config [%u](Bytes) as default.",
            bufferSize_);
    } else if (config.bufferSize < HCCL_CCL_COMM_BUFFER_MIN) {
        RPT_INPUT_ERR(
            true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
            std::vector<std::string>(
                {"HcclCommInitRootInfoConfig", std::to_string(config.bufferSize), "hcclBufferSize",
                 "should be equal to or greater than 1(MB)."}));
        HCCL_ERROR(
            "[%s][%s] The configuration of hcclBufferSize[%u(MB)] is invalid, which should be "
            "greater than %u(MB).",
            LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), config.bufferSize,
            HCCL_CCL_COMM_BUFFER_MIN);
        return HCCL_E_PARA;
    } else {
        // 使用config配置
        bufferSize_ = static_cast<u64>(config.bufferSize) * HCCL_CCL_COMM_FIXED_CALC_BUFFER_SIZE; // MByte 转 Byte
    }
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigDeterministic(const CommConfigHandle& config)
{
    if (config.deterministic == HCCL_COMM_DETERMINISTIC_CONFIG_NOT_SET) {
        // 默认跟随环境变量配置
        HCCL_INFO(
            "[SetConfigByVersion] The hcclDeterministic is not configured, use the env config [%u] as default.",
            deterministic_);
    } else if (config.deterministic > DETERMINISTIC_STRICT) {
        RPT_INPUT_ERR(
            true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
            std::vector<std::string>(
                {"HcclCommInitRootInfoConfig", std::to_string(config.deterministic), "hcclDeterministic",
                 "should be 0(disable) , 1(enable) or 2(strict)."}));
        HCCL_ERROR(
            "[%s][%s] The configuration of hcclDeterministic[%u] is invalid, "
            "which should be 0(disable) , 1(enable) or 2(strict).",
            LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), config.deterministic);
        return HCCL_E_PARA;
    } else {
        if (config.deterministic == DETERMINISTIC_STRICT) {
            DevType deviceType;
            CHK_RET(hrtGetDeviceType(deviceType));
            if (deviceType != DevType::DEV_TYPE_910B && deviceType != DevType::DEV_TYPE_910_93) {
                RPT_INPUT_ERR(
                    true, "EI0003", std::vector<std::string>({"ccl_op", "value", "parameter", "expect"}),
                    std::vector<std::string>(
                        {"HcclCommInitRootInfoConfig", std::to_string(config.deterministic), "hcclDeterministic",
                         "set to 2(strict), only support A2/A3."}));
                HCCL_ERROR(
                    "[%s][%s] The configuration of hcclDeterministic[%u] is set to "
                    "2(strict), and only support A2/A3",
                    LOG_KEYWORDS_TASK_EXEC.c_str(), LOG_KEYWORDS_INVALID_ARGUMENT.c_str(), config.deterministic);
                return HCCL_E_PARA;
            }
        }
        deterministic_ = static_cast<u8>(config.deterministic); // 前面已保证数值不超过UINT8_MAX，直接进行类型转换
    }
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigCommName(const CommConfigHandle& config)
{
    if (config.commName[0] != '\0') {
        auto commNameLength = strlen(config.commName);
        commNameLength = commNameLength < COMM_NAME_MAX_LENGTH ? commNameLength : COMM_NAME_MAX_LENGTH;
        commName_ = std::string(config.commName, commNameLength);
    }
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigUdi(const CommConfigHandle& config)
{
    if (config.udi[0] == '\0') {
        udi_ = "Unspecified";
        return HCCL_SUCCESS;
    }
    auto udiLength = strlen(config.udi);
    udiLength = udiLength < COMM_NAME_MAX_LENGTH ? udiLength : COMM_NAME_MAX_LENGTH;
    udi_ = std::string(config.udi, udiLength);
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigBufferName(const CommConfigHandle& config)
{
    if (config.bufferName[0] != '\0') {
        auto bufferNameLength = strlen(config.bufferName);
        bufferNameLength = bufferNameLength < BUFFER_NAME_MAX_LENGTH ? bufferNameLength : BUFFER_NAME_MAX_LENGTH;
        bufferName_ = std::string(config.bufferName, bufferNameLength);
    }
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigOpExpansionMode(const CommConfigHandle& config)
{
    switch (config.opExpansionMode) {
        case COMM_CONFIG_OPEXPANSION_DEFAULT:
            HCCL_INFO(
                "CommConfig is set to 0(default), aicpuUnfold_ is [%d] and aivMode_ is [%d].", aicpuUnfold_, aivMode_);
            break;
        case COMM_CONFIG_OPEXPANSION_HOST:
            aivMode_ = false;
            aicpuUnfold_ = false;
            HCCL_INFO(
                "CommConfig is set to 1(host), aicpuUnfold_ is [%d] and aivMode_ is [%d].", aicpuUnfold_, aivMode_);
            break;
        case COMM_CONFIG_OPEXPANSION_AICPU:
            // 目前只有A3和300I支持Aicpu展开
            DevType deviceType;
            CHK_RET(hrtGetDeviceType(deviceType));
            if (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B) {
                aicpuUnfold_ = true;
                aivMode_ = false;
                HCCL_INFO(
                    "CommConfig is set to 2(aicpuUnfold_), aicpuUnfold_ is [%d] and aivMode_ is [%d].", aicpuUnfold_,
                    aivMode_);
            } else {
                HCCL_WARNING(
                    "Only A3 and 300I support aicpu unfold, set aicpuUnfold_ to [%d] and aivMode_ to [%d].",
                    aicpuUnfold_, aivMode_);
            }
            break;
        case COMM_CONFIG_OPEXPANSION_AIV:
            aivMode_ = true;
            HCCL_INFO(
                "CommConfig is set to 3(aivMode), aicpuUnfold_ is [%d] and aivMode_ is [%d].", aicpuUnfold_, aivMode_);
            break;
        case COMM_CONFIG_OPEXPANSION_ONLY_AIV:
            onlyAivMode_ = true;
            HCCL_INFO(
                "CommConfig is set to 4(aivOnly), aicpuUnfold_ is [%d] and aivMode_ is [%d] onlyAivMode_ is[%d].",
                aicpuUnfold_, aivMode_, onlyAivMode_);
            break;
        default:
            // 目前opExpansionMode的合法值为[0,4]，值不合法时回退为环境变量配置
            HCCL_WARNING(
                "Current version not support opExpansionMode[%u], set aicpuUnfold_ to [%d] and aivMode_ to [%d].",
                config.opExpansionMode, aicpuUnfold_, aivMode_);
            break;
    }

    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigExecTimeout(const CommConfigHandle& config)
{
    if (config.execTimeOut == COMM_EXECTIMEOUT_CONFIG_NOT_SET) {
        // 默认跟随环境变量
        HCCL_INFO(
            "[SetConfigByVersion] The hcclExecTimeOut is not configured, use the env config [%u s] as default.",
            execTimeOut_);
    } else {
        s32 execTimeOut = config.execTimeOut;

        DevType deviceType;
        CHK_RET(hrtGetDeviceType(deviceType)); // 910A和910B/C要分开
        if (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B) {
            // 910B和910_93算子超时时间范围0s-2147483647s,其中0代表永不超时
            if ((execTimeOut < 0) || (execTimeOut > HCCL_EXEC_TIME_OUT_S_910_93)) {
                HCCL_WARNING(
                    "[SetConfigByVersion][SetConfigExecTimeout] The configuration of ComConfigHcclExecTimeOut[%d]s is "
                    "invalid, "
                    "which should be a number greater than or equal to 0s and less "
                    "than or equal to 2147483647s",
                    execTimeOut);
                return HCCL_SUCCESS;
            }
        } else {
            // 非910B和910_93算子超时时间范围1s-17340s
            if ((execTimeOut <= 0) || (execTimeOut > HCCL_EXEC_TIME_OUT_S)) {
                HCCL_WARNING(
                    "[SetConfigByVersion] The configuration of ComConfigHcclExecTimeOut[%d]s is invalid, "
                    "which should be a number greater than 0s and less "
                    "than or equal to 17340s",
                    execTimeOut);
                return HCCL_SUCCESS;
            }

            s32 intPart = execTimeOut / HCCL_INTEVAL_EXEC_TIME_OUT_S;
            intPart = (intPart == 0) ? 1 : intPart;
            execTimeOut = intPart * HCCL_INTEVAL_EXEC_TIME_OUT_S;
        }

        double timeout = execTimeOut;
        execTimeOut_ = static_cast<s32>(std::ceil(timeout));
        execTimeOutSetByConfig_ = true;
        HCCL_INFO("[SetConfigByVersion] HCCL_EXEC_TIMEOUT set by config to [%d]s", execTimeOut);
    }

    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigHcclAlgo(const CommConfigHandle& config)
{
    if (config.hcclAlgo[0] == '\0') {
        return HCCL_SUCCESS;
    }

    auto hcclAlgoLength = strlen(config.hcclAlgo);
    hcclAlgoLength = hcclAlgoLength < COMM_ALGO_MAX_LENGTH ? hcclAlgoLength : COMM_ALGO_MAX_LENGTH;
    std::string algoConfig = std::string(config.hcclAlgo, hcclAlgoLength);

    algoConfig.erase(std::remove(algoConfig.begin(), algoConfig.end(), ' '), algoConfig.end());
    if (algoConfig.empty()) {
        HCCL_WARNING("[SetConfigHcclAlgo]hccl algo config is empty, HCCL use externalinput algo selection.");
        return HCCL_SUCCESS;
    }
    std::vector<std::string> algoPerOptype;
    CHK_RET(SplitHcclOpType(algoConfig, algoPerOptype));
    bool anyCommonConfig = false;
    bool anySpecificConfig = false;
    CHK_RET(CheckAlgoConfigValid(algoPerOptype, anyCommonConfig, anySpecificConfig));
    if (anyCommonConfig) {
        std::vector<HcclAlgoType> algType;
        CHK_RET(ParseAlgoString("all op type", algoPerOptype[0], algType));
        for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
            algoConfig_[static_cast<HcclCMDType>(opType)] = algType;
        }
    } else {
        CHK_RET(SetSpecificAlgTypeConfig(algoPerOptype));
    }
    HCCL_RUN_INFO("HCCL_ALGO set by HcclCommConfig to [%s]", algoConfig.c_str());

    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigHcclRetryEnable(const CommConfigHandle& config)
{
    if (config.hcclRetryEnable[0] == '\0') {
        return HCCL_SUCCESS;
    }
    auto retryEnableLength = strlen(config.hcclRetryEnable);
    retryEnableLength
        = retryEnableLength < COMM_RETRY_ENABLE_MAX_LENGTH ? retryEnableLength : COMM_RETRY_ENABLE_MAX_LENGTH;
    std::string retryConfig = std::string(config.hcclRetryEnable, retryEnableLength);
    // 去除空格
    retryConfig.erase(std::remove(retryConfig.begin(), retryConfig.end(), ' '), retryConfig.end());
    if (retryConfig.empty()) {
        HCCL_WARNING(
            "[%s] Hccl retry config is empty. The retryEnable of all levels is "
            "set by environment variable.",
            __func__);
        return HCCL_SUCCESS;
    }
    std::vector<std::string> retryEnables;
    HcclResult ret = SplitRetryEnable(retryConfig, retryEnables);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_WARNING(
            "[CommConfig][SetConfigHcclRetryEnable] Hccl retry config[%s] is invalid. "
            "expect: L1:0, L2:0",
            retryConfig.c_str()),
        ret);
    CHK_RET(SetConfigRetryEnable(retryEnables));
    HCCL_RUN_INFO("HCCL_OP_RETRY_ENABLE set by commconfig to [%s].", retryConfig.c_str());
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SplitRetryEnable(const std::string& retryConfig, std::vector<std::string>& retryEnables)
{
    std::string remainRetryConfig;
    std::size_t found = retryConfig.find(",");
    if ((found == 0) || (found == (retryConfig.length() - 1))) {
        HCCL_ERROR("[SplitRetryEnable] retry config is invalid.");
        return HCCL_E_PARA;
    } else if (found != std::string::npos) {
        remainRetryConfig = retryConfig.substr(found + 1);
    } else {
        // 最后一组配置,剩余的字符串为空
        remainRetryConfig = "";
    }
    retryEnables.push_back(retryConfig.substr(0, found));

    if (retryEnables.size() > HCCL_RETRY_ENABLE_LEVEL_NUM) {
        HCCL_ERROR(
            "[SplitRetryEnable] retryEnable config is invalid. retryEnable level is more than %u.",
            HCCL_RETRY_ENABLE_LEVEL_NUM);
        return HCCL_E_PARA;
    }
    if (!remainRetryConfig.empty()) {
        CHK_RET(SplitRetryEnable(remainRetryConfig, retryEnables));
    }
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigRetryEnable(const std::vector<std::string>& retryEnables)
{
    const std::map<std::string, u32> hcclRetryLevelMap
        = {{"L0", HCCL_RETRY_ENABLE_LEVEL_0}, {"L1", HCCL_RETRY_ENABLE_LEVEL_1}, {"L2", HCCL_RETRY_ENABLE_LEVEL_2}};

    std::map<std::string, u32> countHcclRetryLevelMap = {{"L0", 0}, {"L1", 0}, {"L2", 0}};

    const std::map<std::string, bool> hcclRetryEnableMap = {{"0", false}, {"1", true}};
    for (const auto& retryEnableLevel : retryEnables) {
        u32 level = 0;
        bool retryEnable = false;
        std::size_t found = retryEnableLevel.find(":");
        if ((found == 0) || (found == (retryEnableLevel.length() - 1))) {
            HCCL_INFO("[SetRetryEnable] Hccl retryEnableLevel is invalid.");
            return HCCL_SUCCESS;
        }
        std::string orginalLevel = retryEnableLevel.substr(0, found);
        std::string orginalRetryEnable = retryEnableLevel.substr(found + 1);
        if (orginalLevel == "L0") {
            HCCL_RUN_WARNING("[SetConfigRetryEnable] L0 config does not take effect");
        }
        // 检查是否存在重复配置level
        auto iterCountRetryLevel = countHcclRetryLevelMap.find(orginalLevel);
        if (iterCountRetryLevel == countHcclRetryLevelMap.end()) {
            HCCL_RUN_WARNING(
                "[SetRetryEnable] Retry config is invalid, level %s is not supported.", orginalLevel.c_str());
            return HCCL_SUCCESS;
        }
        if (countHcclRetryLevelMap[orginalLevel] == 1) {
            HCCL_RUN_WARNING(
                "[SetRetryEnable] Retry config level[%s] is repeated, expect: L1:0, L2:0", orginalLevel.c_str());
            return HCCL_SUCCESS;
        }
        countHcclRetryLevelMap[orginalLevel] += 1;
        // 获取level和对应的retryEnable，并赋值给g_externalInput.hcclRetryConfig
        auto iterRetryLevel = hcclRetryLevelMap.find(orginalLevel);
        if (iterRetryLevel == hcclRetryLevelMap.end()) {
            HCCL_RUN_WARNING(
                "[SetRetryEnable] Retry config is invalid, level %s is not supported.", orginalLevel.c_str());
            return HCCL_SUCCESS;
        }
        auto iterRetryEnable = hcclRetryEnableMap.find(orginalRetryEnable);
        if (iterRetryEnable == hcclRetryEnableMap.end()) {
            HCCL_RUN_WARNING(
                "[SetRetryEnable] Retry config is invalid, retryEnable %s is not supported.",
                orginalRetryEnable.c_str());
            return HCCL_SUCCESS;
        }
        level = iterRetryLevel->second;
        retryEnable = iterRetryEnable->second;
        retryEnable_[level] = retryEnable;
    }
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigHcclRetryParams(const CommConfigHandle& config)
{
    if (config.hcclRetryParams[0] == '\0') {
        return HCCL_SUCCESS;
    }
    auto retryParamsLength = strlen(config.hcclRetryParams);
    retryParamsLength
        = retryParamsLength < COMM_RETRY_PARAMS_MAX_LENGTH ? retryParamsLength : COMM_RETRY_PARAMS_MAX_LENGTH;
    std::string retryParams = std::string(config.hcclRetryParams, retryParamsLength);
    u32 maxcnt = 0;
    u32 holdtime = 0;
    u32 intervaltime = 0;
    int ret = 0;
    ret = sscanf_s(retryParams.c_str(), "MaxCnt:%u, HoldTime:%u, IntervalTime:%u", &maxcnt, &holdtime, &intervaltime);
    /* 三个参数全部解析成功，返回值为3，否则不等于3 */
    if ((ret != 3) || (maxcnt > HCCL_RETRY_MAXCNT_MAX) || (maxcnt < HCCL_RETRY_MAXCNT_MIN)
        || (holdtime > HCCL_RETRY_HLOD_TIME_MAX) || (intervaltime > HCCL_RETRY_INTERVAL_MAX)) {
        HCCL_ERROR(
            "[SetConfigHcclRetryParams]fail, HCCL_OP_RETRY_PARAMS: %s is invalid, format must be: "
            "MaxCnt:cnt, HoldTime:time, IntervalTime:time, cnt range is [1, 10], time range is [0, 60000]ms.",
            retryParams.c_str());
        return HCCL_E_PARA;
    }
    retryMaxCnt_ = maxcnt;
    retryHoldTime_ = holdtime;
    retryIntervalTime_ = intervaltime;

    HCCL_RUN_INFO(
        "[SetConfigHcclRetryParams]HCCL_OP_RETRY_PARAMS is set, "
        "MaxCnt is [%u], HoldTime is [%u]ms, IntervalTime is [%u]ms.",
        maxcnt, holdtime, intervaltime);
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetSpecificAlgTypeConfig(std::vector<std::string>& algos)
{
    for (std::string& algConfig : algos) {
        std::size_t found = algConfig.find("=");
        std::string opStringName = algConfig.substr(0, found);
        if (opStringName == "others") {
            std::vector<HcclAlgoType> algType;
            std::string remainAlgoConfig = algConfig.substr(found + 1);
            CHK_RET(ParseAlgoString("others op type", remainAlgoConfig, algType));
            for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
                algoConfig_[static_cast<HcclCMDType>(opType)] = algType;
            }
        }
    }
    std::map<std::string, HcclCMDType> hcclOpTypeMap = {
        {"broadcast", HcclCMDType::HCCL_CMD_BROADCAST},
        {"allreduce", HcclCMDType::HCCL_CMD_ALLREDUCE},
        {"reduce", HcclCMDType::HCCL_CMD_REDUCE},
        {"send", HcclCMDType::HCCL_CMD_SEND},
        {"receive", HcclCMDType::HCCL_CMD_RECEIVE},
        {"allgather", HcclCMDType::HCCL_CMD_ALLGATHER},
        {"reducescatter", HcclCMDType::HCCL_CMD_REDUCE_SCATTER},
        {"alltoall", HcclCMDType::HCCL_CMD_ALLTOALL},
        {"gather", HcclCMDType::HCCL_CMD_GATHER},
        {"scatter", HcclCMDType::HCCL_CMD_SCATTER},
        {"sendrecv", HcclCMDType::HCCL_CMD_BATCH_SEND_RECV},
    };
    for (std::string& algConfig : algos) {
        std::size_t found = algConfig.find("=");
        std::string opStringName = algConfig.substr(0, found);
        if (hcclOpTypeMap.find(opStringName) != hcclOpTypeMap.end()) {
            HcclCMDType optype = hcclOpTypeMap[opStringName];
            std::string remainAlgoConfig = algConfig.substr(found + 1);
            std::vector<HcclAlgoType> algType;
            CHK_RET(ParseAlgoString(opStringName, remainAlgoConfig, algType));
            if (algType[0] == HcclAlgoType::HCCL_ALGO_TYPE_NULL) {
                HCCL_WARNING("[SetSpecificAlgType] specific config level0 not support null type.");
                return HCCL_SUCCESS;
            }
            algoConfig_[optype] = algType;
        } else {
            HCCL_WARNING(
                "[SetSpecificAlgType] specific config optype[%s] is invalid, please check", opStringName.c_str());
            return HCCL_SUCCESS;
        }
    }
    algoConfig_[HcclCMDType::HCCL_CMD_ALLTOALLV] = algoConfig_[HcclCMDType::HCCL_CMD_ALLTOALL];
    algoConfig_[HcclCMDType::HCCL_CMD_ALLTOALLVC] = algoConfig_[HcclCMDType::HCCL_CMD_ALLTOALL];
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigTrafficClass(u32 trafficClass)
{
    trafficClass_ = trafficClass;
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigServiceLevel(u32 serviceLevel)
{
    serviceLevel_ = serviceLevel;
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigHcclQos(u32 hcclQos)
{
    hcclQos_ = hcclQos;
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigHcclAlgoStr(const std::string& hcclAlgo)
{
    hcclAlgoStr_ = hcclAlgo;
    return HCCL_SUCCESS;
}

const std::string& CommConfig::GetConfigHcclAlgoStr() const { return hcclAlgoStr_; }

HcclResult CommConfig::SetConfigSqDepth(u32 sqDepth)
{
    sqDepth_ = sqDepth;
    return HCCL_SUCCESS;
}

HcclResult CommConfig::SetConfigExecTimeOut(s32 execTimeOut)
{
    execTimeOut_ = execTimeOut;
    return HCCL_SUCCESS;
}

u64 CommConfig::GetConfigBufferSize() const { return bufferSize_; }

u8 CommConfig::GetConfigDeterministic() const { return deterministic_; }

const std::string& CommConfig::GetConfigCommName() const { return commName_; }

const std::string& CommConfig::GetConfigUdi() const { return udi_; }

bool CommConfig::GetConfigAivMode() const { return aivMode_; }

bool CommConfig::GetConfigIsOnlyAivMode() const { return onlyAivMode_; }

bool CommConfig::GetConfigAicpuUnfold() const { return aicpuUnfold_; }

u32 CommConfig::GetConfigTrafficClass() const { return trafficClass_; }

u32 CommConfig::GetConfigServiceLevel() const { return serviceLevel_; }

u32 CommConfig::GetConfigWorldRankID() const { return worldRankID_; }

u64 CommConfig::GetConfigJobID() const { return jobID_; }

u8 CommConfig::GetConfigAclGraphZeroCopyEnable() const { return aclGraphZeroCopyEnable_; }

s32 CommConfig::GetConfigExecTimeOut() const { return execTimeOut_; }

bool CommConfig::GetConfigExecTimeOutSet() const { return execTimeOutSetByConfig_; }

std::vector<HcclAlgoType> CommConfig::GetConfigHcclAlgo(HcclCMDType opType) { return algoConfig_[opType]; }

const std::map<HcclCMDType, std::vector<HcclAlgoType>>& CommConfig::GetConfigHcclAlgoMap() const { return algoConfig_; }

bool CommConfig::GetConfigIntraServerRetryEnable() const { return retryEnable_[HCCL_RETRY_ENABLE_LEVEL_0]; }

bool CommConfig::GetConfigInterServerRetryEnable() const { return retryEnable_[HCCL_RETRY_ENABLE_LEVEL_1]; }

bool CommConfig::GetConfigInterSuperPodRetryEnable() const { return retryEnable_[HCCL_RETRY_ENABLE_LEVEL_2]; }

u32 CommConfig::GetConfigRetryMaxCnt() const { return retryMaxCnt_; }

u32 CommConfig::GetConfigRetryHoldTime() const { return retryHoldTime_; }

u32 CommConfig::GetConfigRetryIntervalTime() const { return retryIntervalTime_; }

const std::string& CommConfig::GetConfigBufferName() const { return bufferName_; }

u32 CommConfig::GetConfigHcclQos() const
{
    HCCL_INFO("[GetConfigHcclQos] hcclQos = %u", hcclQos_);
    return hcclQos_;
}

u64 CommConfig::GetConfigSymmetricMemoryStride() const { return symmetricMemoryStride_; }

u32 CommConfig::GetConfigSqDepth() const { return sqDepth_; }
} // namespace hccl
