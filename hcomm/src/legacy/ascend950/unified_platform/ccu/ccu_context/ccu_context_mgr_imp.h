/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_CTX_MGR_IMP_H
#define HCCL_CCU_CTX_MGR_IMP_H

#include "ccu_ctx_mgr.h"

#include <mutex>
#include <unordered_map>
#include "ccu_device_manager.h"
#include "ccu_rep_translator.h"

namespace Hccl {

using namespace CcuRep;

class CtxMgrImp {
public:
    ~CtxMgrImp();

    CtxMgrImp(const CtxMgrImp& that) = delete;

    CtxMgrImp& operator=(const CtxMgrImp& that) = delete;

    /**
     * @brief 获取CtxMgrImp的实例
     * @param deviceLogicId 设备逻辑ID
     * @return 返回对应deviceLogicId的CtxMgrImp实例
     */
    static CtxMgrImp& GetInstance(s32 deviceLogicId);

    /**
     * @brief 初始化CtxMgrImp的实例
     * @param deviceLogicId 设备逻辑ID
     * @return void
     */
    void Init();

    /**
     * @brief 解初始化CtxMgrImp的实例
     * @param deviceLogicId 设备逻辑ID
     * @return void
     */
    void Deinit();

    /**
     * @brief 分配资源
     * @param ctxGroup ccuContext上下文组
     * @param resPack 资源包
     * @return 分配资源的结果
     */
    HcclResult AllocRes(CcuCtxGroup& ctxGroup, CcuResPack& resPack);

    /**
     * @brief 释放资源
     * @param ctxGroup 资源句柄
     * @return HcclResult 返回释放资源的结果
     * @note 释放由CcuCtxGroup handle指定的资源
     */
    HcclResult ReleaseRes(CcuCtxGroup& ctxGroup) const;

    /**
     * @brief 注册CcuCtxGroup
     * @param ctxGroup 要注册的CCU上下文组
     * @return 返回一个64位无符号整型值，注册成功的标识的ctxGroup映射的id
     */
    uint64_t Register(CcuCtxGroup& ctxGroup, bool isFuncBlock = false);

    /**
     * @brief 注销执行器
     *
     * @param executorId 执行器ID
     * @return HcclResult 注销结果
     * @note 此函数用于注销指定ID的执行器
     */
    HcclResult UnRegister(const uint64_t executorId);

    /**
     * 获取任务参数
     * @param executorId 任务ID
     * @param args CCU参数
     * @return 返回任务参数:对外层vector对应多个ctx(mission), 内存vector对应一个ctx下的多个任务参数
     */
    std::vector<std::vector<CcuTaskParam>> GetTaskParam(CcuTaskArg& ccuTaskArg, const uint64_t executorId);

    /**
     * 获取任务Profiling信息
     * @param entityId 任务ID
     * @return 返回任务Profiling信息:对外层vector对应多个ctx(mission), 内层vector对应一个ctx下的多个任务Profiling信息
     */
    std::vector<std::vector<CcuProfilingInfo>> GetProfilingInfo(CcuTaskArg& ccuTaskArg, const uint64_t entityId);

    /**
     * 获取CcuContext
     * @param executorId 任务ID
     * @param dieId Die ID
     * @param missionId Mission ID
     * @return 返回匹配的CcuContext, 未找到则返回nullptr
     */
    CcuContext* GetCtx(uint64_t executorId, uint32_t dieId, uint32_t missionId);

private:
    explicit CtxMgrImp();

    void CtxInit(CcuCtxGroup& ctxGroup) const;

    HcclResult AllocInstrRes(CcuCtxGroup& ctxGroup) const;
    HcclResult ReleaseInstrRes(CcuCtxGroup& ctxGroup) const;

    HcclResult GetResPackTotalResNum(const CcuResPack& resPack, CcuResReq& totalRes) const;

    inline int32_t GetResTotalNum(const vector<ResInfo>& resInfos) const
    {
        int32_t resNum = 0;

        for (ResInfo resInfo : resInfos) {
            resNum += static_cast<int32_t>(resInfo.num);
        }
        return resNum;
    }

    CcuResReq GetCtxGroupResReq(CcuCtxGroup& ctxGroup) const;

    void MergeCcuResReq(CcuResReq& resReqA, const CcuResReq& resReqB) const;

    HcclResult
    CompareResAndApplyAsNeeded(const CcuResReq& totalRes, const CcuResReq& resReq, CcuResPack& resPack) const;

    inline uint32_t GetReqResNum(uint32_t reqRes, uint32_t totalRes) const
    {
        return ((reqRes > totalRes) ? (reqRes - totalRes) : 0);
    }

    void SaveResPackToCtx(CcuCtxGroup& ctxGroup, CcuResPack& resPack) const;

    HcclResult InstantiationTranslator(uint16_t dieId);

    HcclResult TransRepResToPhyRes(CcuCtxGroup& ctxGroup) const;

    HcclResult GetResPackTotalResRepository(const CcuResPack& resPack, CcuResRepository& totalRes) const;

    inline void ExpandResInfo(vector<ResInfo>& expendResInfos, const vector<ResInfo>& resInfos) const
    {
        // 将resInfo中的资源信息扩展到mergedRes中
        for (auto& resInfo : resInfos) {
            for (uint32_t id = 0; id < resInfo.num; id++) {
                expendResInfos.push_back({(resInfo.startId + id), {1}});
            }
        }
    }

    CcuRepResource GetTotalCcuRepResource(CcuCtxGroup& ctxGroup) const;

    void MergeCtxRepResource(CcuRepResource& repResourceA, CcuRepResource& repResourceB) const;

    template <typename T1, typename T2>
    void ResetRepResourceTemplate(std::vector<T1>& resource, const std::vector<T2>& repository) const;

    void ResetRepResourceToResRepository(CcuRepResource& totalRepRes, const CcuResRepository& totalResRepository) const;

    template <typename T>
    void ProcessSharedResources(
        std::unordered_map<std::string, T>& resources,
        std::vector<std::unordered_map<std::string, T>>& exportedResources, uint32_t i) const;

    void ProcessInterCtxRes(CcuCtxGroup& ctxGroup) const;

    HcclResult SaveCtxMissionInfo(CcuCtxGroup& ctxGroup, array<vector<ResInfo>, MAX_CCU_IODIE_NUM>& missionId) const;

    void TransRepSequenceToMicrocode(CcuCtxGroup& ctxGroup, bool isFuncBlock);

    void LoadInstruction(CcuRep::CcuInstrInfo& instrInfo, uint32_t dieId);

    void DumpResReqInfo(const CcuResReq& totalRes) const;

    void DumpResRepositoryInfo(const CcuResRepository& resRepo) const;

private:
    bool initializedFlag_{false};

    // 创建一个无序的映射，键为uint64_t类型的executorId，值为CcuRepContext的智能指针
    std::unordered_map<uint64_t, CcuCtxGroup> ctxGroupMap_{};

    // 创建一个互斥量，用于保护contextMap_的并发访问
    std::mutex contextMapMutex_;

    uint64_t executorId_ = 0;

    s32 deviceLogicId_ = -1;

    // 指令模块加载使用的device的内存地址
    void* instructionLoadDevMem_ = nullptr;
    // translator实例
    std::unordered_map<uint16_t, std::unordered_map<uint16_t, std::shared_ptr<CcuRepTranslator>>> translators;
    std::unordered_map<uint16_t, std::unordered_map<uint16_t, std::shared_ptr<CcuRepReferenceManager>>> referenceMgrs;
    CcuResPack translatorResPack;
};
}; // namespace Hccl

#endif // HCCL_CCU_CTX_MGR_IMP_H
