/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_KERNEL_MGR_H
#define HCOMM_CCU_KERNEL_MGR_H

#include <mutex>
#include <unordered_map>

#include "ccu_kernel.h"
#include "../ccu_instance/ccu_res_pack.h"
#include "../ccu_instance/ccu_instance.h"

#include "ccu_dev_mgr_imp.h"
#include "../ccu_representation/reps/translator/ccu_rep_translator_v1.h"

#include "ccu_ins_generator_base.h"

namespace hcomm {

using namespace CcuRep;

class CcuKernelMgr {
public:
    static CcuKernelMgr& GetInstance(const s32 userDevId);

    HcclResult Init();
    HcclResult Deinit();

    CcuResult Register(
        CcuResPack& resPack, const uint32_t dieId, const char* kernelFuncName, const void* kernelFunc,
        const void** kernelArgs, const uint32_t argNum, CcuInstance* ccuIns, CcuKernelHandle& kernelHandle);

    CcuResult GetKernelResourceRequest(
        const uint32_t dieId, const char* kernelFuncName, const void* kernelFunc, const void** kernelArgs,
        const uint32_t argNum, CcuResReq& resReq, uint32_t& instrCount);

    CcuResult Translate(const std::vector<CcuKernelHandle>& kernelHandles);

    CcuKernel* GetKernel(CcuKernelHandle kernelHandle);
    // 在锁内查找 kernel 并填充 CcuKernelInfo，裸指针不逃逸锁。
    CcuResult GetCcuKernelInfo(CcuKernelHandle kernelHandle, CcuKernelInfo& info);
    CcuResult UnRegister(CcuKernelHandle kernelHandle);

    CcuKernel* GetCurrentKernel();

private:
    explicit CcuKernelMgr() = default;
    ~CcuKernelMgr();

    CcuKernelMgr(const CcuKernelMgr& that) = delete;
    CcuKernelMgr& operator=(const CcuKernelMgr& that) = delete;

private:
    struct CcuTranslatResPack {
        std::vector<CcuResHandle> handles{};
    };

private:
    CcuResult BuildKernel(
        const uint32_t dieId, const char* kernelFuncName, const void* kernelFunc, const void** kernelArgs,
        const uint32_t argNum, CcuInstance* ccuIns);
    CcuResult PrepareConstValueResources();
    CcuResult AllocRes(CcuResPack& resPack);

    HcclResult InstantiationTranslator(const uint16_t dieId);
    HcclResult TransRepSequenceToMicrocode(const std::vector<CcuKernel*>& kernels, bool isFuncBlock);
    HcclResult LoadInstruction(const CcuRep::CcuInstrInfo& instrInfo, const uint32_t dieId);

    HcclResult GetResPackTotalResRepository(const CcuTranslatResPack& resPack, CcuResRepository& totalRes) const;

private:
    bool initializedFlag_{false};
    int32_t userDevId_{-1};
    std::mutex kernelMapMutex_{};
    std::mutex translateMutex_{};
    CcuKernelHandle kernelId_ = 0;
    std::unordered_map<CcuKernelHandle, std::unique_ptr<CcuKernel>> kernelMap_{};
    void* instructionLoadDevMem_{nullptr};

    std::unordered_map<uint16_t, std::unordered_map<uint16_t, std::shared_ptr<CcuRep::CcuRepTranslator>>> translators;
    std::unordered_map<uint16_t, std::unordered_map<uint16_t, std::shared_ptr<CcuRep::CcuRepReferenceManager>>>
        referenceMgrs;
    CcuTranslatResPack translatorResPack{};
    std::unique_ptr<CcuKernel> currKernel_{nullptr};
    std::shared_ptr<CcuInsGeneratorBase> insGenePtr;
    CcuVersion ccuVersion_{CcuVersion::CCU_INVALID};
};
}; // namespace hcomm
#endif // HCOMM_CCU_KERNEL_MGR_IMP_H
