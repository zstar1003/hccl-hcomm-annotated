/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_binary_data_operator.h"

#include <cstdint>
#include <cstdlib> // strtoull
#include <cstring>
#include <iostream>
#include <sys/stat.h>
#include "sim_log.h"
#include "sim_models.h"

namespace HcclSim {
static int WriteExact(FILE* fp, const void* ptr, size_t size, size_t n)
{
    return fwrite(ptr, size, n, fp) == n ? 0 : -1;
}

static int ReadExact(FILE* fp, void* ptr, size_t size, size_t n) { return fread(ptr, size, n, fp) == n ? 0 : -1; }

HcclVmResult FileHeaderWrite(FILE* fp, const FileHeader& header)
{
    if (WriteExact(fp, &header, sizeof(header), 1)) {
        HCCL_VM_ERROR("Write file header failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult FileHeaderRead(FILE* fp, FileHeader& header, uint32_t magic)
{
    if (ReadExact(fp, &header, sizeof(header), 1)) {
        HCCL_VM_ERROR("Read file header failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    if (header.magic != magic) {
        HCCL_VM_ERROR("Unmatched magic number: {} != {}", header.magic, magic);
        return HcclVmResult::HCCL_SIM_E_NOT_SUPPORT;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclVmSynDataRead(FILE* fp, HcclVmSynData& synData, uint32_t magic)
{
    auto ret = FileHeaderRead(fp, synData.header, magic);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Read file header failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = ModelInfoRead(fp, synData.model_info);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Read file model info failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    bool isCcuMode = synData.model_info.comm.op_expansion_mode == sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    bool isAivMode = synData.model_info.comm.op_expansion_mode == sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AIV;
    if (isCcuMode) {
        ret = ChannelInfoRead(fp, synData.channel_info);
    } else if (isAivMode) {
        HCCL_VM_ERROR("AIV mode do not read channel or jetty. ");
    } else {
        ret = ChannelInfoRead(fp, synData.channel_info);
    }

    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Read file channel info failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = MemLayoutRead(fp, synData.memory_info);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Read file memory layout info failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclVmSynDataWrite(FILE* fp, const HcclVmSynData& synData)
{
    auto ret = FileHeaderWrite(fp, synData.header);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Write file header failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = ModelInfoWrite(fp, synData.model_info);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Write file model info failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    bool isCcuMode = synData.model_info.comm.op_expansion_mode == sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_CCU;
    bool isAivMode = synData.model_info.comm.op_expansion_mode == sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AIV;
    if (isCcuMode) {
        ret = ChannelInfoWrite(fp, synData.channel_info);
    } else if (isAivMode) {
        HCCL_VM_WARN("AIV mode do not write channel or jetty. ");
    } else {
        ret = ChannelInfoWrite(fp, synData.channel_info);
    }

    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Write file channel info failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = MemLayoutWrite(fp, synData.memory_info);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Write file memory layout info failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclVmInstrDataRead(FILE* fp, HcclVmInstrData& instrData, uint32_t magic)
{
    auto ret = FileHeaderRead(fp, instrData.header, magic);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Read file header failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    instrData.instr_data.resize(instrData.header.count);
    for (uint32_t i = 0; i < instrData.header.count; i++) {
        ret = MicrocodeInstrRead(fp, instrData.instr_data[i]);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("Read file microcode instruction info failed. times= {}", i);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclVmInstrDataWrite(FILE* fp, const HcclVmInstrData& instrData)
{
    auto ret = FileHeaderWrite(fp, instrData.header);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Write file header failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    for (uint32_t i = 0; i < instrData.header.count; i++) {
        ret = MicrocodeInstrWrite(fp, instrData.instr_data[i]);
        if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
            HCCL_VM_ERROR("Write file microcode instruction info failed. times= {}", i);
            return HcclVmResult::HCCL_SIM_E_INTERNAL;
        }
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ModelInfoCommWrite(FILE* fp, const ModelInfoCommInner& comm)
{
    if (WriteExact(fp, &comm, sizeof(comm), 1)) {
        HCCL_VM_ERROR("Write file model info failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ModelInfoCommRead(FILE* fp, ModelInfoCommInner& comm)
{
    if (ReadExact(fp, &comm, sizeof(comm), 1)) {
        HCCL_VM_ERROR("Read file model info failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult VDataDesTagWrite(FILE* fp, const VDataDesTagInner& vDataDes)
{
    if (WriteExact(fp, &vDataDes.dataType, sizeof(vDataDes.dataType), 1)) {
        HCCL_VM_ERROR("Write file vDataDes dataType failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, &vDataDes.count, sizeof(vDataDes.count), 1)) {
        HCCL_VM_ERROR("Write file vDataDes count failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, vDataDes.displs.data(), sizeof(uint64_t), vDataDes.count)) {
        HCCL_VM_ERROR("Write file vDataDes displs failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, vDataDes.counts.data(), sizeof(uint64_t), vDataDes.count)) {
        HCCL_VM_ERROR("Write file vDataDes counts failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult VDataDesTagRead(FILE* fp, VDataDesTagInner& vDataDes)
{
    if (ReadExact(fp, &vDataDes.dataType, sizeof(vDataDes.dataType), 1)) {
        HCCL_VM_ERROR("Read file vDataDes dataType failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (ReadExact(fp, &vDataDes.count, sizeof(vDataDes.count), 1)) {
        HCCL_VM_ERROR("Read file vDataDes count failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    vDataDes.displs.resize(vDataDes.count);
    if (ReadExact(fp, vDataDes.displs.data(), sizeof(uint64_t), vDataDes.count)) {
        HCCL_VM_ERROR("Read file vDataDes displs failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    vDataDes.counts.resize(vDataDes.count);
    if (ReadExact(fp, vDataDes.counts.data(), sizeof(uint64_t), vDataDes.count)) {
        HCCL_VM_ERROR("Read file vDataDes counts failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult All2AllDataDesTagWrite(FILE* fp, const All2AllDataDesTagInner& all2AllDataDes)
{
    if (WriteExact(fp, &all2AllDataDes.sendType, sizeof(all2AllDataDes.sendType), 1)) {
        HCCL_VM_ERROR("Write file all2allDataDes sendType failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, &all2AllDataDes.recvType, sizeof(all2AllDataDes.recvType), 1)) {
        HCCL_VM_ERROR("Write file all2allDataDes recvType failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, &all2AllDataDes.sendCount, sizeof(all2AllDataDes.sendCount), 1)) {
        HCCL_VM_ERROR("Write file all2allDataDes sendCount failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, &all2AllDataDes.recvCount, sizeof(all2AllDataDes.recvCount), 1)) {
        HCCL_VM_ERROR("Write file all2allDataDes recvCount failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, &all2AllDataDes.count, sizeof(all2AllDataDes.count), 1)) {
        HCCL_VM_ERROR("Write file all2allDataDes count failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, all2AllDataDes.sendCountMatrix.data(), sizeof(uint64_t), all2AllDataDes.count)) {
        HCCL_VM_ERROR("Write file all2allDataDes sendCountMatrix failed. count= {}", all2AllDataDes.count);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult All2AllDataDesTagRead(FILE* fp, All2AllDataDesTagInner& all2AllDataDes)
{
    if (ReadExact(fp, &all2AllDataDes.sendType, sizeof(all2AllDataDes.sendType), 1)) {
        HCCL_VM_ERROR("Read file all2allDataDes sendType failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (ReadExact(fp, &all2AllDataDes.recvType, sizeof(all2AllDataDes.recvType), 1)) {
        HCCL_VM_ERROR("Read file all2allDataDes recvType failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (ReadExact(fp, &all2AllDataDes.sendCount, sizeof(all2AllDataDes.sendCount), 1)) {
        HCCL_VM_ERROR("Read file all2allDataDes sendCount failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (ReadExact(fp, &all2AllDataDes.recvCount, sizeof(all2AllDataDes.recvCount), 1)) {
        HCCL_VM_ERROR("Read file all2allDataDes recvCount failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (ReadExact(fp, &all2AllDataDes.count, sizeof(all2AllDataDes.count), 1)) {
        HCCL_VM_ERROR("Read file all2allDataDes count failed. ");
        HCCL_VM_ERROR("Read file all2allDataDes count failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    all2AllDataDes.sendCountMatrix.resize(all2AllDataDes.count);
    if (ReadExact(fp, all2AllDataDes.sendCountMatrix.data(), sizeof(uint64_t), all2AllDataDes.count)) {
        HCCL_VM_ERROR("Read file all2allDataDes sendCountMatrix failed. count= {}", all2AllDataDes.count);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ModelInfoWrite(FILE* fp, const ModelInfoInner& modelInfo)
{
    auto ret = ModelInfoCommWrite(fp, modelInfo.comm);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Write model info comm data failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = VDataDesTagWrite(fp, modelInfo.vDataDes);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Write model info vdatades data failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = All2AllDataDesTagWrite(fp, modelInfo.all2AllDataDes);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Write model info all2allDataDes data failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ModelInfoRead(FILE* fp, ModelInfoInner& modelInfo)
{
    auto ret = ModelInfoCommRead(fp, modelInfo.comm);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Read model info comm data failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = VDataDesTagRead(fp, modelInfo.vDataDes);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Read model info vdatades data failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    ret = All2AllDataDesTagRead(fp, modelInfo.all2AllDataDes);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Read model info all2allDataDes data failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ChannelInfoWrite(FILE* fp, const ChannelInfoInner& chInfo)
{
    if (WriteExact(fp, &chInfo.count, sizeof(chInfo.count), 1)) {
        HCCL_VM_ERROR("Write channel info count failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, chInfo.data.data(), sizeof(ChannelData), chInfo.count)) {
        HCCL_VM_ERROR("Write channel info data failed. count= {}", chInfo.count);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult ChannelInfoRead(FILE* fp, ChannelInfoInner& chInfo)
{
    if (ReadExact(fp, &chInfo.count, sizeof(chInfo.count), 1)) {
        HCCL_VM_ERROR("Read channel info count failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    chInfo.data.resize(chInfo.count);
    if (ReadExact(fp, chInfo.data.data(), sizeof(ChannelData), chInfo.count)) {
        HCCL_VM_ERROR("Read channel info data failed. count= {}", chInfo.count);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult JettyInfoWrite(FILE* fp, const JettyInfoInner& jettyInfo)
{
    if (WriteExact(fp, &jettyInfo.count, sizeof(jettyInfo.count), 1)) {
        HCCL_VM_ERROR("Write jetty info count failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, jettyInfo.data.data(), sizeof(JettyData), jettyInfo.count)) {
        HCCL_VM_ERROR("Write jetty info data failed. count= {}", jettyInfo.count);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult JettyInfoRead(FILE* fp, JettyInfoInner& jettyInfo)
{
    if (ReadExact(fp, &jettyInfo.count, sizeof(jettyInfo.count), 1)) {
        HCCL_VM_ERROR("Read jetty info count failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    jettyInfo.data.resize(jettyInfo.count);
    if (ReadExact(fp, jettyInfo.data.data(), sizeof(JettyData), jettyInfo.count)) {
        HCCL_VM_ERROR("Read jetty info data failed. count= {}", jettyInfo.count);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult MemLayoutWrite(FILE* fp, const MemLayoutInfoInner& memLayoutInfo)
{
    if (WriteExact(fp, &memLayoutInfo.count, sizeof(memLayoutInfo.count), 1)) {
        HCCL_VM_ERROR("Write mem layout info count failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, memLayoutInfo.data.data(), sizeof(MemLayoutData), memLayoutInfo.count)) {
        HCCL_VM_ERROR("Write mem layout info data failed. count= {}", memLayoutInfo.count);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult MemLayoutRead(FILE* fp, MemLayoutInfoInner& memLayoutInfo)
{
    if (ReadExact(fp, &memLayoutInfo.count, sizeof(memLayoutInfo.count), 1)) {
        HCCL_VM_ERROR("Read mem layout info count failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    memLayoutInfo.data.resize(memLayoutInfo.count);
    if (ReadExact(fp, memLayoutInfo.data.data(), sizeof(MemLayoutData), memLayoutInfo.count)) {
        HCCL_VM_ERROR("Read mem layout info data failed. count= {}", memLayoutInfo.count);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult MicrocodeInstrWrite(FILE* fp, const MicrocodeInstrInner& mcInstrInfo)
{
    if (WriteExact(fp, &mcInstrInfo.desc, sizeof(mcInstrInfo.desc), 1)) {
        HCCL_VM_ERROR("Write microcode instr desc failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, mcInstrInfo.data.data(), sizeof(hcomm::CcuRep::CcuInstr), mcInstrInfo.desc.count)) {
        HCCL_VM_ERROR("Write microcode instr data failed. count= {}", mcInstrInfo.desc.count);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult MicrocodeInstrRead(FILE* fp, MicrocodeInstrInner& mcInstrInfo)
{
    if (ReadExact(fp, &mcInstrInfo.desc, sizeof(mcInstrInfo.desc), 1)) {
        HCCL_VM_ERROR("Read microcode instr desc failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    mcInstrInfo.data.resize(mcInstrInfo.desc.count);
    if (ReadExact(fp, mcInstrInfo.data.data(), sizeof(hcomm::CcuRep::CcuInstr), mcInstrInfo.desc.count)) {
        HCCL_VM_ERROR("Read microcode instr data failed. count= {}", mcInstrInfo.desc.count);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult TaskMetaWrite(FILE* fp, const HcclTaskMetaData& taskData)
{
    if (WriteExact(fp, &taskData, sizeof(HcclTaskMetaData), 1)) {
        HCCL_VM_ERROR("Write task meta failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult TaskMetaRead(FILE* fp, HcclTaskMetaData& taskData)
{
    if (ReadExact(fp, &taskData, sizeof(HcclTaskMetaData), 1)) {
        HCCL_VM_ERROR("Read task meta failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

// task读写
HcclVmResult HcclVmTaskMetaDataWrite(FILE* fp, const HcclVmTaskMetaData& taskMeta)
{
    auto ret = FileHeaderWrite(fp, taskMeta.header);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Write file header failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, taskMeta.task_meta.data(), sizeof(HcclTaskMetaData), taskMeta.header.count)) {
        HCCL_VM_ERROR("Write task meta failed. count= {}", taskMeta.header.count);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclVmTaskMetaDataRead(FILE* fp, HcclVmTaskMetaData& taskMeta, uint32_t magic)
{
    auto ret = FileHeaderRead(fp, taskMeta.header, magic);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Read file header failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    taskMeta.task_meta.resize(taskMeta.header.count);
    if (ReadExact(fp, taskMeta.task_meta.data(), sizeof(HcclTaskMetaData), taskMeta.header.count)) {
        HCCL_VM_ERROR("Read task meta failed. count= {}", taskMeta.header.count);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult HcclVmFlagDataWrite(FILE* fp, const HcclVmFlagData& flagData)
{
    auto ret = FileHeaderWrite(fp, flagData.header);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Write file header failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (WriteExact(fp, &flagData.runner_status, sizeof(uint16_t), 1)) {
        HCCL_VM_ERROR("Write runner status failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

bool CheckVmFlagDataFileSize(FILE* fp)
{
    struct stat st;
    if (fstat(fileno(fp), &st) != 0) {
        HCCL_VM_ERROR("Check file status failed. ");
        return false;
    }
    // FileHeader(20) + runner_status(2) = 22
    if (st.st_size < static_cast<off_t>(sizeof(FileHeader) + sizeof(uint16_t))) {
        HCCL_VM_ERROR("File size too small: {}", st.st_size);
        return false;
    }

    return true;
}

HcclVmResult HcclVmFlagDataRead(FILE* fp, HcclVmFlagData& flagData, uint32_t magic)
{
    if (!CheckVmFlagDataFileSize(fp)) {
        flagData.runner_status = 0;
        return HcclVmResult::HCCL_SIM_SUCCESS;
    }

    auto ret = FileHeaderRead(fp, flagData.header, magic);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("Read file header failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    if (ReadExact(fp, &flagData.runner_status, sizeof(uint16_t), 1)) {
        HCCL_VM_ERROR("Read runner status failed. ");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}
} // namespace HcclSim
