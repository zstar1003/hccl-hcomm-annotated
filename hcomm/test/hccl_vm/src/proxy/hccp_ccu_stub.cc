/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "CCU_STUB"

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <vector>
#include "sim_ccu_channel_ctx.h"
#include "ccu_common.h"
#include "sim_ccu_jetty_ctx.h"
#include "ccu_microcode_v1.h"
#include "ccu_u_comm.h"
#include "dtype_common.h"
#include "store_sim_store_pub.h"
#include "sim_log.h"
#include "sim_common_api.h"
#include "sim_yaml_config.h"
#include "hccp_common.h"
#include "hccp_ctx.h"
#include "hccp_tlv.h"
#include "sim_ip_address.h"
#include "rt_external_kernel.h"
#include "db_sim_op_db_ops.h"
#include "db_sim_runner_common.h"
#include "db_sim_runner_ops.h"
#include "sim_common_defs.h"

extern uint64_t g_cur_server_key;
extern thread_local std::string g_currentOpFastLaunchTag;

using namespace HcclSim;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

int GetEnableCcuDie(channel_info_out* output, uint8_t dieId)
{
    HCCL_VM_INFO("dieId:{}.", dieId);
    output->data.data_info.data_array[0].dieinfo.enable_flag = 1;
    return 0;
}

void SetCcuV1ResourceBasicInfo(channel_info_out* output, uint8_t dieId, uint32_t devId)
{
    // todo: 后续根据dieId，设置不同的resourceAddr
    if (dieId == 0) {
        output->data.data_info.data_array[0].baseinfo.resourceAddr = (void*)0x123123123;
        // RunnerDB::Update<sim::SimModelData>(simModelKey, [](sim::SimModelData &smd) { smd.ccu0_resource_base_addr =
        // 0x123123123; });
    } else {
        output->data.data_info.data_array[0].baseinfo.resourceAddr = (void*)0x456456456;
        // RunnerDB::Update<sim::SimModelData>(simModelKey, [](sim::SimModelData &smd) { smd.ccu0_resource_base_addr =
        // 0x456456456; });
    }
    output->data.data_info.data_array[0].baseinfo.missionKey = 0;
    output->data.data_info.data_array[0].baseinfo.ms_id = 3; //
    uint32_t instructionNum = 0x8000;                        // Instruction 32k
    uint32_t missionNum = 16;                                // Mission ctx 16
    uint32_t loopEngineNum = 200;                            // Loop ctx 200
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap0
        = (instructionNum - 1) | ((missionNum - 1) << MOVE_TOW_BYTES) | ((loopEngineNum - 1) << MOVE_THREE_BYTES);
    uint32_t gsaNum = 3072; // GSA 3072
    uint32_t xnNum = 3072;  // Xn 3072
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap1 = ((xnNum - 1) << MOVE_TOW_BYTES) | (gsaNum - 1);
    uint32_t ckeNum = 1024; // Checlist Entry(CKE) 1024
    uint32_t msNum = 1536;  // MemorySlice(MS) 1536
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap2 = ((msNum - 1) << MOVE_TOW_BYTES) | (ckeNum - 1);
    uint32_t channelNum = 128; // Channel map 128 for v1
    uint32_t jettyNum = 128;   // Jetty context 128
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap3
        = ((jettyNum - 1) << MOVE_TOW_BYTES) | (channelNum - 1);
    uint32_t pfeNum = 16; // PFE 16 for v1
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap4 = (pfeNum - 1) & 0x000000FF;
}

void SetCcuV2ResourceBasicInfo(channel_info_out* output, uint8_t dieId, uint32_t devId)
{
    // todo: 后续根据dieId，设置不同的resourceAddr
    if (dieId == 0) {
        output->data.data_info.data_array[0].baseinfo.resourceAddr = (void*)0x123123123;
        // RunnerDB::Update<sim::SimModelData>(simModelKey, [](sim::SimModelData &smd) { smd.ccu0_resource_base_addr =
        // 0x123123123; });
    } else {
        output->data.data_info.data_array[0].baseinfo.resourceAddr = (void*)0x456456456;
        // RunnerDB::Update<sim::SimModelData>(simModelKey, [](sim::SimModelData &smd) { smd.ccu0_resource_base_addr =
        // 0x456456456; });
    }
    output->data.data_info.data_array[0].baseinfo.missionKey = 0;
    output->data.data_info.data_array[0].baseinfo.ms_id = 3; //
    uint32_t instructionNum = 0x8000;                        // Instruction 32k
    uint32_t missionNum = 16;                                // Mission ctx 16
    uint32_t loopEngineNum = 512;                            // Loop ctx 512
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap0
        = ((instructionNum - 1) & 0xFFFF) | (((missionNum - 1) & 0xF) << 16) | (((loopEngineNum - 1) & 0xFFF) << 20);
    uint32_t gsaNum = 0;    // GSA 0 for v2
    uint32_t xnNum = 16384; // Xn 16384
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap1 = ((xnNum - 1) << MOVE_TOW_BYTES);
    uint32_t ckeNum = 1024; // Checlist Entry(CKE) 1024
    uint32_t msNum = 1536;  // MemorySlice(MS) 1536
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap2 = ((msNum - 1) << MOVE_TOW_BYTES) | (ckeNum - 1);
    uint32_t channelNum = 1024; // Channel map 1024 for v2
    uint32_t jettyNum = 128;    // Jetty context 128
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap3
        = ((jettyNum - 1) << MOVE_TOW_BYTES) | (channelNum - 1);
    uint32_t pfeNum = 20; // PFE 20 for v2
    output->data.data_info.data_array[0].baseinfo.caps.lqc_ccu_cap4 = (pfeNum - 1) & 0x000000FF;
}

int SetCcuResourceBasicInfo(channel_info_out* output, uint8_t dieId, uint32_t devId)
{
    sim::Device device{};
    if (GetDeviceByPhysicalId(devId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by logic id {} failed.", devId);
        return -1;
    }
    if (strcmp(device.soc_version, "Ascend950") == 0) {
        SetCcuV1ResourceBasicInfo(output, dieId, devId);
    } else if (strcmp(device.soc_version, "Ascend960") == 0) {
        SetCcuV2ResourceBasicInfo(output, dieId, devId);
    } else {
        HCCL_VM_ERROR("unknown device version: {} for devId: {}", device.soc_version, devId);
        return -1;
    }
    return 0;
}
extern void* GetRealPtrByAddr(const void* devPtr);

void DumpInstrDecToFile(uint32_t instrCnt, const hcomm::CcuRep::CcuInstr* instrData, const std::string& fileName)
{
    std::ofstream ofs(fileName, std::ios::out | std::ios::trunc);
    ofs << "ccu total instruction number: " << instrCnt << "\n";
    for (uint32_t idx = 0; idx < instrCnt; idx++) {
        ofs << "[InstrData][ " + std::to_string(idx) + "]" + hcomm::CcuRep::ParseInstr(&instrData[idx]) + "\n";
    }
    return;
}

void DumpCcuSqeToFile(
    uint32_t startId, uint32_t instrCnt, uint32_t argSize, uint64_t args[], const std::string& fileName)
{
    std::ofstream ofs(fileName, std::ios::out | std::ios::trunc);
    ofs << "ccu sqe info: startInstrId= " << startId << ", instrCnt= " << instrCnt << ", argSize= " << argSize << "\n";
    for (uint32_t idx = 0; idx < argSize; idx++) {
        ofs << "[SQE Arg][" << idx << "]: " << args[idx] << "\n";
    }
    return;
}

namespace fs = std::filesystem;

bool write_or_overwrite_in_cwd(const std::string& filename, const std::string& data)
{
    fs::path target = fs::path(filename).is_absolute() ? fs::path(filename) : fs::current_path() / filename;

    std::error_code ec;
    bool exists = fs::exists(target, ec);
    if (ec) {
        // 读取状态出错，视为失败
        return false;
    }

    std::ofstream ofs;
    if (exists) {
        ofs.open(target, std::ios::out | std::ios::app);
    } else {
        ofs.open(target, std::ios::out | std::ios::trunc);
    }

    if (!ofs.is_open()) {
        return false;
    }

    if (exists) {
        ofs << "\n\n";
    }

    ofs << data;
    ofs.flush();
    return true;
}

// input->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_INSTRUCTION
int LoadMicrocodeInstructionStub(uint32_t devId, uint8_t dieId, const channel_info_in* input)
{
    HCCL_VM_INFO("enter LoadMicrocodeInstruction .....");
    if (dieId >= DIE_NUM) {
        HCCL_VM_ERROR("wrong param of die id: {}", dieId);
        return -1;
    }

    sim::Device device{};
    if (GetDeviceByPhysicalId(devId, device) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by logic id {} failed.", devId);
        return -1;
    }
    sim::Ccu ccu{};
    if (GetCcuFromDeviceByDieId(device.id, dieId, ccu) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get ccu from device by die id {} failed.", dieId);
        return -1;
    }
    sim::CcuResource ccuRes;
    if (GetCcuResourceByCcu(ccu.id, ccuRes) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get ccu resource by ccu {} failed.", ccu.id);
        return -1;
    }

    auto devKey = device.id;
    auto rank = RunnerDB::GetOneByPred<sim::Rank>([devKey](const sim::Rank& r) {
        return r.device_id == devKey;
    });
    if (!rank.second) {
        HCCL_VM_ERROR("can not find any rank");
        return -1;
    }
    auto rankId = rank.first.rank_id;
    auto startId = input->offset_start;
    auto instrInfoSize = input->data.data_info.data_array_len;
    auto instrCnt = instrInfoSize / sizeof(hcomm::CcuRep::CcuInstr);

    uint64_t ccuId = ccu.id;
    auto instrOffset = startId * sizeof(hcomm::CcuRep::CcuInstr);
    auto ccuDataTmp = (ccu_data_type_union)(input->data.data_info.data_array[0]);
    auto instructionData
        = reinterpret_cast<hcomm::CcuRep::CcuInstr*>(GetRealPtrByAddr((void*)ccuDataTmp.insinfo.resourceAddr));
    if (instructionData == nullptr) {
        HCCL_VM_ERROR("get ccu instruction data by resourceAddr failed  addr:0x{:x}", ccuDataTmp.insinfo.resourceAddr);
        return -1;
    }
    if (sim::UpdateAndInsertByCcuId(ccuId, devKey, rankId, dieId, instrCnt, instrOffset, instrInfoSize, instructionData)
        != 0) {
        HCCL_VM_ERROR("update ccu failed");
        return -1;
    }

    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    std::ostringstream fileName;
    fileName << "mc_instr_info_rank_" << rankId << "_die_" << static_cast<uint32_t>(dieId) << ".txt";
    std::ostringstream mcData;
    mcData << "ccu total instruction number: " << instrCnt << "\n";
    if (strcmp(device.soc_version, "Ascend950") == 0) {
        for (uint32_t idx = 0; idx < instrCnt; idx++) {
            mcData << "[InstrData][ " + std::to_string(startId + idx) + "]"
                          + hcomm::CcuRep::ParseInstr(&instructionData[idx]) + "\n";
        }
#ifdef BUILD_A6_CCU_INSTR
    } else if (strcmp(device.soc_version, "Ascend960") == 0) {
        for (uint32_t idx = 0; idx < instrCnt; idx++) {
            mcData << "[InstrData][ " + std::to_string(startId + idx) + "]"
                          + hcomm::CcuRep::CcuV2::ParseInstrV2(&instructionData[idx]) + "\n";
        }
#endif
    } else {
        HCCL_VM_ERROR("not support device soc version: {:s}", device.soc_version);
        return HCCL_E_NOT_SUPPORT;
    }

    auto status = write_or_overwrite_in_cwd(InstallPath::ResolveToInstallRoot("data/" + fileName.str()), mcData.str());

    sim::CcuInstrTab instr{};
    instr.id = 0;
    instr.ccuInstrResId = ccuId;
    instr.rankId = rankId;
    instr.startId = startId;
    instr.instrInfoSize = instrInfoSize;
    if (sim::InsertCcuInstr(instr) != 0) {
        HCCL_VM_ERROR("insert instr failed");
        return -1;
    }
    return 0;
}

uint64_t GetRemoteCcuVa(const ChannelCtxDataV1& chDataTmp)
{
    uint64_t dstVa = 0;
    dstVa |= (uint64_t)(chDataTmp.dstVaLow & MASK_VA_LOW);                         // 低 8 位
    dstVa |= ((uint64_t)(chDataTmp.dstVaMiddle & MASK_VA_MID) << SHIFT_8BITS);     // 位 8-23
    dstVa |= ((uint64_t)(chDataTmp.dstVaHigh & MASK_VA_HIGH) << SHIFT_24BITS);     // 位 24-39
    dstVa |= ((uint64_t)(chDataTmp.dstVaHigher & MASK_VA_HIGHER) << SHIFT_40BITS); // 位 40+

    return dstVa << REMOTE_CCU_VA_RIGHT_SHIFT_NUM;
}

int GetLocalEndPointByJetty(uint64_t jettyId, uint16_t dieId, sim::EndPoint& endPoint)
{
    auto localJetty = RunnerDB::GetOneByPred<sim::RaJetty>([jettyId, dieId](const sim::RaJetty& jetty) {
        return jetty.jetty_id == jettyId && jetty.pid == getpid() && jetty.dieId == dieId;
    });
    if (!localJetty.second) {
        HCCL_VM_ERROR("can not find jetty {} die:{} in local jetty map", jettyId, dieId);
        return -1;
    }

    auto ctxRes = RunnerDB::GetById<sim::RaContext>(localJetty.first.ctx_handle);
    if (!ctxRes.has_value()) {
        HCCL_VM_ERROR("can not find context {} in local context map", localJetty.first.ctx_handle);
        return -1;
    }

    auto localEp = ctxRes->endpoint_id;
    auto endPointOpt = RunnerDB::GetById<sim::EndPoint>(localEp);
    if (!endPointOpt.has_value()) {
        HCCL_VM_ERROR("can not find endpoint:{:d}", localEp);
        return -1;
    }
    endPoint = *endPointOpt;
    return 0;
}

// 配置channel信息：input->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_CHANNEL
int ConfigChannelInfo(channel_info_in* input, uint32_t deviceId)
{
    sim::Device locDevice{};
    if (GetDeviceByPhysicalId(deviceId, locDevice) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by physic id {} failed.", deviceId);
        return -1;
    }
    uint8_t dieId = input->data.data_info.udie_idx;
    uint32_t chId = input->offset_start;

    // 配置channel信息：input->op == Hccl::CcuOpcodeType::CCU_U_OP_SET_CHANNEL
    ChannelCtxDataV1 chDataTmp;
    memcpy(&chDataTmp, input->data.data_info.data_array, sizeof(struct ChannelCtxDataV1));
    Eid eid{};
    for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
        eid.raw[i] = chDataTmp.eidRaw[URMA_EID_LEN - i - 1];
    }
    IpAddress addr(eid);
    auto ipAddr = addr.GetIpStr().substr(2);
    if (ipAddr == "") {
        return 0;
    }

    sim::EndPoint rmtEndPoint{};
    if (GetEndPointByIpAddr(ipAddr, rmtEndPoint) != 0) {
        HCCL_VM_ERROR("Get remote endpoint failed. ip:{}, eid:{}", ipAddr.c_str(), addr.EidToHexString());
        return -1;
    }

    HCCL_VM_INFO(
        "channel info: loc phyId: {:d}, loc devKey: {:d}, loc dieId: {:d}, chId: {:d}, rmt devKey: {:d}, rmt dieId: "
        "{:d}, rmt ip: {}",
        deviceId, locDevice.id, static_cast<uint32_t>(dieId), chId, rmtEndPoint.device_id,
        static_cast<uint32_t>(chDataTmp.ioDieId), ipAddr);

    auto locDevKey = locDevice.id;
    auto locRank = RunnerDB::GetOneByPred<sim::Rank>([locDevKey](const sim::Rank& r) {
        return r.device_id == locDevKey;
    });
    if (!locRank.second) {
        HCCL_VM_ERROR("can not find loc rank by device key: {}", locDevKey);
        return -1;
    }

    auto rmtDevKey = rmtEndPoint.device_id;
    auto rmtRank = RunnerDB::GetOneByPred<sim::Rank>([rmtDevKey](const sim::Rank& r) {
        return r.device_id == rmtDevKey;
    });
    if (!rmtRank.second) {
        HCCL_VM_ERROR("can not find rmt rank by device key: {}", rmtDevKey);
        return -1;
    }
    uint16_t srcJettyId{0};
    uint16_t srcDieId{0};
    DevType devType;
    hrtGetDeviceType(devType);
    if (devType == DevType::DEV_TYPE_950) {
        srcJettyId = (uint16_t)((chDataTmp.startJettyIdHigh << 4) | chDataTmp.startJettyIdLow);
        srcDieId = chDataTmp.ioDieId;
#ifdef BUILD_A6_CCU_INSTR
    } else if (devType == DevType::DEV_TYPE_960) {
        srcJettyId = chId / 8 + 1024;
        ChannelDataV2* chnV2 = (ChannelDataV2*)input->data.data_info.data_array;
        srcDieId = chnV2->ioDieId;
#endif
    }
    auto jettyNum = (uint16_t)((chDataTmp.jettyNumHigh << 4) | chDataTmp.jettyNumLow);
    // 根据endPointPair获取src eid
    sim::EndPoint localEndPoint{};
    if (GetLocalEndPointByJetty(srcJettyId, srcDieId, localEndPoint) != 0) {
        HCCL_VM_ERROR("can not find local endPoint by jettyId: {}", srcJettyId);
        return -1;
    }

    HCCL_VM_INFO(
        "add chn:{:d}, srcJetty:{:d},jettyNum:{:d}, srcAddr:{}, dstAddr:{}, {:d}<-->{:d}", chId, srcJettyId, jettyNum,
        localEndPoint.ip_addr, rmtEndPoint.ip_addr, localEndPoint.id, rmtEndPoint.id);

    sim::CcuChannelTab ccuChannelTab{};
    ccuChannelTab.id = 0;
    ccuChannelTab.channelId = chId;
    ccuChannelTab.srcDieId = localEndPoint.die_id;
    ccuChannelTab.dstDieId = rmtEndPoint.die_id;
    ccuChannelTab.srcRankId = sim::GetRankIdByDeviceId(localEndPoint.device_id);
    ccuChannelTab.dstRankId = sim::GetRankIdByDeviceId(rmtEndPoint.device_id);
    memcpy(ccuChannelTab.leid, &localEndPoint.eid, sizeof(localEndPoint.eid));
    memcpy(ccuChannelTab.reid, &rmtEndPoint.eid, sizeof(rmtEndPoint.eid));
    ccuChannelTab.protocol = 0;
    ccuChannelTab.jettyNum = jettyNum + 1;
    for (uint32_t i = 0; i < ccuChannelTab.jettyNum; i++) {
        ccuChannelTab.jettyId[i] = srcJettyId + i;
    }
    auto ret = sim::InsertCcuChannel(ccuChannelTab);
    if (ret != 0) {
        HCCL_VM_ERROR("insert ccu channel table failed for channel id: {}", chId);
        return -1;
    }

    return 0;
}

int ConfigJettyInfo(channel_info_in* input, uint32_t deviceId)
{
    HCCL_VM_INFO("Enter into config jetty info...");
    uint8_t dieId = input->data.data_info.udie_idx;
    uint32_t jettyNum = input->data.data_info.data_array_size;
    uint32_t startJettyCtxId = input->offset_start;

    std::vector<LocalJettyCtxData> jettyCtxData;
    jettyCtxData.resize(jettyNum);
    for (size_t i = 0; i < jettyNum; i++) {
        (void)memcpy(&jettyCtxData[i], &input->data.data_info.data_array[i], sizeof(LocalJettyCtxData));
    }

    for (auto& tmp : jettyCtxData) {
        HCCL_VM_DEBUG(
            "doorbellAddr: [3]0x{:04x}, [2]0x{:04x}, [1]0x{:04x}, [0]0x{:04x}",
            tmp.doorbellAddr[3], // 3: doorbell ��ַ����
            tmp.doorbellAddr[2], // 2: doorbell ��ַ����
            tmp.doorbellAddr[1], tmp.doorbellAddr[0]);

        // ��ȫ���⣺��ֹ��ӡtoken�����Ϣ
        HCCL_VM_DEBUG(
            "pfeIdx: 0x{:04x}, ioDieId: 0x{:04x}, doorbellAddrType: 0x{:04x}, tokenValueIsValid: 0x{:04x}",
            static_cast<uint16_t>(tmp.pfeIdx), static_cast<uint16_t>(tmp.ioDieId),
            static_cast<uint16_t>(tmp.doorbellAddrType), static_cast<uint16_t>(tmp.tokenValueIsValid));

        HCCL_VM_DEBUG(
            "sqeBasicBlockLeftShifts: 0x{:04x}, pi: 0x{:04x}, ci: 0x{:04x}, "
            "maxCi: 0x{:04x}, oooCqeCnt: 0x{:04x}, startWqeBasicBlockIdxLow: 0x{:04x}, "
            "startWqeBasicBlockIdxHigh: 0x{:04x}, doorbellSendState: 0x{:04x}",
            static_cast<uint16_t>(tmp.sqeBasicBlockLeftShifts), tmp.pi, tmp.ci, tmp.maxCi,
            static_cast<uint16_t>(tmp.oooCqeCnt), static_cast<uint16_t>(tmp.startWqeBasicBlockIdxLow),
            static_cast<uint16_t>(tmp.startWqeBasicBlockIdxHigh), static_cast<uint16_t>(tmp.doorbellSendState));
    }

    return 0;
}

int GetCcuVersion(channel_info_out* output, uint32_t deviceId)
{
    sim::Device locDevice{};
    if (GetDeviceByPhysicalId(deviceId, locDevice) != ACL_SUCCESS) {
        HCCL_VM_ERROR("get device by physic id {} failed.", deviceId);
        return -1;
    }
    if (strcmp(locDevice.soc_version, "Ascend950") == 0) {
        output->data.data_info.data_array[0].version = static_cast<ccu_version_e>(CcuVersion::CCU_V1);
    } else if (strcmp(locDevice.soc_version, "Ascend960") == 0) {
        output->data.data_info.data_array[0].version = static_cast<ccu_version_e>(CcuVersion::CCU_V2);
    } else {
        HCCL_VM_ERROR("unknown soc version: {}", locDevice.soc_version);
        return -1;
    }
    return 0;
}

int SimRaCustomChannel(void* tlvHandle, struct TlvMsg* sendMsg, struct TlvMsg* recvMsg)
{
    HCCL_VM_INFO("Enter into custom channel...");
    auto in = reinterpret_cast<struct channel_info_in*>(sendMsg->data);
    auto out = reinterpret_cast<struct channel_info_out*>(recvMsg->data);

    auto tlvId = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(tlvHandle));
    auto curTlv = RunnerDB::GetById<sim::RaTlv>(tlvId);
    if (!curTlv.has_value()) {
        HCCL_VM_ERROR("tlv id not found:{:d}", tlvId);
        return ACL_ERROR_INVALID_PARAM;
    }

    uint8_t dieId = in->data.data_info.udie_idx;
    uint32_t devId = curTlv->physical_id;

    switch (in->op) {
        case ccu_u_opcode_t::CCU_U_OP_GET_DIE_WORKING:
            return GetEnableCcuDie(out, dieId);
        case ccu_u_opcode_t::CCU_U_OP_GET_BASIC_INFO:
            return SetCcuResourceBasicInfo(out, dieId, devId);
        case ccu_u_opcode_t::CCU_U_OP_SET_INSTRUCTION:
            return LoadMicrocodeInstructionStub(devId, dieId, in);
        case ccu_u_opcode_t::CCU_U_OP_SET_CHANNEL:
            return ConfigChannelInfo(in, devId);
        case ccu_u_opcode_t::CCU_U_OP_SET_JETTY_CTX:
            return ConfigJettyInfo(in, devId);
        case ccu_u_opcode_t::CCU_U_OP_GET_VERSION:
            return GetCcuVersion(out, devId);
        default:
            break;
    }

    return 0;
}

int RaCtxGetAsyncEvents(void* ctxHandle, struct AsyncEvent events[], unsigned int* num)
{
    sleep(1);
    *num = 0;
    return 0;
}

int GetAllUsedEndPoint(uint32_t phyDevId, std::vector<sim::EndPoint>& endPoints)
{
    sim::Device device{};
    if (GetDeviceByPhysicalId(phyDevId, device) != 0) {
        HCCL_VM_ERROR("can not find device by id:{:d}", phyDevId);
        return -1;
    }

    auto deviceId = device.id;
    endPoints = RunnerDB::GetByPred<sim::EndPoint>([deviceId](const sim::EndPoint& ep) {
        return ep.device_id == deviceId && ep.status == 1;
    });
    return 0;
}

int RaGetDevEidInfoNum(struct RaInfo info, unsigned int* num)
{
    std::vector<sim::EndPoint> endPoints;
    if (GetAllUsedEndPoint(info.phyId, endPoints) != 0) {
        return -1;
    }
    HCCL_VM_INFO("return success {:d}", endPoints.size());
    *num = endPoints.size();
    return 0;
}

int RaGetDevEidInfoList(struct RaInfo info, struct HccpDevEidInfo infoList[], unsigned int* num)
{
    std::vector<sim::EndPoint> endPoints;
    if (GetAllUsedEndPoint(info.phyId, endPoints) != 0) {
        HCCL_VM_ERROR("get all EndPoint failed phyId:{:d}", info.phyId);
        return -1;
    }

    for (uint32_t idx = 0; idx < *num; idx++) {
        infoList[idx].type = 0;
        infoList[idx].eidIndex = 0;
        infoList[idx].funcId = endPoints[idx].func_id;
        infoList[idx].chipId = info.phyId; // todo: 单server, logic id与rank id相等，但多server此处有问题。
        infoList[idx].dieId = endPoints[idx].die_id;
        memcpy(infoList[idx].eid.raw, endPoints[idx].eid, sizeof(endPoints[idx].eid));

        Eid eid{};
        for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
            eid.raw[i] = endPoints[idx].eid[i];
        }
        IpAddress addr(eid);
        auto ipAddr = addr.GetIpStr().substr(2);
        HCCL_VM_INFO("ipAddr: {}, phyId: {}, serverKey=: {}", ipAddr, info.phyId, g_cur_server_key);
    }

    return 0;
}

int rtCCULaunch(rtCcuTaskInfo_t* taskInfo, rtStream_t const stream)
{
    uint64_t streamId = sim::GetCurrentStreamId((uint64_t)(uintptr_t)stream);

    uint32_t curRank = (uint32_t)sim::GetCurrRankId();

    HcclTaskMetaData taskMetaData;
    taskMetaData.taskType = HccLTaskMetaType::CCU_GRAPH;
    taskMetaData.commId = 0;
    taskMetaData.streamId = streamId;
    taskMetaData.rankId = curRank;

    memcpy(&taskMetaData.taskData.ccu, taskInfo, sizeof(rtCcuTaskInfo_t));

    fs::create_directories(fs::path(InstallPath::ResolveToInstallRoot("data")));
    std::ostringstream fileName;
    fileName << "sqe_info_rank_" << curRank << "_die_" << static_cast<uint32_t>(taskInfo->dieId) << "_mission_"
             << static_cast<uint32_t>(taskInfo->missionId) << "_startId_" << taskInfo->instStartId << ".txt";
    std::ostringstream sqeData;
    sqeData << "ccu sqe info: startInstrId= " << taskInfo->instStartId << ", instrCnt= " << taskInfo->instCnt
            << ", argSize= " << taskInfo->argSize << "\n";
    for (uint32_t idx = 0; idx < taskInfo->argSize; idx++) {
        sqeData << "[SQE Arg][" << idx << "]: " << taskInfo->args[idx] << "\n";
    }
    auto status = write_or_overwrite_in_cwd(InstallPath::ResolveToInstallRoot("data/" + fileName.str()), sqeData.str());

    uint32_t index{0};
    auto ret = InsertTaskToCollection(&taskMetaData, &index);
    if (ret != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("insert task fail");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    return 0;
}

#ifdef __cplusplus
}
#endif // __cplusplus
