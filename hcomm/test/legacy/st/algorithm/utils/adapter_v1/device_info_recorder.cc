/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "device_info_recorder.h"
#include "transformer.h"
namespace hccl {

DeviceInfoRecorder* DeviceInfoRecorder::Global()
{
    static DeviceInfoRecorder* rankInfoRecorder = new DeviceInfoRecorder;
    return rankInfoRecorder;
}

void DeviceInfoRecorder::Reset()
{
    rankId2devType.clear();
    rankId2superdeviceId.clear();
    return;
}

void DeviceInfoRecorder::InitDeviceInfo(TopoMeta topoMeta, RankTable_t& rankTable, CheckerDevType uniDevType)
{
    u32 myRankId = 0;
    for (int i = 0; i < topoMeta.size(); i++) {
        for (int j = 0; j < topoMeta[i].size(); j++) {
            for (int k = 0; k < topoMeta[i][j].size(); k++) {
                if (g_HcclDevType2CheckerDevType[rankTable.rankList[myRankId].deviceInfo.deviceType]
                    != CheckerDevType::DEV_TYPE_NOSOC) {
                    rankId2devType[myRankId]
                        = g_HcclDevType2CheckerDevType[rankTable.rankList[myRankId].deviceInfo.deviceType];
                } else {
                    rankId2devType[myRankId] = uniDevType;
                }

                rankId2superdeviceId[myRankId] = rankTable.rankList[myRankId].superDeviceId;
                myRankId++;
            }
        }
    }
}

} // namespace hccl
