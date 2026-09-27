/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "securec.h"
#include "ra_rs_err.h"
#include "rs_inner.h"
#include "dl_hal_function.h"

int RsSensorNodeRegister(unsigned int phyId, struct rs_cb *rsCb)
{
    struct halSensorNodeCfg cfg = {0};
    int ret;

    if (rsCb->sensorNode.sensorHandle != 0) {
        return 0;
    }

    // some non-hdc scenarios don't have corresponding API, skip to register sensor node
    if (rsCb->hccpMode != NETWORK_OFFLINE) {
        return 0;
    }

    ret = rsGetLocalDevIDByHostDevID(phyId, &rsCb->sensorNode.logicDevid);
    if (ret) {
        hccp_err("[init][rs_rdev]rsGetLocalDevIDByHostDevID failed, phyId(%u), ret(%d)", phyId, ret);
        return ret;
    }

    ret = sprintf_s(cfg.name, sizeof(cfg.name), "roce_rs_%d", getpid());
    if (ret <= 0) {
        hccp_err("[init][rs_rdev]sprintf_s name err, ret:%d, phyId:%u", ret, phyId);
        return -ESAFEFUNC;
    }

    cfg.NodeType = HAL_DMS_DEV_TYPE_HCCP;
    cfg.SensorType = RDMA_CQE_ERR_SENSOR_TYPE;
    cfg.AssertEventMask = RDMA_CQE_ERR_RETRY_TIMEOUT_EVENT_MASK;
    cfg.DeassertEventMask = RDMA_CQE_ERR_RETRY_TIMEOUT_EVENT_TYPE_MASK;
    ret = DlHalSensorNodeRegister(rsCb->sensorNode.logicDevid, &cfg, &rsCb->sensorNode.sensorHandle);
    if (ret != 0) {
        hccp_err("[init][rs_rdev]dl_hal_sensor_node_register failed, phyId(%u), logicDevid(%u), ret(%d)", phyId,
            rsCb->sensorNode.logicDevid, ret);
        return ret;
    }

    return 0;
}

void RsSensorNodeUnregister(struct rs_cb *rsCb)
{
    // no need to unregister sensor node
    if (rsCb->sensorNode.sensorHandle == 0) {
        return;
    }

    RS_PTHREAD_MUTEX_LOCK(&rsCb->mutex);
    if (RsListEmpty(&rsCb->rdevList)) {
        (void)DlHalSensorNodeUnregister(rsCb->sensorNode.logicDevid, rsCb->sensorNode.sensorHandle);
        rsCb->sensorNode.sensorUpdateCnt = 0;
        rsCb->sensorNode.sensorHandle = 0;
    }
    RS_PTHREAD_MUTEX_ULOCK(&rsCb->mutex);
}
