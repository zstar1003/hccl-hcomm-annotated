/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __HAL_H__
#define __HAL_H__

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DCMI_URMA_EID_SIZE (16)
#define MAX_EID_NUM (32)
#define MAX_NPU_COUNT (64)

typedef union dcmi_urma_eid {
    unsigned char raw[DCMI_URMA_EID_SIZE];
    struct {
        unsigned long subnet_prefix;
        unsigned long interface_id;
    } in6;
} dcmi_urma_eid_t;

typedef struct dcmi_urma_eid_info {
    dcmi_urma_eid_t eid;
    unsigned int eid_index;
} dcmi_urma_eid_info_t;

#define MAX_EID_PER_UE (32)
typedef struct {
    dcmi_urma_eid_info_t eidList[MAX_EID_PER_UE];
    unsigned int eidNum;
} UBEntity;

#define MAX_UE_PER_NPU (8)
typedef struct {
    UBEntity ueList[MAX_UE_PER_NPU];
    unsigned int ueNum;
} UEList;

int HalGetUBEntityList(int phyId, UEList* ueList);

struct dcmi_pcie_info_all {
    unsigned int venderid;     // 厂商ID
    unsigned int subvenderid;  // 厂商子ID
    unsigned int deviceid;     // 设备ID
    unsigned int subdeviceid;  // 设备子ID
    int domain;                // pcie domain
    unsigned int bdf_busid;    // BDF（Bus，Device，Function）中的总线ID
    unsigned int bdf_deviceid; // BDF（Bus，Device，Function）中的设备ID
    unsigned int bdf_funcid;   // BDF（Bus，Device，Function）中的功能ID
    unsigned char reserve[32];
};

struct dcmi_spod_info {
    unsigned int sdid;
    unsigned int super_pod_size;
    unsigned int super_pod_id;
    unsigned int server_index;
    unsigned int chassis_id;
    unsigned int super_pod_type;
    unsigned int reserve[6];
};

#define MAIN_BOARD_INVALID (0)
#define MAIN_BOARD_ID_CARD_NOMESH (0x68)
#define MAIN_BOARD_ID_CARD_2PMESH (0x6a)
#define MAIN_BOARD_ID_CARD_4PMESH (0x6c)
#define MAIN_BOARD_ID_SERVER_TYPE1 (0x23)
#define MAIN_BOARD_ID_SERVER_8PMESH (0x25)
#define MAIN_BOARD_ID_SERVER_8PMESH_UBOE (0x27)
#define MAIN_BOARD_ID_SERVER_8PMESH_NOSP (0x29)
#define MAIN_BOARD_ID_SERVER_8PMESH_NOSP_UBOE (0x2B)
#define MAIN_BOARD_ID_SERVER_350L (0x44)
#define MAIN_BOARD_ID_SERVER_550EL_100 (0x46)
#define MAIN_BOARD_ID_SERVER_550EL_200 (0x48)
#define MAIN_BOARD_ID_POD (0x07)
#define MAIN_BOARD_ID_POD_2D (0x03)
#define MAIN_BOARD_ID_POD_FLEX (0x2D)
#define MAIN_BOARD_ID_POD_FLEX_RTP (0x2F)

/* 定义拓扑类型 */
#define TOPO_TYPE_IGNORE (99)     // 忽略该拓扑类型
#define TOPO_TYPE_SERVER_8P (0)   // 普通服务
#define TOPO_TYPE_SERVER_16FM (3) // 两个服务器组16p fullmesh

int hal_get_eid_list_by_phy_id(int phyId, dcmi_urma_eid_info_t* eidList, size_t* eidCnt);

int hal_get_mainboard_id(int phyId, unsigned int* mainboardId);

int hal_get_device_pcie_info(int phyId, struct dcmi_pcie_info_all* pcieInfo);

int hal_get_spod_info(int phyId, struct dcmi_spod_info* spodInfo);

int hal_get_npu_count();

int hal_get_logicid_from_phyid(unsigned int phyId, unsigned int* logicId);

int hal_get_userdevid_by_phyid(int phyId, int* userDevId);

int get_server_id(char* server_id, size_t buf_size);

int hal_get_driver_install_path(char* value_buf, size_t buf_size);

/**
 * 添加一层适配，便于测试
 */
void* hal_dlopen(const char* filename, int flag);
void* hal_dlsym(void* handle, const char* symbol);

#ifdef __cplusplus
}
#endif

#endif // __HAL_H__
