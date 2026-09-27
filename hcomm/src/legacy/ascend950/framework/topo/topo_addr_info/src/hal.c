/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hal.h"
#include "topo_addr_info_log.h"
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <net/if_arp.h>
#include <errno.h>
#include <pthread.h>
#include <syslog.h>
#include "securec.h"

#define MAX_LINE_LENGTH 256                    // 每行最大长度
#define TARGET_KEY "Driver_Install_Path_Param" // 要查找的key

#define DRIVER_DRFAULT_INSTALL_PATH "/usr/local/Ascend"
#define DRIVER_TOPO_FILE_DIR_PATH "driver/topo/950"
#define MAX_TOPO_FILENAME_LEN (64)

#define HAL_TRUE (1)
#define HAL_FALSE (0)

#define MODULE_TYPE_SYSTEM (0)
#define INFO_TYPE_SERVER_ID (27)
#define INFO_TYPE_SPOD_ID (29)
#define INFO_TYPE_MAINBOARD_ID (39)
#define INFO_TYPE_CHASSI_ID (48)
#define INFO_TYPE_SPOD_TYPE (49)

enum dcmi_main_cmd {
    DCMI_MAIN_CMD_DVPP = 0,
    DCMI_MAIN_CMD_ISP,
    DCMI_MAIN_CMD_TS_GROUP_NUM,
    DCMI_MAIN_CMD_CAN,
    DCMI_MAIN_CMD_UART,
    DCMI_MAIN_CMD_UPGRADE = 5,
    DCMI_MAIN_CMD_UFS,
    DCMI_MAIN_CMD_OS_POWER,
    DCMI_MAIN_CMD_LP,
    DCMI_MAIN_CMD_MEMORY,
    DCMI_MAIN_CMD_RECOVERY,
    DCMI_MAIN_CMD_TS,
    DCMI_MAIN_CMD_CHIP_INF,
    DCMI_MAIN_CMD_QOS,
    DCMI_MAIN_CMD_SOC_INFO,
    DCMI_MAIN_CMD_HCCS = 16,
    DCMI_MAIN_CMD_TEMP = 50,
    DCMI_MAIN_CMD_SVM = 51,
    DCMI_MAIN_CMD_VDEV_MNG,
    DCMI_MAIN_CMD_SIO = 56,
    DCMI_MAIN_CMD_DEVICE_SHARE = 0x8001,
    DCMI_MAIN_CMD_MAX
};

typedef enum {
    DCMI_CHIP_INFO_SUB_CMD_CHIP_ID,
    DCMI_CHIP_INFO_SUB_CMD_SPOD_INFO,
    DCMI_CHIP_INFO_SUB_CMD_MAX = 0xFF,
} DCMI_CHIP_INFO_SUB_CMD;

static int (*dcmi_init)(void);

static int (*dcmiv2_get_urma_device_cnt)(int npu_id, unsigned int* dev_cnt);

static int (*dcmiv2_get_eid_list_by_urma_dev_index)(
    int npu_id, int urma_dev_index, dcmi_urma_eid_info_t* eid_list, int* eid_cnt);

static int (*dcmiv2_get_device_pcie_info)(int npu_id, struct dcmi_pcie_info_all* pcie_info);

static int (*aclrtGetUserDevIdByPhyDevId)(const int32_t phyId, int32_t* const userDevId);

static int (*aclrtGetLogicDevIdByUserDevId)(const int32_t userDevId, int32_t* const logicId);

static int (*halGetDeviceInfo)(unsigned int devId, uint32_t moduleType, int32_t infoType, int64_t* value);

void* hal_dlopen(const char* filename, int flag) { return dlopen(filename, flag); }

void* hal_dlsym(void* handle, const char* symbol) { return dlsym(handle, symbol); }

int load_dcmi()
{
    static pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
    static void* dcmi = NULL;
    static void* acl = NULL;
    static int isInit = HAL_FALSE;
    pthread_mutex_lock(&mutex);
    if (dcmi != NULL && acl != NULL && isInit == HAL_TRUE) {
        pthread_mutex_unlock(&mutex);
        return 0;
    }
    if (dcmi == NULL || acl == NULL) {
        dcmi = hal_dlopen("libdcmi.so", RTLD_LAZY);
        acl = hal_dlopen("libacl_rt.so", RTLD_LAZY);
    }

    if (dcmi == NULL || acl == NULL) {
        pthread_mutex_unlock(&mutex);
        return -1;
    }
    dcmi_init = hal_dlsym(dcmi, "dcmiv2_init");
    dcmiv2_get_urma_device_cnt = hal_dlsym(dcmi, "dcmiv2_get_urma_device_cnt");
    dcmiv2_get_eid_list_by_urma_dev_index = hal_dlsym(dcmi, "dcmiv2_get_eid_list_by_urma_dev_index");
    dcmiv2_get_device_pcie_info = hal_dlsym(dcmi, "dcmiv2_get_device_pcie_info");

    // 兼容性处理 aclrtGetLogicDevIdByPhyDevId接口语义错误,实际返回的是UserDevId，优先使用新接口
    aclrtGetUserDevIdByPhyDevId = hal_dlsym(acl, "aclrtGetUserDevIdByPhyDevId");
    if (aclrtGetUserDevIdByPhyDevId == NULL) {
        aclrtGetUserDevIdByPhyDevId = hal_dlsym(acl, "aclrtGetLogicDevIdByPhyDevId");
    }

    halGetDeviceInfo = hal_dlsym(acl, "halGetDeviceInfo");
    aclrtGetLogicDevIdByUserDevId = hal_dlsym(acl, "aclrtGetLogicDevIdByUserDevId");

    if ((dcmi_init == NULL) || (dcmiv2_get_urma_device_cnt == NULL) || (dcmiv2_get_eid_list_by_urma_dev_index == NULL)
        || (halGetDeviceInfo == NULL) || (dcmiv2_get_device_pcie_info == NULL) || (aclrtGetUserDevIdByPhyDevId == NULL)
        || (aclrtGetLogicDevIdByUserDevId == NULL)) {
        pthread_mutex_unlock(&mutex);
        return -1;
    }
    (void)dcmi_init(); //  dcmi_init可能已经调用过了
    isInit = HAL_TRUE;
    pthread_mutex_unlock(&mutex);
    return 0;
}

int hal_get_mainboard_id(int phyId, unsigned int* mainboardId)
{
    unsigned int logicId = 0;
    if (hal_get_logicid_from_phyid((unsigned int)phyId, &logicId) != 0) {
        return -1;
    }
    int64_t value = 0;
    int ret = halGetDeviceInfo(logicId, MODULE_TYPE_SYSTEM, INFO_TYPE_MAINBOARD_ID, &value);
    if (ret == 0) {
        *mainboardId = (unsigned int)value;
    }
    return ret;
}

#define MAX_IFREQ_NUM (16)
int get_server_id(char* server_id, size_t len)
{
    int sock_fd;
    struct ifconf ifc;
    struct ifreq ifr[MAX_IFREQ_NUM];
    int if_count, i;

    if ((sock_fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
        return -1;
    }

    ifc.ifc_len = sizeof(ifr);
    ifc.ifc_buf = (char*)ifr;
    if (ioctl(sock_fd, SIOCGIFCONF, &ifc) < 0) {
        close(sock_fd);
        return -1;
    }

    if_count = ifc.ifc_len / sizeof(struct ifreq);
    for (i = 0; i < if_count; ++i) {
        if (strcmp(ifr[i].ifr_name, "lo") == 0) {
            continue;
        }
        if (ioctl(sock_fd, SIOCGIFHWADDR, &ifr[i]) < 0) {
            continue;
        }
        if (ifr[i].ifr_hwaddr.sa_family != ARPHRD_ETHER) {
            continue;
        }
        unsigned char* mac = (unsigned char*)ifr[i].ifr_hwaddr.sa_data;
        (void)sprintf_s(server_id, len, "%02X%02X%02X%02X%02X%02X", mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
        close(sock_fd);
        return 0;
    }
    close(sock_fd);
    return -1;
}

int hal_get_eid_list_by_phy_id(int phyId, dcmi_urma_eid_info_t* eidList, size_t* eidCnt)
{
    unsigned int logicId = 0;
    if (hal_get_logicid_from_phyid((unsigned int)phyId, &logicId) != 0) {
        return -1;
    }
    unsigned int dev_cnt = 0;
    int ret = dcmiv2_get_urma_device_cnt((int)logicId, &dev_cnt);
    if (ret != 0) {
        return ret;
    }
    size_t eid_current_cnt = 0;
    size_t eid_space_left = (*eidCnt);
    for (size_t i = 0; i < dev_cnt; ++i) {
        int left = (int)eid_space_left;
        ret = dcmiv2_get_eid_list_by_urma_dev_index((int)logicId, i, &eidList[eid_current_cnt], &left);
        if (ret != 0) {
            continue;
        }
        eid_space_left -= left;
        eid_current_cnt += left;
    }
    (*eidCnt) = eid_current_cnt;
    return 0;
}

int HalGetUBEntityList(int phyId, UEList* ueList)
{
    if (ueList == NULL) {
        return 0;
    }
    (void)memset_s(ueList, sizeof(UEList), 0x00, sizeof(UEList));
    unsigned int logicId = 0;
    if (hal_get_logicid_from_phyid((unsigned int)phyId, &logicId) != 0) {
        return -1;
    }
    int ret = dcmiv2_get_urma_device_cnt((int)logicId, &ueList->ueNum);
    if (ret != 0) {
        return ret;
    }
    for (size_t i = 0; i < ueList->ueNum; ++i) {
        int num = MAX_EID_PER_UE;
        ret = dcmiv2_get_eid_list_by_urma_dev_index((int)logicId, i, ueList->ueList[i].eidList, &num);
        if (ret != 0) {
            continue;
        }
        ueList->ueList[i].eidNum = (unsigned int)num;
    }
    return 0;
}

int hal_get_device_pcie_info(int phyId, struct dcmi_pcie_info_all* pcieInfo)
{
    unsigned int logicId = 0;
    if (hal_get_logicid_from_phyid((unsigned int)phyId, &logicId) != 0) {
        return -1;
    }
    return dcmiv2_get_device_pcie_info(logicId, pcieInfo);
}

int hal_get_spod_info(int phyId, struct dcmi_spod_info* spodInfo)
{
    unsigned int logicId = 0;
    if (hal_get_logicid_from_phyid((unsigned int)phyId, &logicId) != 0) {
        return -1;
    }
    int64_t spodId = 0;
    int64_t serverId = 0;
    int64_t chassisId = 0;
    int64_t spodType = 0;
    int ret1 = halGetDeviceInfo(logicId, MODULE_TYPE_SYSTEM, INFO_TYPE_SPOD_ID, &spodId);
    int ret2 = halGetDeviceInfo(logicId, MODULE_TYPE_SYSTEM, INFO_TYPE_SERVER_ID, &serverId);
    int ret3 = halGetDeviceInfo(logicId, MODULE_TYPE_SYSTEM, INFO_TYPE_CHASSI_ID, &chassisId);
    int ret4 = halGetDeviceInfo(logicId, MODULE_TYPE_SYSTEM, INFO_TYPE_SPOD_TYPE, &spodType);
    if (ret1 != 0 || ret2 != 0 || ret3 != 0 || ret4 != 0) {
        return -1;
    }
    spodInfo->super_pod_id = (unsigned int)spodId;
    spodInfo->server_index = (unsigned int)serverId;
    spodInfo->chassis_id = (unsigned int)chassisId;
    spodInfo->super_pod_type = (unsigned int)spodType;
    return 0;
}

int hal_get_npu_count()
{
#define MAX_NPU_COUNT (64)
#define MAX_DAVINCI_DEV_LEN (64)
    int count = 0;
    for (int i = 0; i < MAX_NPU_COUNT; ++i) {
        char davinci_dev[MAX_DAVINCI_DEV_LEN] = {0};
        (void)sprintf_s(davinci_dev, sizeof(davinci_dev), "/dev/davinci%d", i);
        if (access(davinci_dev, F_OK) == 0) {
            count++;
        }
    }
    return count;
}

int hal_get_logicid_from_phyid(unsigned int phyId, unsigned int* logicId)
{
    if (load_dcmi() != 0) {
        return -1;
    }
    int userDevId = -1;
    int ret = aclrtGetUserDevIdByPhyDevId((int)phyId, &userDevId);
    if (ret != 0) {
        return -1;
    }
    int value = -1;
    ret = aclrtGetLogicDevIdByUserDevId(userDevId, &value);
    if (ret != 0) {
        return -1;
    }
    *logicId = (unsigned int)value;
    return 0;
}

int hal_get_userdevid_by_phyid(int phyId, int* userDevId)
{
    if (load_dcmi() != 0) {
        return -1;
    }
    return aclrtGetUserDevIdByPhyDevId(phyId, userDevId);
}

// 去除字符串首尾的空白字符
static char* trim_whitespace(char* str)
{
    char* end;

    // 去除开头空格
    while (isspace((unsigned char)*str)) {
        str++;
    }
    // 如果字符串全是空格，返回空字符串
    if (*str == 0) {
        return str;
    }

    // 去除结尾空格
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) {
        end--;
    }
    // 添加字符串结束符
    end[1] = '\0';
    return str;
}

/**
 * @brief 解析ascend_install.info文件，获取指定key的value
 * @param value_buf 存储结果的缓冲区
 * @param buf_size 缓冲区大小
 * @return 成功返回0，失败返回-1值
 */
int hal_get_driver_install_path(char* value_buf, size_t buf_size)
{
    FILE* fp = NULL;
    char* line = NULL;
    size_t len = 0;
    ssize_t read;
    // 参数合法性检查
    if (value_buf == NULL || buf_size == 0) {
        return -1;
    }
    // 初始化缓冲区
    value_buf[0] = '\0';

    // 打开文件
    fp = fopen("/etc/ascend_install.info", "r");
    if (fp == NULL) {
        if (strcpy_s(value_buf, buf_size, DRIVER_DRFAULT_INSTALL_PATH) != 0) {
            return -1;
        }
        return 0;
    }

    // 逐行读取文件
    while ((read = getline(&line, &len, fp)) != -1) {
        // 去除换行符
        char* newline = strchr(line, '\n');
        if (newline)
            *newline = '\0';

        // 查找等号位置
        char* equal_sign = strchr(line, '=');
        if (equal_sign == NULL) {
            continue; // 跳过没有等号的行
        }

        // 分割key和value
        *equal_sign = '\0';
        char* key = trim_whitespace(line);
        char* value = trim_whitespace(equal_sign + 1);

        // 匹配目标key
        if (strcmp(key, TARGET_KEY) == 0) {
            // 检查缓冲区大小
            if (strlen(value) + 1 > buf_size) {
                break;
            }
            // 复制value到缓冲区
            errno_t ret = strcpy_s(value_buf, buf_size, value);
            if (ret != 0) {
                break;
            }
            value_buf[buf_size - 1] = '\0';
            break;
        }
    }

    // 释放资源
    if (line) {
        free(line);
    }
    if (fp) {
        fclose(fp);
    }
    if (strlen(value_buf) == 0) {
        // 默认值兜底
        if (strcpy_s(value_buf, buf_size, DRIVER_DRFAULT_INSTALL_PATH) != 0) {
            return -1;
        }
    }
    return 0;
}
