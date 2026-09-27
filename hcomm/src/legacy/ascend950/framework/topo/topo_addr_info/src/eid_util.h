/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __EID_UTIL_H__
#define __EID_UTIL_H__

#include "hal.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 用于获取EID中的信息
 * EID是一个128bit的应用层地址, EID的编址规则中包含了物理端口ID、iodie ID、UBEntity ID等信息
 * 本文件中的相关函数为解析EID使用
 */

/**
 * 从EID的16进制字符串中解析出UBEntityID
 */
int EidGetFeId(const char* eid_str);

/**
 * 获取低3-6bit表示的物理端口号,  012bit表示NPU号
 */
int EidGetPortId(const char* eid_str, int* port_id);
int EidGetDieId(const char* eid_str, int* die_id);

// 标卡
int UrmaEidGetDieIdForCard(dcmi_urma_eid_t* eid);
int UrmaEidGetPortIdForCard(dcmi_urma_eid_t* eid);

// 服务器和PoD
int UrmaEidGetFeId(const dcmi_urma_eid_t* eid);
int UrmaEidGetPortId(const dcmi_urma_eid_t* eid);
int UrmaEidGetDieId(const dcmi_urma_eid_t* eid);
/**
 * 判断是否为portgroup
 * 判断依据:portID为0x3F时，为portgroup
 * @param eid URMA eid结构体指针
 * @return int 1为portgroup，0 不是portgroup
 */
int UrmaEidIsPortGroup(const dcmi_urma_eid_t* eid);

int UrmaEidIsUBOE(const dcmi_urma_eid_t* eid);

int UrmaEidIsUbRtp(const dcmi_urma_eid_t* eid);
/**
 * 从EID中解析出CNA地址,
 * @param eid URMA eid结构体指针
 * @param ip CNA以IP地址的方式表达
 * @param ip_len CNA地址字符串长度
 * @return int 0成功，-1失败
 */
int UrmaEid2CNA(const dcmi_urma_eid_t* eid, char* cna, size_t cnaSize);

/**
 * 获取FE ID, FE是UB中的功能实体, 在EID编址规则中，讲FE ID编在了EID中
 */
int EidGetFeId(const char* eidhexstr);

/**
 * 根据EID编址规范，FE最大的是mesh链接使用
 */
int GetMaxFeId(const dcmi_urma_eid_info_t* eidList, size_t eid_cnt);

/**
 * 获取UBEntity ID
 */
int UBEntityGetId(const UBEntity* ue);

/**
 * 获取UBEntity的die ID, 一个UBEntity只属于一个iodie
 */
int UBEntityGetDieId(const UBEntity* ue);

int UBEntityGetPortGroupIdx(const UBEntity* ue);

int UBGetMaxEntityId(const UEList* ueList, int dieId);

#ifdef __cplusplus
}
#endif

#endif // __EID_UTIL_H__
