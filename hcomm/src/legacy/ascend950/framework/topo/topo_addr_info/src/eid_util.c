/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "eid_util.h"
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "securec.h"

#define UE_ID_POS (14)
#define HEX_BASE (16)
#define PORT_POS (30)
#define DIE_POS (39)
#define DIE_OFFSET (3)

static int Char2Int(char c, unsigned char* i)
{
    if (!isxdigit(c)) {
        return -1;
    }
    if (!isdigit(c)) {
        return -1;
    }
    *i = toupper(c) - '0';
    return 0;
}

int EidGetFeId(const char* eidhexstr)
{
    char fe = eidhexstr[UE_ID_POS];
    int fe_id = (int)strtol(&fe, NULL, HEX_BASE) >> 1;
    return fe_id;
}

int EidGetPortId(const char* eidhexstr, int* port_id)
{
    unsigned char end = (unsigned char)strtoul(&eidhexstr[PORT_POS], NULL, HEX_BASE);
    unsigned int flag = 0x80;
    unsigned int tmp = (~flag & end);
    unsigned int port = tmp >> 3;
    *port_id = (int)port;
    return 0;
}

int EidGetDieId(const char* eidhexstr, int* die_id)
{
    unsigned char d = 0;
    Char2Int(eidhexstr[DIE_POS], &d);
    *die_id = (int)((8 & d) >> DIE_OFFSET);
    return 0;
}

// 标卡
int UrmaEidGetDieIdForCard(dcmi_urma_eid_t* eid)
{
    unsigned char last = eid->raw[DCMI_URMA_EID_SIZE - 1];
    int die_id = (4 & last) == 0 ? 0 : 1;
    return die_id;
}

int UrmaEidGetPortIdForCard(dcmi_urma_eid_t* eid)
{
#define EID_PORT_LEFT_OFFSET (1)
#define EID_PORT_RIGHT_OFFSET (4)
    unsigned char last = eid->raw[DCMI_URMA_EID_SIZE - 1];
    last = last << EID_PORT_LEFT_OFFSET;
    last = last >> EID_PORT_RIGHT_OFFSET;
    return (int)last;
}

// 服务器和PoD
int UrmaEidGetFeId(const dcmi_urma_eid_t* eid)
{
#define FE_POS (6)
    unsigned char fe = eid->raw[FE_POS];
    const unsigned char mask = 0x7f; // fe使用7bit
    fe = fe & mask;
    return (int)fe;
}

#define RAW_PORT_AND_DIE_POS (5)
#define RAW_PORT_BIT_LEN (6)
/**
 * @brief 获取URMA eid中的portID
 * port_id和die id 在同一个unsigned char中
 * |0|1|2|3|4|5|6|7|
 * |die| port_id   |
 * @param eid URMA eid结构体指针
 * @return int portID
 */
int UrmaEidGetPortId(const dcmi_urma_eid_t* eid)
{
    unsigned char dieAndPort = eid->raw[RAW_PORT_AND_DIE_POS];
    const unsigned char mask = 0x3F;
    unsigned char port = dieAndPort & mask;
    return (int)port;
}

int UrmaEidGetDieId(const dcmi_urma_eid_t* eid)
{
    unsigned char dieAndPort = eid->raw[RAW_PORT_AND_DIE_POS];
    unsigned char dieId = dieAndPort >> RAW_PORT_BIT_LEN;
    return (int)dieId;
}

int GetMaxFeId(const dcmi_urma_eid_info_t* eidList, size_t eidCnt)
{
    int maxFeId = -1;
    for (size_t i = 0; i < eidCnt; ++i) {
        int feId = UrmaEidGetFeId(&eidList[i].eid);
        if (feId > maxFeId) {
            maxFeId = feId;
        }
    }
    return maxFeId;
}

/**
 * @brief 判断是否为portgroup
 * 判断依据:portID为0x3F时，为portgroup
 * @param eid URMA eid结构体指针
 * @return int 1为portgroup，0为物理ID
 */
int UrmaEidIsPortGroup(const dcmi_urma_eid_t* eid)
{
    const int pgPortId = 0x3F;
    int port = UrmaEidGetPortId(eid);
    if (port == pgPortId) {
        return 1;
    }
    return 0;
}

/**
 * 判断是否是UBOE
 * 在第7 byte位置上
 * 11000000 为UBOE, 使用0xc0判断
 * 10000000 为UBOE, 使用0x80判断
 * @param eid URMA eid结构体指针
 * @return int 1为UBOE，0为不是UBOE
 */
int UrmaEidIsUBOE(const dcmi_urma_eid_t* eid)
{
    const int uboeMask = 0xc0;
    const int flagPos = 7; // UBOE和UB_RTP标志位的位置
    unsigned char flag = eid->raw[flagPos];
    if ((uboeMask & flag) == uboeMask) {
        return 1;
    }
    return 0;
}

/**
 * 判断是否是UB_RTP,  判断原理见UrmaEidIsUBOE
 * @param eid URMA eid结构体指针
 * @return int 1为UB_RTP，0为不是UB_RTP
 */
int UrmaEidIsUbRtp(const dcmi_urma_eid_t* eid)
{
    const unsigned char bitMask = 0xc0;
    const unsigned char ubRtpFlag = 0x80; // UB_RTP的flag
    const int flagPos = 7;                // UBOE和UB_RTP标志位的位置
    unsigned char flag = eid->raw[flagPos];
    int bit = (flag & bitMask); //
    if (bit == ubRtpFlag) {
        return 1;
    }
    return 0;
}

int UrmaEid2CNA(const dcmi_urma_eid_t* eid, char* cna, size_t cnaSize)
{
    if (cna == NULL || cnaSize == 0) {
        return -1;
    }
    const int ipPos1 = 6; // ip 第一个字节位置
    const int ipPos2 = 12;
    const int ipPos3 = 13;
    const int ipPos4 = 14;
    unsigned char ip1 = eid->raw[ipPos1];
    unsigned char ip2 = eid->raw[ipPos2];
    unsigned char ip3 = eid->raw[ipPos3];
    unsigned char ip4 = eid->raw[ipPos4];
    return sprintf_s(cna, cnaSize, "%u.%u.%u.%u", ip1, ip2, ip3, ip4);
}

int UBEntityGetId(const UBEntity* ue)
{
    if (ue->eidNum == 0) {
        return -1;
    }
    return UrmaEidGetFeId(&ue->eidList[0].eid);
}

int UBEntityGetDieId(const UBEntity* ue)
{
    if (ue->eidNum == 0) {
        return -1;
    }
    return UrmaEidGetDieId(&ue->eidList[0].eid);
}

int UBEntityGetPortGroupIdx(const UBEntity* ue)
{
    for (int i = 0; i < (int)ue->eidNum; ++i) {
        if (UrmaEidIsPortGroup(&ue->eidList[i].eid)) {
            return i;
        }
    }
    // 没有找到
    return -1;
}

int UBGetMaxEntityId(const UEList* ueList, int dieId)
{
    int maxId = -1;
    for (unsigned int i = 0; i < ueList->ueNum; ++i) {
        if (ueList->ueList[i].eidNum == 0 || UBEntityGetDieId(&ueList->ueList[i]) != dieId) {
            continue;
        }
        // 一个UBEntity下的所有eid的Entity ID相同
        int id = UrmaEidGetFeId(&ueList->ueList[i].eidList[0].eid);
        if (id > maxId) {
            maxId = id;
        }
    }
    return maxId;
}
