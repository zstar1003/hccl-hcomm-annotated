/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RA_RS_CTX_H
#define RA_RS_CTX_H

#include "hccp_ctx.h"
#include "ra_rs_comm.h"

#define MAX_RSGE_NUM 2
#define MAX_CTX_WR_NUM 4
#define MAX_INLINE_SIZE 64
#define REM_JETTY_ID_DEPRECATED_SIZE (sizeof(unsigned int))
#define REM_JETTY_ID_SIZE 24 // refer to urma_jetty_id_t in struct QpKey

struct MemRegAttrT {
    struct HccpMemInfo mem;
    union {
        struct {
            int access; /**< refer to enum mem_mr_access_flags */
        } rdma;

        struct {
            union RegSegFlag flags;
            uint32_t tokenValue;  /**< refer to urma_token_t */
            uint64_t tokenIdAddr; /**< NULL means unspecified, valid if flags.token_id_valid been set */
        } ub;
    };
    uint32_t resv[8U];
};

struct MemRegInfoT {
    struct MemKey key;
    union {
        struct {
            uint32_t lkey;
        } rdma;

        struct {
            uint32_t tokenId;
            uint64_t targetSegHandle; /**< refer to urma_target_seg_t */
        } ub;
    };
    uint32_t resv[8U];
};

struct MemImportAttrT {
    struct MemKey key;

    union {
        struct {
            union ImportSegFlag flags; /**< refer to urma_import_seg_flag_t */
            uint64_t mappingAddr;      /**< addr is needed if flag mapping set value */
            uint32_t tokenValue;       /**< refer to urma_token_t */
        } ub;
    };
    uint32_t resv[4U];
};

struct MemImportInfoT {
    union {
        struct {
            uint32_t rkey;
        } rdma;

        struct {
            uint64_t targetSegHandle; /**< refer to urma_target_seg_t */
        } ub;
    };
    uint32_t resv[4U];
};

struct CtxCqAttr {
    uint64_t chanAddr; /**< resv for chan_cb index */
    uint32_t depth;
    union {
        struct {
            uint64_t cqContext;
            uint32_t mode; /**< refer to enum RA_RDMA_NOR_MODE etc. */
            uint32_t compVector;
        } rdma;

        struct {
            uint64_t userCtx;
            enum JfcMode mode;
            uint32_t ceqn;
            union JfcFlag flag; /**< refer to urma_jfc_flag_t */
            struct {
                bool valid;
                uint32_t cqeFlag;
            } ccuExCfg;
        } ub;
    };
    uint32_t resv[4U];
};

struct CtxCqInfo {
    uint64_t addr; /**< refer to ibv_cq*, urma_jfc_t* for cq_cb index */
    struct {
        uint32_t id; /**< jfc id */
        uint32_t cqeSize;
        uint64_t bufAddr;
        uint64_t swdbAddr;
        uint32_t resv[2U]; /**< resv for stars poll cq */
    } ub;
    uint32_t resv[4U];
};

struct CtxQpAttr {
    uint64_t scqIndex;
    uint64_t rcqIndex;
    uint64_t srqIndex;

    uint32_t sqDepth;
    uint32_t rqDepth;

    enum TransportModeT transportMode;

    union {
        struct {
            uint32_t mode;        /**< refer to enum RA_RDMA_NOR_MODE etc. */
            uint32_t udpSport;    /**< UDP source port */
            uint8_t trafficClass; /**< traffic class */
            uint8_t sl;           /**< service level */
            uint8_t timeout;      /**< local ack timeout */
            uint8_t rnrRetry;     /**< RNR retry count */
            uint8_t retryCnt;     /**< retry count */
        } rdma;

        struct {
            enum JettyMode mode;
            uint32_t jettyId;      /**< [optional] user specified jetty id, 0 means not specified */
            union JettyFlag flag;  /**< refer to union urma_jetty_flag */
            union JfsFlag jfsFlag; /**< refer to union urma_jfs_flag; jfs_cfg->flag */
            uint32_t tokenValue;   /**< refer to urma_token_t; jfr_cfg->token_value */
            uint64_t tokenIdAddr;  /**< NULL means unspecified */
            uint8_t priority; /**< the priority of JFS. services with low delay need set high priority. Range:[0-0xf] */
            uint8_t rnrRetry; /**< the RNR retry count when receive RNR response; Range:[0-7] */
            uint8_t errTimeout; /**< the timeout to report error. Range: [0-31] */
            union {
                struct {
                    struct JettyQueCfgEx
                        sq;      /**< specify sq buffer config, required when cstm_flag.bs.sq_cstm specified */
                    bool piType; /**< false: op mode, true: async mode */
                    union CstmJfsFlag cstmFlag; /**< refer to union udma_jfs_flag */
                    uint32_t sqebbNum;          /**< required when cstm_flag.bs.sq_cstm specified */
                } extMode;
                struct {
                    bool lockFlag;
                    uint32_t sqeBufIdx;
                } taCacheMode;
            };
        } ub;
    };

    uint32_t resv[16U];
};

struct RaRsJettyImportAttr {
    enum JettyImportMode mode;
    uint32_t tokenValue;
    enum JettyGrpPolicy policy;
    enum TargetType type;
    union ImportJettyFlag flag;
    struct JettyImportExpCfg expImportCfg; /**< only valid on mode JETTY_IMPORT_MODE_EXP */
    uint32_t tpType;
    uint32_t resv[15U];
};

struct RaRsJettyImportInfo {
    uint64_t tjettyHandle;
    uint32_t tpn;
    uint32_t resv[16U];
};

struct WrlistBaseInfo {
    unsigned int phyId;
    unsigned int devIndex;
    unsigned int qpn;
};

struct RaWrSge {
    uint64_t addr;
    uint32_t len;
    uint64_t devLmemHandle;
};

struct HdcWrNotifyInfo {
    uint64_t notifyData;
    uint64_t notifyAddr;
    uint64_t notifyHandle;
};

struct BatchSendWrData {
    struct RaWrSge sges[MAX_SGE_NUM];
    uint32_t numSge;

    char inlineData[MAX_INLINE_SIZE];
    uint32_t inlineSize;

    uint64_t remoteAddr;
    uint64_t devRmemHandle;

    union {
        struct {
            uint64_t wrId;
            enum RaWrOpcode opcode;
            unsigned int flags; /**< reference to ra_send_flags */
            struct WrAuxInfo aux;
        } rdma;

        struct {
            uint64_t userCtx;
            enum RaUbOpcode opcode;
            union JfsWrFlag flags;
            uint64_t remJetty;
            struct HdcWrNotifyInfo notifyInfo;
            struct WrReduceInfo reduceInfo;
        } ub;
    };

    uint32_t immData;

    uint32_t resv[16U];
};
#endif // RA_RS_CTX_H
