/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_CTX_H
#define HCCP_CTX_H

#include "hccp_common.h"
#include "hccp_ctx_data_plane.h"
#include "hccp_ctx_tp.h"
#include "hccp_ctx_dfx.h"

#ifdef __cplusplus
extern "C" {
#endif

#define HCCP_EID_IP_QUERY_MAX_NUM 32
#define DEV_EID_INFO_MAX_NAME 64

struct HccpDevEidInfo {
    char name[DEV_EID_INFO_MAX_NAME];
    uint32_t type;
    uint32_t eidIndex;
    union HccpEid eid;
    uint32_t dieId;
    uint32_t chipId;
    uint32_t funcId;
    uint32_t devFeature;
};

struct CtxInitCfg {
    int mode; /**< refer to enum NetworkMode */
};

struct CtxInitAttr {
    unsigned int phyId; /**< physical device id */
    union {
        struct {
            uint32_t eidIndex;
            uint32_t resv0;
            union HccpEid eid;
        } ub;
    };
    uint32_t resv[16U];
};

#define MEM_KEY_SIZE 128

struct MemKey {
    // RDMA: 4Bytes for uint32_t rkey
    // UB: 52Bytes for urma_seg_t seg
    uint8_t value[MEM_KEY_SIZE];
    uint8_t size;
};

union TpTypeCap {
    struct {
        uint32_t rtp : 1;
        uint32_t ctp : 1;
        uint32_t utp : 1;
        uint32_t reserved : 29;
    } bs;
    uint32_t value;
};

union TpFeature {
    struct {
        uint32_t rmMultiPath : 1;
        uint32_t rcMultiPath : 1;
        uint32_t reserved : 30;
    } bs;
    uint32_t value;
};

union TpTypeEn {
    struct {
        uint32_t rtp : 1;
        uint32_t ctp : 1;
        uint32_t utp : 1;
        uint32_t reserved : 29;
    } bs;
    uint32_t value;
};

struct CtxSlInfo {
    uint32_t SL;
    union TpTypeEn tpType;
};

#define MAX_PRIORITY_CNT 16

struct DevBaseAttr {
    uint32_t sqMaxDepth;
    uint32_t rqMaxDepth;
    uint32_t sqMaxSge;
    uint32_t rqMaxSge;
    union {
        struct {
            uint32_t maxJfsInlineLen;
            uint32_t maxJfsRsge;
            uint32_t dieId;
            uint32_t chipId;
            uint32_t funcId;
            union TpTypeCap rmTpCap;
            union TpTypeCap rcTpCap;
            union TpTypeCap umTpCap;
            union TpFeature tpFeat;
            struct CtxSlInfo priorityInfo[MAX_PRIORITY_CNT];
            uint32_t resv0;
        } ub;
    };

    uint32_t maxReadSize;
    uint32_t maxWriteSize;
    uint64_t maxMsgSize;
    uint32_t resv[12U];
};

struct HccpMemInfo {
    uint64_t addr;
    uint64_t size;
};

enum MemSegTokenPolicy {
    MEM_SEG_TOKEN_NONE = 0,
    MEM_SEG_TOKEN_PLAIN_TEXT = 1,
    MEM_SEG_TOKEN_SIGNED = 2,
    MEM_SEG_TOKEN_ALL_ENCRYPTED = 3,
    MEM_SEG_TOKEN_RESERVED = 4
};

enum MemSegAccessFlags {
    MEM_SEG_ACCESS_LOCAL_ONLY = 1,
    MEM_SEG_ACCESS_READ = (1 << 1U),
    MEM_SEG_ACCESS_WRITE = (1 << 2U),
    MEM_SEG_ACCESS_ATOMIC = (1 << 3U),
};

union RegSegFlag {
    struct {
        uint32_t tokenPolicy : 3; /**< refer to enum mem_seg_token_policy */
        uint32_t cacheable : 1;   /* 0: URMA_NON_CACHEABLE.
                                     1: URMA_CACHEABLE. */
        uint32_t dsva : 1;
        uint32_t access : 6;       /**< refer to enum mem_seg_access_flags */
        uint32_t nonPin : 1;       /* 0: segment pages pinned.
                                       1: segment pages non-pinned. */
        uint32_t userIova : 1;     /* 0: segment without user iova addr.
                                       1: segment with user iova addr. */
        uint32_t tokenIdValid : 1; /* 0: token id in cfg is invalid.
                                        1: token id in cfg is valid. */
        uint32_t reserved : 18;
    } bs;
    uint32_t value;
};

struct MemRegAttr {
    struct HccpMemInfo mem;
    union {
        struct {
            union RegSegFlag flags;
            uint32_t tokenValue; /**< refer to urma_token_t */
            void *tokenIdHandle; /**< NULL means unspecified, valid if flags.token_id_valid been set */
        } ub;
    };
    uint32_t resv[8U];
};

struct MemRegInfo {
    struct MemKey key;
    union {
        struct {
            uint32_t tokenId;
            uint64_t targetSegHandle; /**< refer to urma_target_seg_t */
        } ub;
    };
    uint32_t resv[8U];
};

struct MrRegInfoT {
    struct MemRegAttr in;
    struct MemRegInfo out;
};

union ImportSegFlag {
    struct {
        uint32_t cacheable : 1; /* 0: URMA_NON_CACHEABLE.
                                   1: URMA_CACHEABLE. */
        uint32_t access : 6;    /**< refer to enum mem_seg_access_flags */
        uint32_t mapping : 1;   /* 0: URMA_SEG_NOMAP/
                                   1: URMA_SEG_MAPPED. */
        uint32_t reserved : 24;
    } bs;
    uint32_t value;
};

struct MemImportAttr {
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

struct MemImportInfo {
    union {
        struct {
            uint64_t targetSegHandle; /**< refer to urma_target_seg_t */
        } ub;
    };
    uint32_t resv[4U];
};

struct MrImportInfoT {
    struct MemImportAttr in;
    struct MemImportInfo out;
};

struct HccpTokenId {
    uint32_t tokenId;
};

enum JfcMode {
    JFC_MODE_NORMAL = 0,     /* Corresponding jetty mode：JETTY_MODE_URMA_NORMAL and JETTY_MODE_USER_CTL_NORMAL */
    JFC_MODE_STARS_POLL = 1, /* Corresponding jetty mode：JETTY_MODE_CACHE_LOCK_DWQE and JETTY_MODE_USER_CTL_NORMAL */
    JFC_MODE_CCU_POLL = 2,   /* Corresponding jetty mode: JETTY_MODE_CCU */
    JFC_MODE_USER_CTL_NORMAL = 3, /* Corresponding jetty mode: JETTY_MODE_USER_CTL_NORMAL */
    JFC_MODE_MAX
};

union JfcFlag {
    struct {
        uint32_t lockFree : 1;
        uint32_t jfcInline : 1;
        uint32_t reserved : 30;
    } bs;
    uint32_t value;
};

struct CqCreateAttr {
    void *chanHandle;
    uint32_t depth;
    union {
        struct {
            uint64_t userCtx;
            enum JfcMode mode;
            uint32_t ceqn;
            union JfcFlag flag; /**< refer to urma_jfc_flag_t */
            struct {
                bool valid;
                uint32_t cqeFlag; /* Indicates whether the jfc is handling the current die or cross-die CCU CQE */
            } ccuExCfg;
        } ub;
    };
};

struct CqCreateInfo {
    uint64_t va; /**< refer to struct urma_jfc*, struct ibv_cq* */
    uint32_t id; /**< jfc id */
    uint32_t cqeSize;
    uint64_t bufAddr;
    uint64_t swdbAddr;
};

struct CqInfoT {
    struct CqCreateAttr in;
    struct CqCreateInfo out;
};

union DataPlaneCstmFlag {
    struct {
        uint32_t pollCqCstm : 1; // 0: hccp poll cq; 1: caller poll cq
        uint32_t reserved : 31;
    } bs;
    uint32_t value;
};

struct ChanInfoT {
    struct {
        union DataPlaneCstmFlag dataPlaneFlag;
    } in;
    struct {
        int fd;
    } out;
};

enum JettyMode {
    JETTY_MODE_URMA_NORMAL = 0,     /* jetty_id belongs to [0, 1023] */
    JETTY_MODE_CACHE_LOCK_DWQE = 1, /* jetty_id belongs to [1216, 5311] */
    JETTY_MODE_CCU = 2,             /* jetty_id belongs to [1024, 1151] */
    JETTY_MODE_USER_CTL_NORMAL = 3, /* jetty_id belongs to [5312, 9407] */
    JETTY_MODE_CCU_TA_CACHE = 4,    /* jetty_id belongs to [1024, 1151] */
    JETTY_MODE_MAX
};

union JettyFlag {
    struct {
        uint32_t shareJfr : 1; /* 0: URMA_NO_SHARE_JFR.
                                   1: URMA_SHARE_JFR.   */
        uint32_t reserved : 31;
    } bs;
    uint32_t value;
};

union JfsFlag {
    struct {
        uint32_t lockFree : 1;     /* default as 0, lock protected */
        uint32_t errorSuspend : 1; /* 0: error continue; 1: error suspend */
        uint32_t outorderComp : 1; /* 0: not support; 1: support out-of-order completion */
        uint32_t orderType : 8;    /* (0x0): default, auto config by driver */
                                   /* (0x1): OT, target ordering */
                                   /* (0x2): OI, initiator ordering */
                                   /* (0x3): OL, low layer ordering */
                                   /* (0x4): UNO, unreliable non ordering */
        uint32_t multiPath : 1;    /* 1: multi-path, 0: single path, for ubagg only. */
        uint32_t reserved : 20;
    } bs;
    uint32_t value;
};

struct JettyQueCfgEx {
    uint32_t buffSize;
    uint64_t buffVa;
};

union CstmJfsFlag {
    struct {
        uint32_t sqCstm : 1; /**< valid in jetty mode: JETTY_MODE_CCU */
        uint32_t dbCstm : 1;
        uint32_t dbCtlCstm : 1;
        uint32_t reserved : 29;
    } bs;
    uint32_t value;
};

struct QpCreateAttr {
    void *scqHandle;
    void *rcqHandle;
    void *srqHandle;

    uint32_t sqDepth;
    uint32_t rqDepth;

    enum TransportModeT transportMode;

    union {
        struct {
            enum JettyMode mode;
            uint32_t jettyId;      /**< [optional] user specified jetty id, 0 means not specified */
            union JettyFlag flag;  /**< refer to union urma_jetty_flag */
            union JfsFlag jfsFlag; /**< refer to union urma_jfs_flag; jfs_cfg->flag */
            void *tokenIdHandle;   /**< NULL means unspecified */
            uint32_t tokenValue;   /**< refer to urma_token_t; jfr_cfg->token_value */
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
                    bool lockFlag;      /**< false: unlocked, true: locked by buffer. forced: true. */
                    uint32_t sqeBufIdx; /* base sqe index */
                } taCacheMode;
            };
        } ub;
    };

    uint32_t resv[16U];
};

#define DEV_QP_KEY_SIZE 64

struct QpKey {
    // ROCE: qpn(4), psn(4), gid_idx(4), gid(16), tc(4), sl(4), retry_cnt(4), timeout(4) etc
    // UB: jetty_id(24), trans_mode(4)
    uint8_t value[DEV_QP_KEY_SIZE];
    uint8_t size;
};

struct CtxQpShareInfo {
    uint16_t ciVal;
    uint16_t resv;
    uint8_t rawCqe[64U];
};

struct QpCreateInfo {
    struct QpKey key; /**< for modify qp or import & bind jetty*/
    union {
        struct {
            uint32_t uasid;
            uint32_t id;        /**< jetty id */
            uint64_t sqBuffVa;  /**< valid in jetty mode：JETTY_MODE_CACHE_LOCK_DWQE and JETTY_MODE_USER_CTL_NORMAL */
            uint64_t wqebbSize; /**< valid in jetty mode: JETTY_MODE_CACHE_LOCK_DWQE and JETTY_MODE_USER_CTL_NORMAL */
            uint64_t dbAddr;
            uint32_t dbTokenId;
            uint32_t shareInfoLen;  /**< refer to struct ctx_qp_share_info */
            uint64_t shareInfoAddr; /**< refer to struct ctx_qp_share_info */
        } ub;
    };
    uint64_t va; /**< refer to struct urma_jetty*, struct ibv_qp* */
    uint32_t resv[16U];
};

enum JettyGrpPolicy { JETTY_GRP_POLICY_RR = 0, JETTY_GRP_POLICY_HASH_HINT = 1, JETTY_GRP_POLICY_MAX };

enum TargetType { TARGET_TYPE_JFR = 0, TARGET_TYPE_JETTY = 1, TARGET_TYPE_JETTY_GROUP = 2, TARGET_TYPE_MAX };

enum {
    TOKEN_POLICY_NONE = 0,
    TOKEN_POLICY_PLAIN_TEXT = 1,
    TOKEN_POLICY_SIGNED = 2,
    TOKEN_POLICY_ALL_ENCRYPTED = 3,
    TOKEN_POLICY_RESERVED
};

union ImportJettyFlag {
    struct {
        uint32_t tokenPolicy : 3;
        uint32_t orderType : 8; /* (0x0): default, auto config by driver */
                                /* (0x1): OT, target ordering */
                                /* (0x2): OI, initiator ordering */
                                /* (0x3): OL, low layer ordering */
                                /* (0x4): UNO, unreliable non ordering */
        uint32_t shareTp : 1;   /* 1: shared tp; 0: non-shared tp. When rc mode is not ta dst ordering,
                                    this flag can only be set to 0. */
        uint32_t reserved : 20;
    } bs;
    uint32_t value;
};

enum JettyImportMode { JETTY_IMPORT_MODE_NORMAL = 0, JETTY_IMPORT_MODE_EXP = 1, JETTY_IMPORT_MODE_MAX };

struct JettyImportExpCfg {
    uint64_t tpHandle;
    uint64_t peerTpHandle;
    uint64_t tag;
    uint32_t txPsn;
    uint32_t rxPsn;
    uint32_t rsv[16U];
};

struct QpImportAttr {
    struct QpKey key; /**< for RDMA, save key on rem_qp_handle for bind to modify qp */
    union {
        struct {
            enum JettyImportMode mode;
            uint32_t tokenValue;                   /**< refer to urma_token_t */
            enum JettyGrpPolicy policy;            /**< refer to urma_jetty_grp_policy_t */
            enum TargetType type;                  /**< refer to urma_target_type */
            union ImportJettyFlag flag;            /**< refer to urma_import_jetty_flag_t */
            struct JettyImportExpCfg expImportCfg; /**< only valid on mode JETTY_IMPORT_MODE_EXP */
            uint32_t tpType;                       /**< refer to urma_tp_type_t */
        } ub;
    };
    uint32_t resv[7U];
};

struct QpImportInfo {
    union {
        struct {
            uint64_t tjettyHandle; /**< refer to urma_target_jetty_t *tjetty */
            uint32_t tpn;          /**< refer to urma_tp_t tp */
        } ub;
    };
    uint32_t resv[8U];
};

struct QpImportInfoT {
    struct QpImportAttr in;
    struct QpImportInfo out;
};

enum JettyAttrMask { JETTY_ATTR_RX_THRESHOLD = 0x1, JETTY_ATTR_STATE = 0x1 << 1 };

enum JettyState { JETTY_STATE_RESET = 0, JETTY_STATE_READY, JETTY_STATE_SUSPENDED, JETTY_STATE_ERROR };

struct JettyAttr {
    uint32_t mask; // mask value refer to enum jetty_attr_mask
    uint32_t rxThreshold;
    enum JettyState state;
    uint32_t resv[2U];
};

#define HCCP_MAX_QP_QUERY_NUM 128U
#define HCCP_MAX_QP_DESTROY_BATCH_NUM 768U

struct CtxNotifyEvent {
    uint8_t resv0;
    uint8_t resv1;
    uint8_t serviceType;
    uint8_t errorType;
    union HccpEid srcEid;
    union HccpEid dstEid;
    uint32_t resv2;
    union {
        uint32_t tpn; // corresponding serviceType URMA_TYPE:0
        uint64_t hpa; // corresponding serviceType UBMEM_TYPE:1
    } errorInfo;
};

/**
 * @ingroup libudma
 * @brief get total dev eid info num
 * @param info [IN] see RaInfo
 * @param num [OUT] num of dev eid info
 * @see RaGetDevEidInfoList
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetDevEidInfoNum(struct RaInfo info, unsigned int *num);

/**
 * @ingroup libudma
 * @brief get all dev info list
 * @param info [IN] see RaInfo
 * @param info_list [IN/OUT] dev eid info list
 * @param num [IN/OUT] num of dev eid info list
 * @see RaGetDevEidInfoNum
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetDevEidInfoList(struct RaInfo info, struct HccpDevEidInfo infoList[], unsigned int *num);

/**
 * @ingroup libudma
 * @brief ctx initialization will start lite thread by default
 * @param cfg [IN] ctx init cfg
 * @param attr [IN] ctx init attr
 * @param ctx_handle [OUT] ctx handle
 * @see ra_ctx_deinit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxInit(struct CtxInitCfg *cfg, struct CtxInitAttr *attr, void **ctxHandle);

/**
 * @ingroup libudma
 * @brief get dev base attr
 * @param ctx_handle [IN] ctx handle
 * @param attr [OUT] dev base attr
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetDevBaseAttr(void *ctxHandle, struct DevBaseAttr *attr);

/**
 * @ingroup libudma
 * @brief get corresponding eid by ip
 * @param ctx_handle [IN] ctx handle
 * @param ip [IN] ip array, see struct IpInfo
 * @param eid [IN/OUT] eid array, see union HccpEid
 * @param num [IN/OUT] num of ip and eid array, max num is HCCP_EID_IP_QUERY_MAX_NUM
 * @see ra_ctx_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetEidByIp(void *ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int *num);

/**
 * @ingroup libudma
 * @brief get corresponding ip by eid
 * @param ctx_handle [IN] ctx handle
 * @param eid [IN] eid array, see union HccpEid
 * @param ip [IN/OUT] ip array, see struct IpInfo
 * @param num [IN/OUT] num of ip and eid array, max num is HCCP_EID_IP_QUERY_MAX_NUM
 * @see ra_ctx_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetIpByEid(void *ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int *num);

/**
 * @ingroup libudma
 * @brief ctx deinitialization
 * @param ctx_handle [IN] ctx handle
 * @see ra_ctx_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxDeinit(void *ctxHandle);

/**
 * @ingroup libudma
 * @brief alloc token id
 * @param ctx_handle [IN] ctx handle
 * @param info [OUT] see struct hccp_token_id
 * @param token_id_handle [OUT] token id handle
 * @see ra_ctx_token_id_free
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxTokenIdAlloc(void *ctxHandle, struct HccpTokenId *info, void **tokenIdHandle);

/**
 * @ingroup libudma
 * @brief free token id
 * @param ctx_handle [IN] ctx handle
 * @param token_id_handle [IN] token id handle
 * @see ra_ctx_token_id_alloc
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxTokenIdFree(void *ctxHandle, void *tokenIdHandle);

/**
 * @ingroup libudma
 * @brief register local mem
 * @param ctx_handle [IN] ctx handle
 * @param lmem_info [IN/OUT] lmem reg info
 * @param lmem_handle [OUT] lmem handle
 * @see ra_ctx_lmem_unregister
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxLmemRegister(void *ctxHandle, struct MrRegInfoT *lmemInfo, void **lmemHandle);

/**
 * @ingroup libudma
 * @brief unregister local mem
 * @param ctx_handle [IN] ctx handle
 * @param lmem_handle [IN] lmem handle
 * @see ra_ctx_lmem_register
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxLmemUnregister(void *ctxHandle, void *lmemHandle);

/**
 * @ingroup libudma
 * @brief import remote mem
 * @param ctx_handle [IN] ctx handle
 * @param rmem_info [IN/OUT] rmem info
 * @param rmem_handle [OUT] rmem handle, key as rkey for send wr
 * @see ra_ctx_rmem_unimport
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxRmemImport(void *ctxHandle, struct MrImportInfoT *rmemInfo, void **rmemHandle);

/**
 * @ingroup libudma
 * @brief unimport remote mem
 * @param ctx_handle [IN] ctx handle
 * @param rmem_handle [IN] rmem handle
 * @see ra_ctx_rmem_import
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxRmemUnimport(void *ctxHandle, void *rmemHandle);

/**
 * @ingroup libudma
 * @brief  create comp channel
 * @param ctx_handle [IN] ctx handle
 * @param chan_info [IN/OUT] see chan_info_t
 * @param chan_handle [OUT] comp chan handle
 * @see ra_ctx_chan_destroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxChanCreate(void *ctxHandle, struct ChanInfoT *chanInfo, void **chanHandle);

/**
 * @ingroup libudma
 * @brief  destroy comp channel
 * @param ctx_handle [IN] ctx handle
 * @param chan_handle [IN] comp chan handle
 * @see ra_ctx_chan_create
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxChanDestroy(void *ctxHandle, void *chanHandle);

/**
 * @ingroup libudma
 * @brief create jfc/cq
 * @param ctx_handle [IN] ctx handle
 * @param info [IN/OUT] cq info
 * @param cq_handle [OUT] cq handle
 * @see ra_ctx_cq_destroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxCqCreate(void *ctxHandle, struct CqInfoT *info, void **cqHandle);

/**
 * @ingroup libudma
 * @brief destroy jfc/cq
 * @param ctx_handle [IN] ctx handle
 * @param cq_handle [IN] cq handle
 * @see ra_ctx_cq_create
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxCqDestroy(void *ctxHandle, void *cqHandle);

/**
 * @ingroup libudma
 * @brief create jetty/qp
 * @param ctx_handle [IN] ctx handle
 * @param attr [IN] qp attr
 * @param info [OUT] qp info
 * @param qp_handle [OUT] qp handle
 * @see ra_ctx_qp_destroy
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxQpCreate(void *ctxHandle, struct QpCreateAttr *attr, struct QpCreateInfo *info,
    void **qpHandle);

/**
 * @ingroup libudma
 * @brief batch query jetty attr
 * @param qp_handle [IN] qp handle
 * @param attr [IN/OUT] see struct jetty_attr
 * @param num [IN/OUT] size of qp_handle and attr array, max num is HCCP_MAX_QP_QUERY_NUM
 * @see ra_ctx_qp_create
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxQpQueryBatch(void *qpHandle[], struct JettyAttr attr[], unsigned int *num);

/**
 * @ingroup libudma
 * @brief destroy jetty/qp
 * @param qp_handle [IN] qp handle
 * @see ra_ctx_qp_create
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxQpDestroy(void *qpHandle);

/**
 * @ingroup libudma
 * @brief get tp info list
 * @param ctxHandle [IN] ctx handle
 * @param cfg [IN] get tp cfg
 * @param infoList [IN/OUT] corresponding tp info list
 * @param num [IN/OUT] size of infoList, max num is HCCP_MAX_TPID_INFO_NUM
 * @see RaCtxInit
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxGetTpInfoList(void *ctxHandle, struct GetTpCfg *cfg, struct HccpTpInfo infoList[],
    unsigned int *num);

/**
 * @ingroup libudma
 * @brief get tp attr
 * @param ctxHandle [IN] ctx handle
 * @param tpHandle [IN] see struct tp_info
 * @param attrBitmap [IN/OUT] see struct TpAttr
 * @param attr [IN/OUT] see struct TpAttr
 * @see RaCtxGetTpInfoList
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxGetTpAttr(void *ctxHandle, uint64_t tpHandle, uint32_t *attrBitmap, struct TpAttr *attr);

/**
 * @ingroup libudma
 * @brief set tp attr
 * @param ctxHandle [IN] ctx handle
 * @param tpHandle [IN] see struct tp_info
 * @param attrBitmap [IN] see struct TpAttr
 * @param attr [IN] see struct TpAttr
 * @see RaCtxGetTpInfoList
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxSetTpAttr(void *ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr *attr);

/**
 * @ingroup libudma
 * @brief import jetty/prepare rem_qp_handle for modify qp
 * @param ctx_handle [IN] ctx handle
 * @param qp_info [IN/OUT] qp import info
 * @param rem_qp_handle [OUT] remote qp handle
 * @see ra_ctx_qp_unimport
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxQpImport(void *ctxHandle, struct QpImportInfoT *qpInfo, void **remQpHandle);

/**
 * @ingroup libudma
 * @brief unimport jetty
 * @param ctx_handle [IN] ctx handle
 * @param rem_qp_handle [IN] qp handle
 * @see ra_ctx_qp_import
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxQpUnimport(void *ctxHandle, void *remQpHandle);

/**
 * @ingroup libudma
 * @brief bind jetty/modify qp
 * @param qp_handle [IN] qp handle
 * @param rem_qp_handle [IN] rem qp handle
 * @see ra_ctx_qp_unbind
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxQpBind(void *qpHandle, void *remQpHandle);

/**
 * @ingroup libudma
 * @brief unbind jetty
 * @param qp_handle [IN] qp handle
 * @see ra_ctx_qp_bind
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxQpUnbind(void *qpHandle);

/**
 * @ingroup libudma
 * @brief get async event
 * @param ctx_handle [IN] ctx handle
 * @param events [IN/OUT] see struct async_event
 * @param num [IN/OUT] num of events, max num is ASYNC_EVENT_MAX_NUM
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxGetAsyncEvents(void *ctxHandle, struct AsyncEvent events[], unsigned int *num);

/**
 * @ingroup libudma
 * @brief get aux info
 * @param ctx_handle [IN] ctx handle
 * @param in [IN] see struct aux_info_in
 * @param out [OUT] see struct aux_info_out
 * @see ra_ctx_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxGetAuxInfo(void *ctxHandle, struct HccpAuxInfoIn *in, struct HccpAuxInfoOut *out);

/**
 * @ingroup libudma
 * @brief get cr err info by ctx_handle
 * @param ctx_handle [IN] ctx_handle
 * @param info_list [IN/OUT] cr err info
 * @param num [IN/OUT] num of cr err info, max num of input is CR_ERR_INFO_MAX_NUM
 * @see ra_ctx_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxGetCrErrInfoList(void *ctxHandle, struct CrErrInfo *infoList, unsigned int *num);

/**
 * @ingroup libudma
 * @brief get jetty context info by qpHandle
 * @param qpHandle [IN] qpHandle
 * @param context [IN/OUT] context buffer, max len is CONTEXT_MAX_LEN
 * @param len [IN/OUT] len of context
 * @see RaCtxQpCreate
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxGetJettyContext(void *qpHandle, uint8_t context[], unsigned int *len);

/**
 * @ingroup libudma
 * @brief notify event to aubdfx
 * @param ctx_handle [IN] ctx handle
 * @param event [IN] event info, see struct CtxNotifyEvent
 * @see ra_ctx_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxNotifyEvent(void *ctxHandle, struct CtxNotifyEvent *event);
#ifdef __cplusplus
}
#endif

#endif // HCCP_CTX_H
