/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IBV_EXTEND_H
#define IBV_EXTEND_H

#include <infiniband/verbs.h>
#include <stdint.h>

/*
 * 版本号说明：
 * - 主版本号(MAJOR): 不兼容的API变更
 * - 次版本号(MINOR): 后向兼容的功能新增
 * - 修订号(PATCH): 后向兼容的问题修复
 *
 * 兼容性规则：
 * 1. 驱动和业务编译时的头文件版本必须 <= 运行时库的版本
 * 2. 主版本号必须完全一致
 * 3. 次版本号：驱动 <= 库
 */
#define IBV_EXTEND_VERSION_MAJOR 4
#define IBV_EXTEND_VERSION_MINOR 2
#define IBV_EXTEND_VERSION_PATCH 0
#define IBV_EXTEND_VERSION_STRING "4.2.0"

/*
 * 驱动操作接口版本号定义
 *
 * 版本号用于标识驱动实现的ops集合版本，确保上层库与底层驱动的兼容性。
 * 每个版本对应一组固定的函数指针集合，新增接口时需要递增版本号。
 *
 * 版本兼容性规则：
 * 1. 驱动注册时必须设置正确的版本号，并初始化该版本支持的所有函数指针
 * 2. 上层库调用接口前会检查驱动版本，版本不匹配时返回 -EOPNOTSUPP
 * 3. 老驱动可以在新扩展库上运行，但只能使用老驱动支持的接口
 * 4. 新驱动不能在老扩展库上运行，需要扩展库升级支持
 *
 */
enum IBV_EXTEND_DRIVER_VERSION {
    IBV_EXTEND_DRIVER_VERSION_UNUSED = 0,
    IBV_EXTEND_DRIVER_VERSION_V1 = 1, /* 版本1: 支持NPU Direct Async基础接口 */
    IBV_EXTEND_DRIVER_VERSION_V2 = 2, /* 版本2: 新增Hyper RoCE接口 */
    IBV_EXTEND_DRIVER_VERSION_V3 = 3, /* 版本3: 新增LAG Port与Hyper RoCE协商接口 */
    /* 当前最新版本，驱动实现全部接口时使用 */
    IBV_EXTEND_DRIVER_VERSION_MAX = IBV_EXTEND_DRIVER_VERSION_V3
};

/* 扩展上下文结构 */
struct ibv_context_extend {
    struct ibv_context *context;        /* 标准RDMA设备上下文 */
    struct ibv_context_extend_ops *ops; /* 扩展操作接口函数指针集合 */
};

/* ==版本兼容性查询接口== */
/**
 * @brief 获取库版本号
 * @param major 主版本号输出参数（可为NULL）
 * @param minor 次版本号输出参数（可为NULL）
 * @param patch 修订号输出参数（可为NULL）
 * @return 版本字符串指针（静态字符串，无需释放）
 */
const char *ibv_extend_get_version(uint32_t *major, uint32_t *minor, uint32_t *patch);

/**
 * @brief 检查版本兼容性
 * @param driver_major 驱动编译时的主版本号
 * @param driver_minor 驱动编译时的次版本号
 * @param driver_patch 驱动编译时的修订号
 * @return 0-兼容，-1-不兼容
 *
 * 此函数在运行时检查驱动编译时使用的头文件版本与当前库版本是否兼容。
 * 建议在驱动和上层应用初始化时调用此函数进行版本检查。
 */
int ibv_extend_check_version(uint32_t driver_major, uint32_t driver_minor, uint32_t driver_patch);

/* 北向接口 */
/**
 * @brief 打开扩展上下文，初始化NDA扩展功能
 * @param context RDMA设备上下文指针
 * @return NULL-打开失败，Non-NULL-打开成功，返回扩展上下文指针
 */
struct ibv_context_extend *ibv_open_extend(struct ibv_context *context);

/**
 * @brief 关闭扩展上下文，释放NDA扩展资源
 * @param context 扩展上下文指针
 * @return 0-成功，其他值-失败
 */
int ibv_close_extend(struct ibv_context_extend *context);

/*
 * -----------------------
 *    NPU Direct Async
 * -----------------------
 */
/* RoCE队列内存的DMA寻址方式 */
enum queue_buf_dma_mode {
    QU_BUF_DMA_MODE_DEFAULT = 0, /* 采用Host所在总线的DMA, NIC - PCIe - NPU */
    QU_BUF_DMA_MODE_INDEP_UB,    /* 采用非Host的独立UB总线做DMA, NIC - UB - NPU */
};

/* Doorbell的地址映射方式 */
enum doorbell_map_mode {
    DB_MAP_MODE_HOST_VA = 0, /* 基于host VA映射 */
    DB_MAP_MODE_UB_RES,      /* 基于UB设备的资源描述映射 */
};

/* 内存拷贝方向 */
enum memcpy_direction {
    MEMCPY_DIR_HOST_TO_HOST = 0, // Host内复制
    MEMCPY_DIR_HOST_TO_DEVICE,   // Host到Device的内存复制
    MEMCPY_DIR_DEVICE_TO_HOST,   // Device到Host的内存复制
    MEMCPY_DIR_DEVICE_TO_DEVICE, // Device内或Device间的内存复制
};

/* QP初始化扩展属性 */
enum ibv_qp_init_cap {
    QP_ENABLE_DIRECT_WQE = 1 << 0, /* SQ启用direct_wqe能力 */
};

/* 扩展设备属性 */
enum ibv_extend_device_cap {
    IBV_EXTEND_DEV_NDR = 1 << 0, // 网卡设备支持NDR
    IBV_EXTEND_DEV_NDA = 1 << 1, // 网卡设备支持NDA
};

/* 用于映射doorbell资源到设备侧 */
struct doorbell_map_desc {
    union {
        uint64_t hva;
        struct {
            uint64_t guid_l;
            uint64_t guid_h;
            struct {
                uint64_t resource_id : 4;
                uint64_t offset : 32; /* 单位：字节 */
                uint64_t resv : 28;
            } bits;
        } ub_res;
    };
    uint64_t size; /* 单位：字节 */
    uint32_t type; /* doorbell映射的寻址模式, 对应doorbell_map_mode枚举变量 */

    uint32_t resv; /* 扩展专用，padding */
};

/* 回调函数集合 */
struct ibv_extend_ops {
    void *(*alloc)(size_t size); /* 申请NPU内存，网卡创建QP时回调，返回内存指针 */
    void (*free)(void *ptr);     /* 释放内存 */

    void (*memset_s)(void *dst, int value, size_t count); /* 初始化NPU内存内容 */
    /* 内存拷贝，需支持不同方向实现，direct类型为 memcpy_direction */
    int (*memcpy_s)(void *dst, size_t dst_max_size, void *src, size_t size, uint32_t direct);

    void *(*db_mmap)(struct doorbell_map_desc *desc); /* 返回映射的起始内存地址，需允许对相同desc的重复map */
    int (*db_unmap)(void *ptr, struct doorbell_map_desc *desc); /* 解除地址映射，需对应的输入描述符传入 */
};

struct iov_addr_desc {
    void *iov_base;
    size_t iov_len;
};

/* 队列buffer信息 */
struct queue_buf {
    uint64_t base;       /* 队列的首地址 */
    uint32_t entry_cnt;  /* 队列中entry个数, 表示WQE/CQE个数 */
    uint32_t entry_size; /* 队列中entry大小, 表示WQE/CQE大小 */
    uint64_t resv[4];    /* 扩展专用 */
};

struct queue_info {
    struct queue_buf qbuf;          /* 队列buffer */
    struct iov_addr_desc dbr_pi_va; /* doorbell recode PI地址, SQ/RQ硬件使用, CQ软件使用 */
    struct iov_addr_desc dbr_ci_va; /* doorbell recode CI地址, SQ/RQ软件使用，CQ硬件使用 */
    struct iov_addr_desc db_hw_va;  /* map之后的doorbell地址 */
    uint64_t resv[4];               /* 扩展专用 */
};

/* QP输出参数 */
struct ibv_qp_extend {
    struct ibv_qp *qp;
    struct queue_info sq_info; /* 发送队列的buffer */
    struct queue_info rq_info; /* 接收队列的buffer */
    uint64_t resv[31];         /* 扩展专用 */
    union {
        struct {
            uint8_t db_cos; /* modify_qp时由驱动刷新db_cos */
            uint8_t resv[7];
        } vendor1;
        uint64_t value;
    } vendor_priv_info; /* 厂商私有信息 */
};

/* CQ输出参数 */
struct ibv_cq_extend {
    struct ibv_cq *cq;
    struct queue_info cq_info;
    uint64_t resv[32]; /* 扩展专用 */
};

/* SRQ输出参数 */
struct ibv_srq_extend {
    struct ibv_srq *srq;
    struct queue_info srq_info;

    uint64_t resv[32]; /* 扩展专用 */
};

/* QP输入参数 */
struct ibv_qp_init_attr_extend {
    struct ibv_pd *pd;            /* 默认参数 */
    struct ibv_qp_init_attr attr; /* 默认参数 */

    uint32_t qp_cap_flag;         /* qp 配置, ibv_qp_init_cap 类型 */
    enum queue_buf_dma_mode type; /* DMA mode */
    struct ibv_extend_ops *ops;   /* 通过入参传递 */

    uint64_t resv[8]; /* 扩展专用 */
};

/* CQ输入参数 */
struct ibv_cq_init_attr_extend {
    struct ibv_cq_init_attr_ex attr; /* 默认参数 */

    uint32_t cq_cap_flag;         /* cq 配置 */
    enum queue_buf_dma_mode type; /* DMA mode */
    struct ibv_extend_ops *ops;   /* 通过入参传递 */

    uint64_t resv[8]; /* 扩展专用 */
};

/* SRQ输入参数 */
struct ibv_srq_init_attr_extend {
    struct ibv_pd *pd;             /* 默认参数 */
    struct ibv_srq_init_attr attr; /* 默认参数 */

    uint32_t comp_mask; /* compatibility mask */

    uint32_t srq_cap_flag;        /* srq 配置 */
    enum queue_buf_dma_mode type; /* DMA mode */
    struct ibv_extend_ops *ops;   /* 通过入参传递 */

    uint64_t resv[8]; /* 扩展专用 */
};

/* 扩展设备属性 */
struct ibv_device_attr_extend {
    uint32_t ext_cap; /* ibv_extend_device_cap 类型 */

    uint32_t resv[32]; /* 扩展专用 */
};

/**
 * @brief 查询设备支持的扩展能力
 * @param context 扩展上下文指针
 * @param ext_dev_attr 返回设备支持的扩展能力
 * @return 0-成功，其他值-失败
 */
int ibv_query_device_extend(struct ibv_context_extend *context, struct ibv_device_attr_extend *ext_dev_attr);

/**
 * @brief ibv_create_qp扩展接口，支持NDA场景创建qp
 * @param context 扩展上下文指针
 * @param qp_init_attr 创建qp的属性
 * @attention qp_init_attr需要指定对应的内存类型和内存操作集，该信息要保存到qp_ctx后续使用。
 * @return Null-创建失败，Non-Null-创建成功，返回qp的句柄和内存信息
 */
struct ibv_qp_extend *ibv_create_qp_extend(struct ibv_context_extend *context,
    struct ibv_qp_init_attr_extend *qp_init_attr);

/**
 * @brief ibv_create_cq扩展接口，支持NDA场景创建cq
 * @param context 扩展上下文指针
 * @param cq_init_attr 创建cq的属性
 * @attention cq_init_attr需要指定对应的内存类型和内存操作集，该信息要保存到cq_ctx后续使用。
 * @return Null-创建失败，Non-Null-创建成功，返回cq的句柄和内存信息
 */
struct ibv_cq_extend *ibv_create_cq_extend(struct ibv_context_extend *context,
    struct ibv_cq_init_attr_extend *cq_init_attr);

/* ibv_create_srq扩展接口，支持NDA场景创建srq */
/**
 * @brief ibv_create_srq扩展接口，支持NDA场景创建srq
 * @param context 扩展上下文指针
 * @param srq_init_attr 创建srq的属性
 * @attention srq_init_attr需要指定对应的内存类型和内存操作集，该信息要保存到srq_ctx后续使用。
 * @return Null-创建失败，Non-Null-创建成功，返回srq的句柄和内存信息
 */
struct ibv_srq_extend *ibv_create_srq_extend(struct ibv_context_extend *context,
    struct ibv_srq_init_attr_extend *srq_init_attr);

/**
 * @brief ibv_destroy_qp扩展接口，支持NDA场景销毁qp
 * @param context 扩展上下文指针
 * @param qp_extend 创建接口返回的qp句柄
 * @attention void
 * @return 接口返回值 0-success，other-fail
 */
int ibv_destroy_qp_extend(struct ibv_context_extend *context, struct ibv_qp_extend *qp_extend);

/**
 * @brief ibv_destroy_cq扩展接口，支持NDA场景销毁cq
 * @param context 扩展上下文指针
 * @param cq_extend 创建接口返回的cq句柄
 * @attention void
 * @return 接口返回值 0-success，other-fail
 */
int ibv_destroy_cq_extend(struct ibv_context_extend *context, struct ibv_cq_extend *cq_extend);

/**
 * @brief ibv_destroy_srq扩展接口，支持NDA场景销毁srq
 * @param context 扩展上下文指针
 * @param srq_extend 创建接口返回的srq句柄
 * @attention void
 * @return 接口返回值 0-success，other-fail
 */
int ibv_destroy_srq_extend(struct ibv_context_extend *context, struct ibv_srq_extend *srq_extend);

/*
 * -----------------------
 *      Hyper RoCE
 * -----------------------
 */
/* 高阶RoCE特性类型枚举 */
enum ibv_hyroce_feature_type {
    IBV_HYPER_FEAT_RoCEv2 = 0, /* 标准RoCEv2协议 */
    IBV_HYPER_FEAT_VEROCE,     /* VelcEngine RoCE */
    IBV_HYPER_FEAT_HYPER_ROCE, /* Hyper RoCE */
};

/* 高阶RoCE特性版本枚举 */
enum ibv_hyroce_feature_version {
    IBV_HYPER_FEAT_V0 = 0, /* V0版本，特定表示标准RoCEv2 */
    IBV_HYPER_FEAT_V1,     /* V1版本 */
    IBV_HYPER_FEAT_V2,     /* V2版本 */
    IBV_HYPER_FEAT_V3      /* V3版本 */
};

/* 负载均衡模式 */
enum ibv_lb_mode {
    IBV_LB_MODE_DEFAULT = 0, /* 网卡默认负载均衡模式 */
    IBV_LB_MODE_MPATH,       /* Multi-Path多路径 */
    IBV_LB_MODE_AR           /* Adaptive-Routing自适应 */
};

/* QP属性扩展掩码，用于指定ibv_modify_qp_extend和ibv_query_qp_extend需要配置/查询的属性 */
enum ibv_qp_attr_extend_mask {
    IBV_QP_ATTR_EXTEND_UDP_SRC_PORT = 1 << 0,   /* 源UDP端口号 */
    IBV_QP_ATTR_EXTEND_HYROCE_FEATURE = 1 << 1, /* 高阶RoCE的特性 */
    IBV_QP_ATTR_EXTEND_LB_MODE = 1 << 2,        /* 负载均衡模式 */
    IBV_QP_ATTR_EXTEND_MPATH_CONFIG = 1 << 3,   /* Multi-Path多路径模式配置 */
    IBV_QP_ATTR_EXTEND_AR_CONFIG = 1 << 4,      /* Adaptive-Routing多路径模式配置 */
    IBV_QP_ATTR_EXTEND_SACK_CONFIG = 1 << 5,    /* Selective Ack选择性重传参数配置 */
    IBV_QP_ATTR_EXTEND_LAG_PORT = 1 << 6,       /* 用于bond模式下QP绑定的网络端口 */
    IBV_QP_ATTR_EXTEND_MAX = 1 << 7
};

/* 高阶RoCE特性配置结构体 */
struct ibv_hyroce_feature {
    uint8_t type;        /* 高阶RoCE类型，取值范围：enum ibv_hyroce_feature_type */
    uint8_t version;     /* 高阶RoCE版本，取值范围：enum ibv_hyroce_feature_version */
    uint8_t sack_enable; /* 选择性重传开关，0-关闭，1-开启 */

    uint8_t resv[61]; /* 预留字段，用于未来扩展特性 */
};

/* 多路径(Multi-Path)配置参数结构体 */
struct ibv_mpath_config {
    uint32_t flowlet_pkg_num; /* 每个子流(flowlet)的包个数，用于流切分 */
    uint32_t path_num;        /* 多路径的路径个数 */
    uint32_t interval;        /* UDP端口号递增间隔，用于区分不同路径 */
    uint32_t path_rr_enable;  /* 路径是否支持轮询(RR)选择网络端口，0-不支持，1-支持 */

    uint32_t resv[32]; /* 预留字段，用于未来扩展 */
};

/* 自适应路由(Adaptive Routing)配置结构体 */
struct ibv_ar_config {
    uint32_t port_rr_enable; /* 是否开启网口侧端口轮询(RR)逐包功能，0-关闭，1-开启 */

    uint32_t resv[32]; /* 预留字段，用于未来扩展 */
};

/* 选择性重传(SACK - Selective Ack)配置结构体 */
struct ibv_sack_config {
    uint32_t srp_range; /* TX(发送)方向最大重传报文个数 */
    uint32_t oor_range; /* RX(接收)方向乱序重传窗口的报文个数 */

    uint32_t resv[32]; /* 预留字段，用于未来扩展 */
};

/* QP扩展属性结构体，用于ibv_modify_qp_extend和ibv_query_qp_extend操作 */
struct ibv_qp_attr_extend {
    struct ibv_qp *qp;                 /* QP句柄，指向需要修改或查询的QP对象 */
    uint32_t udp_src_port;             /* 源UDP端口号 */
    struct ibv_hyroce_feature feature; /* 高阶RoCE特性 */
    uint32_t lb_mode;                  /* 负载均衡模式，取值范围：enum ibv_lb_mode 类型 */
    struct ibv_mpath_config mpath;     /* 多路径(Multi-Path)配置 */
    struct ibv_ar_config ar;           /* 自适应路由(Adaptive Routing)配置 */
    struct ibv_sack_config sack;       /* 选择性重传(SACK)配置 */
    uint32_t resv[47];                 /* 预留字段，用于未来扩展 */
    uint32_t lag_port;                 /* Bond下QP绑定网口端口 */
};

/**
 * @brief ibv_modify_qp扩展接口，支持NDA场景修改qp属性
 * @param context 扩展上下文指针
 * @param attr QP扩展属性结构体
 * @param attr_mask 属性掩码，指定需要修改的属性，enum ibv_qp_attr_extend_mask 类型集合
 * @return 0-成功，其他值-失败
 */
int ibv_modify_qp_extend(struct ibv_context_extend *context, struct ibv_qp_attr_extend *attr, int attr_mask);

/**
 * @brief ibv_query_qp扩展接口，支持NDA场景查询qp属性
 * @param context 扩展上下文指针
 * @param attr QP扩展属性结构体，用于返回查询结果
 * @param attr_mask 属性掩码，指定需要查询的属性，enum ibv_qp_attr_extend_mask 类型集合
 * @return 0-成功，其他值-失败
 */
int ibv_query_qp_extend(struct ibv_context_extend *context, struct ibv_qp_attr_extend *attr, int attr_mask);

/**
 * @brief 查询网卡支持的高阶RoCE特性
 * @param context 扩展上下文指针
 * @param qp QP指针
 * @param sl 服务等级
 * @param tc 流量控制
 * @param feature 返回支持的高阶RoCE特性
 * @return 0-成功，其他值-失败
 */
int ibv_query_qp_supported_hyroce_feature(struct ibv_context_extend *context, struct ibv_qp *qp, uint32_t sl,
    uint32_t tc, struct ibv_hyroce_feature *feature);

/**
 * @brief 网卡协商高阶RoCE特性
 * @param context 扩展上下文指针
 * @param qp QP指针
 * @param input 输入的高阶RoCE特性
 * @param output 协商完成后高阶RoCE特性
 * @param need_more_nego 是否需要再次协商
 * @return 0-成功，其他值-失败
 */
int ibv_nego_qp_hyroce_feature(struct ibv_context_extend *context, struct ibv_qp *qp,
    const struct ibv_hyroce_feature *input, struct ibv_hyroce_feature *output, uint32_t *need_more_nego);

/* 南向接口：驱动侧需要实现的操作接口集合 */
struct ibv_context_extend_ops {
    /*
     * enum IBV_EXTEND_DRIVER_VERSION 类型
     * 为了保障版本兼容性，驱动需按照当前支持的接口情况，按需确定version
     * 若全部接口均配置支持，可使用 IBV_EXTEND_DRIVER_VERSION_MAX 配置最新版本
     *
     * 版本兼容性规则：
     * 1. 驱动版本必须 <= 扩展库支持的最新版本(IBV_EXTEND_DRIVER_VERSION_MAX)
     * 2. 如果驱动版本 > 扩展库版本，驱动注册会被拒绝，防止结构体越界和内存访问错误
     * 3. 如果驱动版本 <= 库版本，驱动可以正常工作，但只能使用该版本支持的接口
     */
    int version;

    /*
     * NPU Direct Async
     * IBV_EXTEND_VERSION_V1 支持
     */
    struct ibv_qp_extend *(*create_qp)(struct ibv_context *context,
        struct ibv_qp_init_attr_extend *qp_init_attr); /* 创建QP */
    struct ibv_cq_extend *(*create_cq)(struct ibv_context *context,
        struct ibv_cq_init_attr_extend *cq_init_attr); /* 创建CQ */
    struct ibv_srq_extend *(*create_srq)(struct ibv_context *context,
        struct ibv_srq_init_attr_extend *srq_init_attr); /* 创建SRQ */

    int (*destroy_qp)(struct ibv_qp_extend *qp_extend);    /* 销毁QP */
    int (*destroy_cq)(struct ibv_cq_extend *cq_extend);    /* 销毁CQ */
    int (*destroy_srq)(struct ibv_srq_extend *srq_extend); /* 销毁SRQ */

    /* 提供设备的扩展能力查询 */
    int (*query_device)(struct ibv_context *context, struct ibv_device_attr_extend *ext_dev_attr);

    /*
     * Hyper RoCE
     * IBV_EXTEND_VERSION_V2 新增支持
     */
    int (*modify_qp)(struct ibv_context *context, struct ibv_qp_attr_extend *attr, int attr_mask);
    int (*query_qp)(struct ibv_context *context, struct ibv_qp_attr_extend *attr, int attr_mask);

    /*
     * Hyper RoCE 协商接口
     * IBV_EXTEND_VERSION_V3 新增支持
     */
    int (*query_qp_supported_hyroce_feature)(struct ibv_context *context, struct ibv_qp *qp, uint32_t sl, uint32_t tc,
        struct ibv_hyroce_feature *feature);

    int (*nego_qp_hyroce_feature)(struct ibv_context *context, struct ibv_qp *qp,
        const struct ibv_hyroce_feature *input, struct ibv_hyroce_feature *output, uint32_t *need_more_nego);
};

/* 扩展驱动操作接口，用于驱动注册 */
struct verbs_device_extend_ops {
    const char *name; /* 驱动名称，与标准驱动匹配 */

    struct ibv_context_extend *(*alloc_context)(struct ibv_context *context); /* 分配扩展上下文 */
    void (*free_context)(struct ibv_context_extend *context);                 /* 释放扩展上下文 */
};

/**
 * @brief 驱动注册扩展ops
 * @param ops 扩展驱动操作接口结构体指针
 */
void verbs_register_driver_extend(const struct verbs_device_extend_ops *ops);

/**
 * @brief 驱动注册扩展ops宏
 * @param drv 扩展驱动操作接口结构体
 *
 */
#define PROVIDER_EXTEND_DRIVER(drv)                                                                                    \
    static __attribute__((constructor)) void drv##__register_extend_driver(void)                                       \
    {                                                                                                                  \
        verbs_register_driver_extend(&(drv));                                                                          \
    }

#endif // IBV_EXTEND_H
