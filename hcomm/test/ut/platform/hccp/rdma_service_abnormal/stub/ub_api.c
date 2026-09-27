/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <config.h>

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <alloca.h>
#include <string.h>

#include <infiniband/verbs.h>
#include <infiniband/driver.h>
#include "urma_types.h"
#include "urma_types_str.h"

/**
 * Init urma environment.
 * @param[in] [Required] conf: urma init attr, a random uasid will be assigned when conf is null.
 * Return: 0 on success, other value on error
 */
urma_status_t urma_init(urma_init_attr_t* conf) { return 0; }

/**
 * get eid by ip info
 * @param[in] ctx: the created urma context pointer;
 * @param[in] net_addr: the ip info (type and net_addr are valid, vlan, mac, prefix_len will not be used);
 * @param[out] eid: device's eid;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_get_eid_by_ip(const urma_context_t* ctx, const urma_net_addr_t* net_addr, urma_eid_t* eid)
{
    return 0;
}

urma_status_t urma_get_ip_by_eid(const urma_context_t* ctx, const urma_eid_t* eid, urma_net_addr_t* net_addr)
{
    return 0;
}

/**
 * Un-init urma environment, it will free uasid.
 * Return: 0 on success, other value on error
 */
urma_status_t urma_uninit(void) { return 0; }

/* Device Manage API */
/**
 *  Get device list.
 * @param[out] num_devices: number of urma device;
 * Return: pointer array of urma_device; NULL means no device returned;
 * Note: urma_free_device_list() needs to be called to free memory;
 */
urma_device_t** urma_get_device_list(int* num_devices)
{
    urma_device_t* urma_device = calloc(*num_devices, sizeof(urma_device_t));
    return &urma_device;
}

/**
*  free device list.
* @param[in] [Required] device_list: pointer array of urma_device,return value of urma_get_device_list.
                         Can be called after using urma_device list;
* Return: void;
*/
void urma_free_device_list(urma_device_t** device_list)
{
    free(device_list);
    device_list = NULL;
    return;
}

/**
 *  Get eid list.
 * @param[in] [Required] dev: device pointer
 * @param[out] cnt: Return the number of valid eids;
 * Return: If it succeeds, it will return the eid_info array pointer, and the number of elements
 * is cnt; if it fails, it will return NULL; it will be released by the user calling
 */
urma_eid_info_t* urma_get_eid_list(urma_device_t* dev, uint32_t* cnt)
{
    urma_eid_info_t* urma_eid_info = calloc(*cnt, sizeof(urma_eid_info_t));
    return urma_eid_info;
}

/**
 *  free eid list.
 * @param[in] [Required] eid_list: The eid array pointer to be released
 * Return: void;
 */
void urma_free_eid_list(urma_eid_info_t* eid_list)
{
    free(eid_list);
    eid_list = NULL;
    return;
}

/**
 *  Get device by device name.
 * @param[in] [Required] dev_name: device's name;
 * Return: urma_device; NULL means no device returned;
 */
urma_device_t* urma_get_device_by_name(char* dev_name)
{
    urma_device_t* urma_device = calloc(1, sizeof(urma_device_t));
    return urma_device;
}

/**
 *  Get device by device eid.
 * @param[in] [Required] eid: device's eid;
 * @param[in] [Required] type: device's transport type;
 * Return: urma_device; NULL means no device returned;
 */
urma_device_t* urma_get_device_by_eid(urma_eid_t eid, urma_transport_type_t type)
{
    urma_device_t* urma_device = calloc(1, sizeof(urma_device_t));
    return urma_device;
}

/**
 * Query the attributes and capabilities of urma devices.
 * @param[in] [Required] dev: urma_device;
 * @param[out] dev_attr: Return device attributes, user needs to allocate and free the memory;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_query_device(urma_device_t* dev, urma_device_attr_t* dev_attr) { return 0; }

urma_status_t urma_get_smac(const urma_context_t* ctx, uint8_t* mac) { return 0; }

urma_status_t urma_get_dmac(const urma_context_t* ctx, const urma_net_addr_t* netaddr, uint8_t* mac) { return 0; }

/**
 * Create an urma context on the urma device.
 * @param[in] [Required] dev: urma device, by get_device apis.
 * @param[in] [Required] eid_index: device's eid index.
 * Return urma context pointer on success, NULL on error.
 */
urma_context_t* urma_create_context(urma_device_t* dev, uint32_t eid_index)
{
    urma_context_t* urma_context = calloc(1, sizeof(urma_context_t));
    return urma_context;
}

/**
 * Delete the created urma context.
 * @param[in] [Required] ctx: handle of the created context.
 * Return: 0 on success, other value on error
 */
urma_status_t urma_delete_context(urma_context_t* ctx)
{
    free(ctx);
    ctx = NULL;
    return 0;
}

/**
 * Create a jetty for completion (jfc).
 * @param[in] [Required] ctx: the urma context created before;
 * @param[in] [Required] jfc_cfg: configuration including: depth, flag, jfce, user context;
 * Return: the handle of created jfc, not NULL on success; NULL on error
 */
urma_jfc_t* urma_create_jfc(urma_context_t* ctx, urma_jfc_cfg_t* jfc_cfg)
{
    urma_jfc_t* urma_jfc = calloc(1, sizeof(urma_jfc_t));
    return urma_jfc;
}

/**
 * Modify JFC attributes.
 * @param[in] [Required] jfc: specify JFC;
 * @param[in] [Required] attr: attributes to be modified;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_modify_jfc(urma_jfc_t* jfc, urma_jfc_attr_t* attr) { return 0; }

/**
 * Delete the created jfc.
 * @param[in] [Required] jfc: handle of the created jfc;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_delete_jfc(urma_jfc_t* jfc)
{
    free(jfc);
    jfc = NULL;
    return 0;
}

/**
 * Create a jetty for send (jfs).
 * @param[in] [Required] ctx: the urma context created before;
 * @param[in] [Required] jfs_cfg: address to pu the jfs config;
 * Return: the handle of created jfs, not NULL on success, NULL on error
 */
urma_jfs_t* urma_create_jfs(urma_context_t* ctx, urma_jfs_cfg_t* jfs_cfg)
{
    urma_jfs_t* urma_jfs = calloc(1, sizeof(urma_jfs_t));
    return urma_jfs;
}

/**
 * Modify a jetty for send (jfs).
 * @param[in] [Required] jfs: the jfs created before;
 * @param[in] [Required] attr: attributes to be modified;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_modify_jfs(urma_jfs_t* jfs, urma_jfs_attr_t* attr) { return 0; }

/**
 * Query a jetty for send (jfs).
 * @param[in] [Required] jfs: the jfs created before;
 * @param[out] [Required] cfg: config of jfs;
 * @param[out] [Required] attr: attributes of jfs;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_query_jfs(urma_jfs_t* jfs, urma_jfs_cfg_t* cfg, urma_jfs_attr_t* attr) { return 0; }

/**
 * Delete the created jfs.
 * @param[in] [Required] jfs: the jfs created before;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_delete_jfs(urma_jfs_t* jfs)
{
    free(jfs);
    jfs = NULL;
    return 0;
}

/**
 * Poll the CRs for all the WRs that posted to JFS, but are not completed.
 * Call the API after modify JFS to error, or polled a suspened done CR.
 * CRs with status of URMA_CR_WR_FLUSH_ERR will be returned on success.
 * @param[in] [Required] jfs: the jfs created before;
 * @param[in] [Required] cr_cnt: Number of CR expected to be received.;
 * @param[out] [Required] cr: Address for storing CR;
 * Return: the number of CR returned, 0 means no CR returned, -1 on error
 */
int urma_flush_jfs(urma_jfs_t* jfs, int cr_cnt, urma_cr_t* cr) { return 0; }

/**
 * Create a jetty for receive (jfr).
 * @param[in] [Required] ctx: the urma context created before;
 * @param[in] [Required] jfr_cfg: address to put the jfr config;
 * Return: the handle of created jfr, not NULL on success, NULL on error
 */
urma_jfr_t* urma_create_jfr(urma_context_t* ctx, urma_jfr_cfg_t* jfr_cfg)
{
    urma_jfr_t* urma_jfr = calloc(1, sizeof(urma_jfr_t));
    return urma_jfr;
}

/**
 * Modify JFR attributes.
 * @param[in] [Required] jfr: specify JFR;
 * @param[in] [Required] attr: attributes to be modified;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_modify_jfr(urma_jfr_t* jfr, urma_jfr_attr_t* attr) { return 0; }

/**
 * Query a jetty for recv(jfr).
 * @param[in] [Required] jfr: the jfr created before;
 * @param[out] [Required] cfg: config of jfr;
 * @param[out] [Required] attr: attributes of jfr;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_query_jfr(urma_jfr_t* jfr, urma_jfr_cfg_t* cfg, urma_jfr_attr_t* attr) { return 0; }

/**
 * Delete the created jfr.
 * @param[in] [Required] jfr: the jfr created before;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_delete_jfr(urma_jfr_t* jfr)
{
    free(jfr);
    jfr = NULL;
    return 0;
}

/**
 * Import a remote jfr to local node.
 * @param[in] [Required] ctx: the urma context created before;
 * @param[in] [Required] rjfr: the information of remote jfr to import into user node, trans_mode required,
 *            trans_mode same to create_jfr trans_mode;
 * @param[in] [Required] token_value: token to put into output jetty/protection table;
 * Return: the address of target jfr, not NULL on success, NULL on error
 */
urma_target_jetty_t* urma_import_jfr(urma_context_t* ctx, urma_rjfr_t* rjfr, urma_token_t* token_value)
{
    urma_target_jetty_t* urma_target_jetty = calloc(1, sizeof(urma_target_jetty_t));
    return urma_target_jetty;
}

/**
 * Unimport the imported remote jfr.
 * @param[in] [Required] target_jfr: the target jfr to unimport;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_unimport_jfr(urma_target_jetty_t* target_jfr)
{
    free(target_jfr);
    target_jfr = NULL;
    return 0;
}

/**
 *  Advise jfr: construct the transport channel for jfs and remote jfr.
 * @param[in] [Required] jfs: jfs to use to construct the transport channel;
 * @param[in] [Required] tjfr: target jfr information including full qualified jfr id;
 * Return: 0 on success, URMA_EEXIST if the jfr has been advised, other value on error
 */
urma_status_t urma_advise_jfr(urma_jfs_t* jfs, urma_target_jetty_t* tjfr) { return 0; }

/**
 *  Async API for urma_advise_jfr
 *  Advise jfr: construct the transport channel for jfs and remote jfr.
 * @param[in] [Required] jfs: jfs to use to construct the transport channel;
 * @param[in] [Required] tjfr: target jfr information including full qulified jfr id;
 * @param[in] [Required] cb_func: user defined callback function.
 * @param[in] [Required] cb_arg: user defined arguments for the callback function.
 * Return: 0 on success, URMA_EEXIST if the jfr has been advised, other value on error.
 * Note: User must define callback function to handle result,
 *  as the async respone will call the cb_func and pass the result to it.
 */
urma_status_t
urma_advise_jfr_async(urma_jfs_t* jfs, urma_target_jetty_t* tjfr, urma_advise_async_cb_func cb_fun, void* cb_arg)
{
    return 0;
}

/**
 *  Unadvise jfr: disconnect the transport channel for jfs and remote jfr. Optional API for optimization
 * @param[in] [Required] jfs: jfs to use to construct the transport channel;
 * @param[in] [Required] tjfr: target jfr information including full qualified jfr id;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_unadvise_jfr(urma_jfs_t* jfs, urma_target_jetty_t* tjfr) { return 0; }

/**
 ******************** Beginning of URMA JETTY APIs ***************************
 */

/**
 * Create jetty, which is a pair of jfs and jfr
 * @param[in] [Required] ctx: the urma context created before;
 * @param[in] [Required] jetty_cfg: pointer of the jetty config;
 * Return: the handle of created jetty, not NULL on success, NULL on error
 */
urma_jetty_t* urma_create_jetty(urma_context_t* ctx, urma_jetty_cfg_t* jetty_cfg)
{
    urma_jetty_t* urma_jetty = calloc(1, sizeof(urma_jetty_t));
    return urma_jetty;
}

/**
 * Modify jetty attributes.
 * @param[in] [Required] jetty: specify jetty;
 * @param[in] [Required] attr: attributes to be modified;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_modify_jetty(urma_jetty_t* jetty, urma_jetty_attr_t* attr) { return 0; }

/**
 * Query jetty attributes.
 * @param[in] [Required] jetty: specify jetty;
 * @param[out] [Required] cfg: cconfig to query;
 * @param[out] [Required] attr: attributes to query;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_query_jetty(urma_jetty_t* jetty, urma_jetty_cfg_t* cfg, urma_jetty_attr_t* attr) { return 0; }

/**
 * Delete the created jetty.
 * @param[in] [Required] jetty: the jetty created before;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_delete_jetty(urma_jetty_t* jetty)
{
    free(jetty);
    jetty = NULL;
    return 0;
}

/**
 * Import a remote jetty.
 * @param[in] [Required] ctx: the urma context created before;
 * @param[in] [Required] rjetty: information of remote jetty to import, including jetty id and trans_mode,
 *            trans_mode same to create_jetty trans_mode;
 * @param[in] [Required] token_value: token to put into output jetty protection table;
 * Return: the address of target jetty, not NULL on success, NULL on error
 */
urma_target_jetty_t* urma_import_jetty(urma_context_t* ctx, urma_rjetty_t* rjetty, urma_token_t* token_value)
{
    urma_target_jetty_t* urma_target_jetty = calloc(1, sizeof(urma_target_jetty_t));
    return urma_target_jetty;
}

/**
 * Unimport the imported remote jetty.
 * @param[in] [Required] tjetty: the target jetty to unimport;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_unimport_jetty(urma_target_jetty_t* tjetty)
{
    free(tjetty);
    tjetty = NULL;
    return 0;
}

/**
 *  Advise jetty: construct the transport channel between local jetty and remote jetty.
 * @param[in] [Required] jetty: local jetty to construct the transport channel;
 * @param[in] [Required] tjetty: target jetty imported before;
 * Return: 0 on success, URMA_EEXIST if the jetty has been advised, other value on error
 * Note: A local jetty can be advised with several remote jetties. A connectionless jetty is free to call the adivse API
 */
/* todo: available after implementing URMA_TM_RM(IB_RC) */
urma_status_t urma_advise_jetty(urma_jetty_t* jetty, urma_target_jetty_t* tjetty) { return 0; }

/**
 *  Async API for urma_advise_jetty
 *  Advise jfr: construct the transport channel between local jetty and remote jetty.
 * @param[in] [Required] jetty: local jetty to construct the transport channel;
 * @param[in] [Required] tjetty: target jetty imported before;
 * @param[in] [Required] cb_func: user defined callback function.
 * @param[in] [Required] cb_arg: user defined arguments for the callback function.
 * Return: 0 on success, URMA_EEXIST if the jetty has been advised, other value on error.
 * Note: User must define callback function to handle result,
 *  as the async respone will call the cb_func and pass the result to it.
 */
/* todo: available after implementing URMA_TM_RM(IB_RC) */
urma_status_t
urma_advise_jetty_async(urma_jetty_t* jfs, urma_target_jetty_t* tjetty, urma_advise_async_cb_func cb_fun, void* cb_arg)
{
    return 0;
}

/**
 *  Unadvise jetty: deconstruct the transport channel between local jetty and remote jetty.
 * @param[in] [Required] jetty: local jetty to deconstruct the transport channel;
 * @param[in] [Required] tjetty: target jetty imported before;
 * Return: 0 on success, other value on error
 */
/* todo: available after implementing URMA_TM_RM(IB_RC) */
urma_status_t urma_unadvise_jetty(urma_jetty_t* jetty, urma_target_jetty_t* tjetty) { return 0; }

/**
 *  Bind jetty: construct the transport channel between local jetty and remote jetty.
 * @param[in] [Required] jetty: local jetty to construct the transport channel;
 * @param[in] [Required] tjetty: target jetty imported before;
 * Return: 0 on success, URMA_EEXIST if the jetty has been binded, other value on error
 * Note: A local jetty can be binded with only one remote jetty. Only supported by jetty under URMA_TM_RC.
 */
urma_status_t urma_bind_jetty(urma_jetty_t* jetty, urma_target_jetty_t* tjetty) { return 0; }

/**
 *  Unbind jetty: deconstruct the transport channel between local jetty and remote jetty.
 * @param[in] [Required] jetty: local jetty to deconstruct the transport channel;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_unbind_jetty(urma_jetty_t* jetty) { return 0; }

/**
 * Poll the CRs for all the WRs that posted to Jetty, but are not completed.
 * Call the API after modify Jetty to error, or polled a suspened done CR.
 * CRs with status of URMA_CR_WR_FLUSH_ERR will be returned on success.
 * @param[in] [Required] jetty: the jetty created before;
 * @param[in] [Required] cr_cnt: Number of CR expected to be received.;
 * @param[out] [Required] cr: Address for storing CR;
 * Return: the number of CR returned, 0 means no CR returned, -1 on error
 */
int urma_flush_jetty(urma_jetty_t* jetty, int cr_cnt, urma_cr_t* cr) { return 0; }

/**
 ******************** Beginning of URMA JETTY GROUP APIs ***************************
 */

/**
 * Create jetty group
 * @param[in] [Required] ctx: the urma context created before;
 * @param[in] [Required] cfg: pointer of the jetty group config;
 * Return: the handle of created jetty group, not NULL on success, NULL on error
 */
urma_jetty_grp_t* urma_create_jetty_grp(urma_context_t* ctx, urma_jetty_grp_cfg_t* cfg)
{
    urma_jetty_grp_t* urma_jetty_grp = calloc(1, sizeof(urma_jetty_grp_t));
    return urma_jetty_grp;
}

/**
 * Destroy jetty group
 * @param[in] [Required] jetty_grp: the Jetty group created before;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_delete_jetty_grp(urma_jetty_grp_t* jetty_grp)
{
    free(jetty_grp);
    jetty_grp = NULL;
    return 0;
}

/**
 * Create a jfce
 * @param[in] [Required] ctx: the urma context created before;
 * Return: the address of created jfce, not NULL on success, NULL on error
 */
urma_jfce_t* urma_create_jfce(urma_context_t* ctx)
{
    urma_jfce_t* urma_jfce = calloc(1, sizeof(urma_jfce_t));
    return urma_jfce;
}

/**
 * Delete a jfce
 * @param[in] [Required] jfce: the jfce to be deleted;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_delete_jfce(urma_jfce_t* jfce)
{
    free(jfce);
    jfce = NULL;
    return 0;
}

/**
 *  Get asyn event.
 * @param[in] [Required] ctx: handle of the created urma context;
 * @param[out] [Required] event: the address to put event
 * Return: 0 on success, other value on error
 */
urma_status_t urma_get_async_event(urma_context_t* ctx, urma_async_event_t* event) { return 0; }

/**
 *  Ack asyn event.
 * @param[in] [Required] event: the address to ack event;
 * Return: void
 */
void urma_ack_async_event(urma_async_event_t* event) { return; }

/**
 *  Request to assign a token id. token id is used to register the segment with the protection table.
 * @param[in] [Required] ctx: specifies the urma context.
 * Return: pointer to key id on success, NULL on error.
 */
urma_token_id_t* urma_alloc_token_id(urma_context_t* ctx)
{
    urma_token_id_t* urma_token_id = calloc(1, sizeof(urma_token_id_t));
    return urma_token_id;
}

/**
 * Request to release token id.
 * @param[in] [Required] token_id: Specifies the token id to be released.
 * Return: 0 on success, other value on error
 */
urma_status_t urma_free_token_id(urma_token_id_t* token_id)
{
    free(token_id);
    token_id = NULL;
    return 0;
}

/**
 * Register a memory segment on specified va address for local or remote access.
 * @param[in] [Required] ctx: the created urma context pointer;
 * @param[in] [Required] seg_cfg: Specify cfg of seg to be registered, including address, len, token, and so on;
 * Return: pointer to target segment on success, NULL on error
 * Note: in current IB provider, all segments to be registerred must use a common jfc,
 * And the immedidate data wrote from clients is polled from this common jfc.
 */
urma_target_seg_t* urma_register_seg(urma_context_t* ctx, urma_seg_cfg_t* seg_cfg)
{
    urma_target_seg_t* urma_target_seg = calloc(1, sizeof(urma_target_seg_t));
    urma_target_seg->token_id = seg_cfg->token_id;
    return urma_target_seg;
}

/**
 * Unregister a local memory segment on specified va address.
 * @param[in] [Required] target_seg: target segment to be unregistered;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_unregister_seg(urma_target_seg_t* target_seg)
{
    free(target_seg);
    target_seg = NULL;
    return 0;
}

/**
 * Import a memory segment on specified ubva address.
 * @param[in] [Required] ctx: the created urma context pointer;
 * @param[in] [Required] seg: handle of memory segment to import;
 * @param[in] [Required] token_value: token to put into output protection table;
 * @param[in] [Optional] addr: the virtual address to which the segment will be mapped;
 * @param[in] [Required] flag: flag to indicate the import attribute of memory segment;
 * Return: pointer to target segment on success, NULL on error
 */
urma_target_seg_t* urma_import_seg(
    urma_context_t* ctx, urma_seg_t* seg, urma_token_t* token_value, uint64_t addr, urma_import_seg_flag_t flag)
{
    urma_target_seg_t* urma_target_seg = calloc(1, sizeof(urma_target_seg_t));
    return urma_target_seg;
}

/**
 *  Unimport a memory segment on specified ubva address.
 * @param[in] [Required] tseg: the address of the target segment to unimport;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_unimport_seg(urma_target_seg_t* tseg)
{
    free(tseg);
    tseg = NULL;
    return 0;
}

/**
 * post a request to read, write, atomic or send data.
 * @param[in] jetty: the jetty created before, which is used to put command;
 * @param[in] wr: the posting request all information, including src addr, dst addr, len, jfc, flag, ordering etc.
 * @param[in] bad_wr: the first of failure request.
 * Return: 0 on success, other value on error
 */
urma_status_t urma_post_jetty_send_wr(urma_jetty_t* jetty, urma_jfs_wr_t* wr, urma_jfs_wr_t** bad_wr) { return 0; }

/**
 * post a request to recv data.
 * @param[in] jetty: the jetty created before, which is used to put command;
 * @param[in] wr: the posting request all information, including sge, flag.
 * @param[in] bad_wr: the first of failure request.
 * Return: 0 on success, other value on error
 */
urma_status_t urma_post_jetty_recv_wr(urma_jetty_t* jetty, urma_jfr_wr_t* wr, urma_jfr_wr_t** bad_wr) { return 0; }

/**
 *  Poll jfc to get completion record.
 * @param[in] jfc: jetty completion queue to poll
 * @param[in] cr_cnt: the expected number of completion record to get
 * @param[out] cr: the completion record array to fill at least cr_cnt completion records
 * Return: the number of completion record returned, 0 means no completion record returned, less than 0 on error
 * Note that: at most 16 completion records can be polled for RDMA device
 */
int urma_poll_jfc(urma_jfc_t* jfc, int cr_cnt, urma_cr_t* cr) { return 0; }

/**
 *  Arm jfc with interrupt mode.
 * @param[in] jfc: jetty completion queue to arm to interrupt mode
 * @param[in] solicited_only: indicate it will trigger event only for packets with solicited flag.
 * Return: 0 on success, other value on error
 */
urma_status_t urma_rearm_jfc(urma_jfc_t* jfc, bool solicited_only) { return 0; }

/**
 *  Wait jfce for event of any completion message is generated.
 * @param[in] jfce: jetty event channel to wait on
 * @param[in] jfc_cnt: expected jfc count to return
 * @param[in] time_out: max time to wait (milliseconds),
 *            timeout = 0: return immediately even if no events are ready,
 *            timeout = -1: an infinite timeout
 * @param[out] jfc: address to put the jfc handle
 * Return: the number of jfc returned, 0 means no jfc returned, -1 on error
 * Note: Repeatedly calling this API without calling [urma_poll_jfc] may lead to
 *       incorrect number of jfc in IP provider. This error is controllable.
 */
int urma_wait_jfc(urma_jfce_t* jfce, uint32_t jfc_cnt, int time_out, urma_jfc_t* jfc[]) { return 0; }

/**
 *  Confirm that a JFC generated event has been processed.
 * @param[in] jfc: jfc pointer array to be acknowledged
 * @param[in] nevents: event count array to be acknowledged
 * @param[in] jfc_cnt: number of elements in the array
 * Return: void
 */
void urma_ack_jfc(urma_jfc_t* jfc[], uint32_t nevents[], uint32_t jfc_cnt) { return; }

/**
 * User defined control of the context.
 * @param[in] ctx: the created urma context pointer;
 * @param[in] in: user ioctl cmd;
 * @param[out] out: result of execution;
 * Return: 0 on success, other value on error
 * Note: This API only supports UB hardware currently.
 */
urma_status_t urma_user_ctl(urma_context_t* ctx, urma_user_ctl_in_t* in, urma_user_ctl_out_t* out) { return 0; }

/**
 * get available tp list from control plane.
 * @param[in] [Required] ctx: the created urma context pointer;
 * @param[in] [Required] tp_cfg: tp configuration to get;
 * @param[in && out] [Required] tp_cnt: tp_cnt is the length of tp_list buffer as in parameter;
 *                                      tp_cnt is the number of tp as out parameter;
 * @param[out] [Required] tp_list: tp list to get, the buffer is allocated by user;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_get_tp_list(urma_context_t* ctx, urma_get_tp_cfg_t* cfg, uint32_t* tp_cnt, urma_tp_info_t* tp_list)
{
    *tp_cnt = 1;
    return 0;
}

/**
 * Import a remote jetty by control plane.
 * @param[in] [Required] ctx: the urma context created before;
 * @param[in] [Required] rjetty: information of remote jetty to import, including jetty id and trans_mode,
 *            trans_mode same to create_jetty trans_mode;
 * @param[in] [Required] token_value: token to put into output jetty protection table;
 * @param[in] [Required] cfg: tp active configuration to exchange with target;
 * Return: the address of target jetty, not NULL on success, NULL on error
 */
urma_target_jetty_t* urma_import_jetty_ex(
    urma_context_t* ctx, urma_rjetty_t* rjetty, urma_token_t* token_value, urma_import_jetty_ex_cfg_t* cfg)
{
    urma_target_jetty_t* urma_target_jetty = calloc(1, sizeof(urma_target_jetty_t));
    return urma_target_jetty;
}

/**
 * Delete the created jetty in a batch.
 * @param[in] [Required] jetty_arr: the array of the jetty pointer;
 * @param[in] [Required] jetty_num: array length;
 * @param[out] [Required] bad_jetty: the address of the first failed jetty pointer;
 * Return: 0 on success, EINVAL on invalid parameter, other value on other batch
 * delete errors.
 * If delete error happens(except invalid parameter), stop at the first failed
 * jetty and return, these jetty before the failed jetty will be deleted normally.
 */
urma_status_t urma_delete_jetty_batch(urma_jetty_t** jetty_arr, int jetty_num, urma_jetty_t** bad_jetty)
{
    unsigned int i = 0;
    for (i = 0; i < jetty_num; ++i) {
        if (jetty_arr[i] != NULL) {
            free(jetty_arr[i]);
            jetty_arr[i] = NULL;
        }
    }
    *bad_jetty = NULL;
    return 0;
}

/**
 * Delete the created jfr in a batch.
 * @param[in] [Required] jfr_arr: the array of the jfr pointer;
 * @param[in] [Required] jfr_num: array length;
 * @param[out] [Required] bad_jfr: the address of the first failed jfr pointer;
 * Return: 0 on success, EINVAL on invalid parameter, other value on other batch
 * delete errors.
 * If delete error happens(except invalid parameter), stop at the first failed
 * jfr and return, these jfr before the failed jfr will be deleted normally.
 */
urma_status_t urma_delete_jfr_batch(urma_jfr_t** jfr_arr, int jfr_num, urma_jfr_t** bad_jfr)
{
    unsigned int i = 0;
    for (i = 0; i < jfr_num; ++i) {
        if (jfr_arr[i] != NULL) {
            free(jfr_arr[i]);
            jfr_arr[i] = NULL;
        }
    }

    *bad_jfr = NULL;
    return 0;
}

/**
 * get tp attribution values in control plane.
 * @param[in] [Required] ctx: the created urma context pointer;
 * @param[in] [Required] tp_handle: tp_handle got by urma_get_tp_list;
 * @param[out] [Required] tp_attr_cnt: number of tp attributions;
 * @param[out] [Required] tp_attr_bitmap: tp attributions bitmap, current bitmap is as follow:
 *       0-retry_times_init: 3 bit       1-at: 5 bit                2-SIP: 128 bit
 *       3-DIP: 128 bit                  4-SMA: 48 bit              5-DMA: 48 bit
 *       6-vlan_id: 12 bit               7-vlan_en: 1 bit           8-dscp: 6 bit
 *       9-at_times: 5 bit               10-sl: 4 bit               11-ttl: 8 bit
 * @param[out] [Required] tp_attr: tp attribution values to get;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_get_tp_attr(
    const urma_context_t* ctx, const uint64_t tp_handle, uint8_t* tp_attr_cnt, uint32_t* tp_attr_bitmap,
    urma_tp_attr_value_t* tp_attr)
{
    return 0;
}

/**
 * set tp attribution values in control plane.
 * @param[in] [Required] ctx: the created urma context pointer;
 * @param[in] [Required] tp_handle: tp_handle got by urma_get_tp_list;
 * @param[in] [Required] tp_attr_cnt: number of tp attributions;
 * @param[in] [Required] tp_attr_bitmap: tp attributions bitmap, current bitmap is as follow:
 *       0-retry_times_init: 3 bit       1-at: 5 bit                2-SIP: 128 bit
 *       3-DIP: 128 bit                  4-SMA: 48 bit              5-DMA: 48 bit
 *       6-vlan_id: 12 bit               7-vlan_en: 1 bit           8-dscp: 6 bit
 *       9-at_times: 5 bit               10-sl: 4 bit               11-ttl: 8 bit
 * @param[in] [Required] tp_attr: tp attribution values to set;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_set_tp_attr(
    const urma_context_t* ctx, const uint64_t tp_handle, const uint8_t tp_attr_cnt, const uint32_t tp_attr_bitmap,
    const urma_tp_attr_value_t* tp_attr)
{
    return 0;
}

/**
 * Alloc a jetty.
 * @param[in] [Required] urma_ctx: the urma context created before;
 * @param[in] [Required] jetty_cfg: configuration including: depth, flag, jfce, user context;
 * @param[out] [Required] jetty: handle of the allocated jetty;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_alloc_jetty(urma_context_t* urma_ctx, urma_jetty_cfg_t* cfg, urma_jetty_t** jetty)
{
    *jetty = calloc(1, sizeof(urma_jetty_t));
    return 0;
}

/**
 * Set the opt of jetty.
 * @param[in] [Required] jetty: handle of the allocated jetty;
 * @param[in] [Required] opt: the opt to change cfg of jetty;
 * @param[in] [Required] len: the len of the opt value(byte);
 * @param[in] [Required] buf: the buffer to store the value;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_set_jetty_opt(urma_jetty_t* jetty, uint64_t opt, void* buf, uint32_t len) { return 0; }

/**
 * Active the allocated jetty.
 * @param[in] [Required] jetty: handle of the allocated jetty;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_active_jetty(urma_jetty_t* jetty) { return 0; }

/**
 * Get the opt of jetty.
 * @param[in] [Required] jetty: handle of the allocated jetty;
 * @param[in] [Required] opt: the opt to change cfg of jetty;
 * @param[in] [Required] len: the len of the opt value(byte);
 * @param[out] [Required] buf: the buffer to store the value;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_get_jetty_opt(urma_jetty_t* jetty, uint64_t opt, void* buf, uint32_t len) { return 0; }

/**
 * Deactive the actived jetty.
 * @param[in] [Required] jetty: handle of the allocated jetty;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_deactive_jetty(urma_jetty_t* jetty) { return 0; }

/**
 * Free the allocated jetty.
 * @param[in] [Required] jetty: handle of the allocated jetty;
 * After free, the jfc pointer is no longer allowed to be accessed.
 * Return: 0 on success, other value on error
 */
urma_status_t urma_free_jetty(urma_jetty_t* jetty)
{
    free(jetty);
    jetty = NULL;
    return 0;
}

/**
 * Alloc a jfc.
 * @param[in] [Required] urma_ctx: the urma context created before;
 * @param[in] [Required] cfg: configuration including: depth, flag, jfce, user context;
 * @param[out] [Required] jfc: handle of the allocated jfc;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_alloc_jfc(urma_context_t* urma_ctx, urma_jfc_cfg_t* cfg, urma_jfc_t** jfc)
{
    *jfc = calloc(1, sizeof(urma_jfc_t));
    return 0;
}

/**
 * Set the opt of jfc.
 * @param[in] [Required] jfc: the jfc allocated before;
 * @param[in] [Required] opt: the opt to change cfg of jfc;
 * @param[in] [Required] len: the len of the opt value(byte);
 * @param[in] [Required] buf: the buffer containing the value to set;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_set_jfc_opt(urma_jfc_t* jfc, uint64_t opt, void* buf, uint32_t len) { return 0; }

/**
 * Active the allocated jfc.
 * @param[in] [Required] jfc: the jfc allocated before;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_active_jfc(urma_jfc_t* jfc) { return 0; }

/**
 * Get the opt of jfc.
 * @param[in] [Required] jfc: the jfc allocated before;
 * @param[in] [Required] opt: the opt to change cfg of jfc;
 * @param[in] [Required] len: the len of the opt value(byte);
 * @param[out] [Required] buf: the buffer to store the value;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_get_jfc_opt(urma_jfc_t* jfc, uint64_t opt, void* buf, uint32_t len) { return 0; }

/**
 * Deactive the created jfc.
 * @param[in] [Required] jfc: the jfc actived before;
 * Return: 0 on success, other value on error
 */
urma_status_t urma_deactive_jfc(urma_jfc_t* jfc) { return 0; }

/**
 * Free the created jfc.
 * @param[in] [Required] jfc: the jfc allocated before;
 * After free, the jfc pointer is no longer allowed to be accessed.
 * Return: 0 on success, other value on error
 */
urma_status_t urma_free_jfc(urma_jfc_t* jfc)
{
    free(jfc);
    jfc = NULL;
    return 0;
}

/**
 * Delete the created jfc in a batch.
 * @param[in] [Required] jfc_arr: the array of the jfc pointer;
 * @param[in] [Required] jfc_num: array length;
 * @param[out] [Required] bad_jfc: the address of the first failed jfc pointer;
 * Return: 0 on success, EINVAL on invalid parameter, other value on other batch
 * delete errors.
 * If delete error happens(except invalid parameter), stop at the first failed
 * jfc and return, these jfc before the failed jfc will be deleted normally.
 */
urma_status_t urma_delete_jfc_batch(urma_jfc_t** jfc_arr, int jfc_num, urma_jfc_t** bad_jfc)
{
    int i;
    for (i = 0; i < jfc_num; i++) {
        (void)urma_free_jfc(jfc_arr[i]);
    }
    return 0;
}
