# 查询拓扑信息

## 背景

为了应对复杂的网络拓扑结构，通信算子需要根据通信域的拓扑结构选择最匹配的算法，为此，HCCL控制面提供了拓扑信息查询功能，供开发者使用。

## 拓扑信息

HCCL控制面接口支持查询的拓扑信息如下表所示。

| 拓扑信息 | 查询接口 |
| --- | --- |
| 获取Device在指定通信域中对应的rank序号。 | [HcclGetRankId](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclGetRankId.md) |
| 查询指定通信域的rank数。 | [HcclGetRankSize](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclGetRankSize.md) |
| 查询包含当前rank的拓扑层级编号列表以及拓扑层级数量。 | [HcclRankGraphGetLayers](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetLayers.md) |
| 给定通信域和拓扑层级编号，返回该层级下本rank所在拓扑实例的所有rank编号列表以及rank数量。 | [HcclRankGraphGetRanksByLayer](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetRanksByLayer.md) |
| 给定通信域和拓扑层级编号，返回该层级下本rank所在拓扑实例的rank数量。 | [HcclRankGraphGetRankSizeByLayer](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetRankSizeByLayer.md) |
| 给定通信域和拓扑层级编号，返回本rank所在拓扑层级中的拓扑类型。 | [HcclRankGraphGetTopoTypeByLayer](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetTopoTypeByLayer.md) |
| 给定通信域和拓扑层级编号，查询该层级下的拓扑实例数量，以及每个实例包含的rank数量。 | [HcclRankGraphGetInstSizeListByLayer](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetInstSizeListByLayer.md) |
| 给定通信域和拓扑层级编号，查询源rank和目的rank之间的通信连接信息。 | [HcclRankGraphGetLinks](../../api_ref/comm_opdev/control_plane_api/topo_info_query/HcclRankGraphGetLinks.md) |

- 查询通信域内当前rank的id。

    ```c
    u32 userRank = INVALID_VALUE_RANKID;
    HcclResult ret = HcclGetRankId(comm, &userRank);
    if (userRank == root && sendBuf == nullptr) {     // root节点的sendBuf不允许为空
        return HCCL_E_PTR;
    }
    ```

- 查询通信域的rank数量。

    ```c
    u32 rankSize = INVALID_VALUE_RANKSIZE;
    HcclResult ret = HcclGetRankSize(comm, &rankSize);
    if (userRank >= rankSize) {     // rank_id超出范围
        return HCCL_E_PARA;
    }
    ```

- 查询本卡与Server内0号卡之间的链路信息。

    ```c
    u32 dstRank = 0;
    u32 srcRank = rank;
    CommLink *linkList = nullptr;
    u32 listSize = 0;
    HcclResult ret = HcclRankGraphGetLinks(comm, 0, srcRank, dstRank, &linkList, &listSize);
    for (u32 i = 0; i < listSize; ++i) {
        CommLink& currentLink = linkList[i]; // 枚举所有链路对象
        if (currentLink.linkAttr.linkProtocol == CommProtocol::COMM_PROTOCOL_HCCS) {
            // 对HCCS链路的判断处理
        }
    }
    ```

- 查询本卡所在通信域包含的超节点数量。

    ```c
    u32 *netlayers = nullptr;
    u32 netLayersNum = 0;
    HcclResult ret = HcclRankGraphGetLayers(comm, &netlayers, &netLayersNum); // 获取通信域中包含的通信网络层次
    u32 superPodNum;
    u32 level1RankListNum = 0;
    u32 *level1SizeList = nullptr;
    if (netLayersNum == 3) { // 超节点场景
        ret = HcclRankGraphGetInstSizeListByLayer(comm, 1, &level1SizeList, &level1RankListNum);
        superPodNum = level1RankListNum;
    }
    ```

## 代码示例

```c
// 以一个单机4卡的Mesh连接为例
struct TopoInfo {
    uint32_t rankId; // 用以唯一标识一个rank
    uint32_t rankSize; // 参与此次集合通信的rank的数量
    std::vector<u32> rankList; // 参与此次集合通信的rank的rankId集合
    CommTopo topoType; // 链路连接类型：COMM_TOPO_1DMESH/COMM_TOPO_CLOS等
    std::vector<HcclChannelDesc> channels; // 本rank和其他卡之间的链路信息
};
HcclResult FillSimpleTopoInfo(HcclComm comm, TopoInfo &topoInfo){
    HcclResult ret = HcclGetRankId(comm, &topoInfo.rankId);
    // 校验ret结果
    ret = HcclGetRankSize(comm, &topoInfo.rankSize);
    // 校验ret结果
    uint32_t *ranks = nullptr;
    uint32_t rankNum = 0;
    // 由于是单机4卡，因此netLayer=0，获取netLayer 0下的ranks即为此次集合通信的rankId的集合
    ret = HcclRankGraphGetRanksByLayer(comm, 0, &ranks, &rankNum);
    // 校验ret结果
    for (size_t index = 0; index < rankNum; index++) {
        topoInfo.rankList.push_back(ranks[index]);
    }
    ret = HcclRankGraphGetTopoTypeByLayer(comm, 0, &topoInfo.topoType);
    // 校验结果
    // 计算需要的channels
    for (auto remoteRankId : topoInfo.rankList) {
       if (remoteRankId == topoInfo.rankId) {continue;}
       CommLink *linkList = nullptr;
       u32 listSize = 0;
       uint32_t netLayer = 0;
       ret = HcclRankGraphGetLinks(comm, netLayer, topoInfo.rankId, remoteRankId, &linkList, &listSize);
       // 校验ret结果
       for (uint32_t idx = 0; idx < listSize; idx++) {
           HcclChannelDesc channelDesc;
           HcclChannelDescInit(&channelDesc, 1);
           channelDesc.remoteRank = remoteRankId;
           CommLink link = linkList[idx];
           channelDesc.localEndpoint.protocol = link.srcEndpointDesc.protocol;
           channelDesc.localEndpoint.commAddr = link.srcEndpointDesc.commAddr;
           channelDesc.localEndpoint.loc = link.srcEndpointDesc.loc;
           channelDesc.remoteEndpoint.protocol = link.dstEndpointDesc.protocol;
           channelDesc.remoteEndpoint.commAddr = link.dstEndpointDesc.commAddr;
           channelDesc.remoteEndpoint.loc = link.dstEndpointDesc.loc;
           channelDesc.channelProtocol = link.linkAttr.linkProtocol;
           channelDesc.notifyNum = 3;
           topoInfo.channels.push_back(channelDesc);
       }
    }
    return ret;    
}
```
