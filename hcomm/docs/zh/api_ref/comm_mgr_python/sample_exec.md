# 样例执行

本节以单机8卡组网、通过rank table文件配置资源信息的方式为例，介绍如何运行[代码示例](code_example.md)中的样例代码。

1. 准备rank table文件。

    rank table的配置方法可参见[集群信息配置](https://gitcode.com/cann/hccl/blob/master/docs/zh/user_guide/cluster_info_config/README.md)，此处以“rank_table.json”命名为例。

2. 构造启动脚本。

    假设命名为hccl_start_8p.sh，示例如下：

    ```bash
    # 配置CANN软件环境变量（以root用户为例）：
    source /usr/local/Ascend/cann/set_env.sh
    # TF Adapter python库，其中${TFPLUGIN_INSTALL_PATH}为TF Adapter软件包安装路径。
    export PYTHONPATH=${TFPLUGIN_INSTALL_PATH}:$PYTHONPATH
    
    export RANK_SIZE=8
    export RANK_TABLE_FILE=/home/test/rank_table.json    # rank table资源配置文件路径，请根据实际情况替换
    export JOB_ID=10087      # 用户自定义，指定任务ID，可以包含大小写字母、数字、中划线或下划线
    
    for((RANK_ID=0;RANK_ID&lt;$((RANK_SIZE));RANK_ID++));
    do
        export RANK_ID=$RANK_ID
        export ASCEND_DEVICE_ID=$RANK_ID
        # 执行脚本，脚本路径与名称请根据实际情况替换
        nohup python3 /home/test/hccl_test.py &
    done
    ```

3. 执行启动脚本。

    ```bash
    bash hccl_start_8p.sh 
    ```

    结果示例如下：

    ```text
    ... ...
    'reduce_sum': array([[ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ...,  0,  0,  0],
           ...,
           [ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ...,  0,  0,  0],
           [ 0,  0,  0, ..., 44, 44, 44]]), 'reduce_max': array([[4097, 4098, 4099, ..., 4222, 4223, 4224],
           [4225, 4226, 4227, ..., 4350, 4351, 4352],
           [4353, 4354, 4355, ..., 4478, 4479, 4480],
           ...,
           [4737, 4738, 4739, ..., 4862, 4863, 4864],
           [4865, 4866, 4867, ..., 4990, 4991, 4992],
           [4993, 4994, 4995, ...,    9,    9,    9]]), 'reduce_min': array([[0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           ...,
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 0, 0, 0],
           [0, 0, 0, ..., 2, 2, 2]]), 'reduce_prod': array([[     0,      0,      0, ...,      0,      0,      0],
           [     0,      0,      0, ...,      0,      0,      0],
           [     0,      0,      0, ...,      0,      0,      0],
           ...,
           [     0,      0,      0, ...,      0,      0,      0],
           [     0,      0,      0, ...,      0,      0,      0],
           [     0,      0,      0, ..., 362880, 362880, 362880]]), 'alltoallv_tensor': array([   1,    2,    3, ..., 8246, 8247, 8248]), 'check_tensors': array([   1,    2,    3, ..., 8246, 8247, 8248])
    train success
    ```
