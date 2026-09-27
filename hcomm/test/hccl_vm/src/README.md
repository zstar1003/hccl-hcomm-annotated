# TABLE CMD

[toc]

## 1 核心功能介绍

### 1.1启动模拟环境

#### 1.1.1 查看当前所有表名称

```bash
(hvm)$> hccl-vm table show all
all
Server
Host
Runner
Device
...
(hvm)$>
```

#### 1.1.2 查看指定表内容

```bash
(hvm)$> hccl-vm table show Device
| id | server_id | logic_id | physical_id | overflow_mode | soc_version | status |
| 1 | 1 | 0 | 0 | 0 | Ascend950 | 0 |
| 2 | 1 | 1 | 1 | 0 | Ascend950 | 0 |
| 3 | 1 | 2 | 2 | 0 | Ascend950 | 0 |
| 4 | 1 | 3 | 3 | 0 | Ascend950 | 0 |

(hvm)$>
```

#### 1.1.3 更新指定表指定行指定内容(目前只支持devicesoc_version修改)

```bash
(hvm)$> hccl-vm table show Device
| id | server_id | logic_id | physical_id | overflow_mode | soc_version | status |
| 1 | 1 | 0 | 0 | 0 | Ascend950 | 0 |
| 2 | 1 | 1 | 1 | 0 | Ascend950 | 0 |
| 3 | 1 | 2 | 2 | 0 | Ascend950 | 0 |
| 4 | 1 | 3 | 3 | 0 | Ascend950 | 0 |
(hvm)$> hccl-vm table update device 1 soc_version Ascend951
update device 1 soc_version Ascend951
Updating device [id=1].soc_version = "Ascend951"
(hvm)$> hccl-vm table show Device
| id | server_id | logic_id | physical_id | overflow_mode | soc_version | status |
| 1 | 1 | 0 | 0 | 0 | Ascend951 | 0 |
| 2 | 1 | 1 | 1 | 0 | Ascend950 | 0 |
| 3 | 1 | 2 | 2 | 0 | Ascend950 | 0 |
| 4 | 1 | 3 | 3 | 0 | Ascend950 | 0 |

(hvm)$>
```
