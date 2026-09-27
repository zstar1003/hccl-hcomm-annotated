# EndpointAttr

## 功能说明

定义Endpoint属性。

## 定义原型

```c
typedef  enum {
    ENDPOINT_ATTR_INVALID = -1, /*不合法的属性*/
    ENDPOINT_ATTR_BW_COEFF =  0, /* 带宽属性*/
    ENDPOINT_ATTR_DIE_ID = 1,   /* DIE ID */
    ENDPOINT_ATTR_LOCATION = 2, /* Endpoint的位置*/
} EndpointAttr;
```
