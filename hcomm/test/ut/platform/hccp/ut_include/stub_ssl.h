/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#ifndef __SUB_TLS_H
#define __SUB_TLS_H

#define SSL_ERROR_SYSCALL 5
#define SSL_ERROR_WANT_WRITE 3
#define SSL_ERROR_WANT_READ 2
#define X509_FILETYPE_ASN1 2
#define SSL_FILETYPE_ASN1 X509_FILETYPE_ASN1

#define X509_FILETYPE_PEM 1
#define SSL_FILETYPE_ASN1 X509_FILETYPE_ASN1

#define SSL_FILETYPE_PEM X509_FILETYPE_PEM

#define X509_V_OK 0

#define X509_V_FLAG_CRL_CHECK 0x4
#define SSL_VERIFY_PEER 0x01
#define TLS1_2_VERSION 0x0303
#define X509_V_ERR_CERT_HAS_EXPIRED 10

#define SSL_MODE_AUTO_RETRY 0x00000004U

#define SSL_CTRL_MODE 33
#define SSL_set_mode(ssl, op) SSL_ctrl((ssl), SSL_CTRL_MODE, (op), NULL)

#define RSA_PKCS1_OAEP_PADDING 4
#define SSL_OP_NO_RENEGOTIATION 0x40000000U
#define EVP_MAX_MD_SIZE 64 /* longest known is SHA512 */
typedef struct struct_ssl_ctx {
    int cs_flag;
} SSL_CTX;

typedef struct struct_x509_crl {
} X509_CRL;

typedef struct struct_x509_store {
} X509_STORE;

typedef struct struct_ssl_method {
    int cs_flag;
} SSL_METHOD;

typedef struct struct_ssl {
    int fd;
} SSL;

typedef struct struct_x509_name {
} X509_NAME;

typedef struct asn1_string_st {
} ASN1_INTEGER;

typedef struct asn1_object_st {
} ASN1_OBJECT;

typedef struct rsa_st {
} RSA;

typedef struct dsa_st {
} DSA;

typedef struct dh_st {
} DH;

typedef struct ec_key_st {
} EC_KEY;

typedef struct evp_pkey_st {
    int type;
    union {
        void* ptr;
        struct rsa_st* rsa;   /* RSA */
        struct dsa_st* dsa;   /* DSA */
        struct dh_st* dh;     /* DH */
        struct ec_key_st* ec; /* ECC */
    } pkey;
} EVP_PKEY;

struct algo {
    int a;
};

struct signature_algo {
    struct algo* algorithm;
};

typedef struct struct_x509_cinf {
    int version;
} X509_CINF;

typedef struct struct_x509 {
    X509_CINF cert_info;
    struct signature_algo sig_alg;
    EVP_PKEY key;
} X509;

typedef struct struct_bio {
} BIO;

typedef struct struct_x509_store_ctx {
} X509_STORE_CTX;

typedef struct openssl_stack {
    int num;
    const void* data;
    int sorted;
    int num_alloc;
} OPENSSL_STACK;

typedef struct struct_pem_password_cb {
} pem_password_cb;

typedef struct stack_st_X509 {
} ST_X509;

typedef struct struct_evp_md {
} EVP_MD;

long X509_get_version(const X509* x);

long ASN1_INTEGER_get(const ASN1_INTEGER* a);

void* X509_get_ext_d2i(const X509* x, int nid, int* crit, int* idx);

void* sk_value(const OPENSSL_STACK* st, int i);

int sk_num(const OPENSSL_STACK* st);

int RSA_bits(const RSA* r);

int DSA_bits(const DSA* dsa);

int DH_bits(const DH* dh);

int ECDSA_size(const EC_KEY* r);

int EVP_PKEY_get_bits(const EVP_PKEY* pkey);

EVP_PKEY* X509_get_pubkey(X509* x);

void OpenSSL_add_all_algorithms();

int OPENSSL_add_all_algorithms_noconf(long opts, void* a);

void SSL_library_init();

void SSL_load_error_strings();

int SSL_CTX_set_min_proto_version(SSL_CTX* ctx, int version);

void SSL_CTX_set_verify(SSL_CTX* ctx, int mode, void* a);

int SSL_CTX_set_cipher_list(SSL_CTX* ctx, const char* str);

int SSL_CTX_use_certificate_chain_file(SSL_CTX* ctx, const char* file);

int SSL_CTX_load_verify_locations(SSL_CTX* ctx, const char* CAfile, const char* CApath);

EVP_PKEY* PEM_read_PrivateKey(FILE* fp, void* a, void* b, void* u);

int SSL_CTX_check_private_key(const SSL_CTX* ctx);

int SSL_CTX_use_PrivateKey(SSL_CTX* ctx, EVP_PKEY* pkey);

void EVP_PKEY_free(EVP_PKEY* x);

X509_CRL* PEM_read_X509_CRL(const char* file, void* a, void* b, void* c);

X509* PEM_read_X509(FILE* fp, X509** x, pem_password_cb* cb, void* u);

int X509_digest(const X509* data, const EVP_MD* type, unsigned char* md, unsigned int* len);

X509_STORE* SSL_CTX_get_cert_store(const SSL_CTX* ctx);

int X509_STORE_set_flags(X509_STORE* ctx, unsigned long flags);

long SSL_ctrl(SSL* s, int cmd, long larg, void* parg);

int X509_STORE_add_crl(X509_STORE* ctx, X509_CRL* x);

void X509_STORE_free(X509_STORE* vfy);

const SSL_METHOD* TLS_server_method(void);

const SSL_METHOD* TLS_client_method(void);

SSL_CTX* SSL_CTX_new(const SSL_METHOD* meth);

void SSL_CTX_free(SSL_CTX* ctx);

int SSL_shutdown(SSL* s);

void SSL_free(SSL* ssl);

SSL* SSL_new(SSL_CTX* ctx);

int SSL_get_error(const SSL* s, int ret_code);

int SSL_set_fd(SSL* s, int fd);

long SSL_ctrl(SSL* ssl, int cmd, long larg, void* parg);

void SSL_set_connect_state(SSL* s);

void SSL_set_accept_state(SSL* s);

int SSL_do_handshake(SSL* s);

long SSL_get_verify_result(const SSL* ssl);

X509* SSL_get_peer_certificate(const SSL* s);

X509_NAME* X509_get_issuer_name(const X509* a);

char* X509_NAME_oneline(const X509_NAME* a, char* buf, int len);

int SSL_write(SSL* ssl, const void* buf, int num);

int SSL_read(SSL* ssl, void* buf, int num);

#define STACK_OF(type) struct stack_st_##type

BIO* BIO_new_mem_buf(const void* buf, int len);

X509* d2i_X509_bio(BIO* bp, X509** x509);

X509* PEM_read_bio_X509(BIO* bp, X509** x, pem_password_cb* cb, void* u);

X509_STORE* X509_STORE_new(void);

X509_STORE_CTX* X509_STORE_CTX_new(void);

int X509_STORE_CTX_init(X509_STORE_CTX* ctx, X509_STORE* store, X509* x509, STACK_OF(X509) * chain);

int X509_verify_cert(X509_STORE_CTX* ctx);

int X509_STORE_CTX_get_error(X509_STORE_CTX* ctx);

const char* X509_verify_cert_error_string(long n);

void X509_STORE_CTX_cleanup(X509_STORE_CTX* ctx);

void X509_STORE_CTX_free(X509_STORE_CTX* ctx);

void X509_STORE_free(X509_STORE* vfy);

void X509_free(X509* buf);

OPENSSL_STACK* sk_new_null(void);

void sk_free(OPENSSL_STACK* buf);

int sk_push(OPENSSL_STACK* st, const void* data);

void X509_STORE_CTX_trusted_stack(X509_STORE_CTX* ctx, STACK_OF(X509) * sk);

unsigned long ERR_peek_last_error(void);

void ERR_clear_error(void);

void ERR_error_string_n(unsigned long e, char* buf, size_t len);

int X509_CRL_free(X509_CRL* crl);

int X509_STORE_add_cert(X509_STORE* ctx, X509* x);

unsigned long long SSL_CTX_set_options(SSL_CTX* ctx, unsigned long long op);

typedef struct ossl_lib_ctx_st {
} OSSL_LIB_CTX;

#define TLS_RSA_KY_BITS_MIN_LEN 2048
#define TLS_DSA_KY_BITS_MIN_LEN 2048
#define TLS_DH_KY_BITS_MIN_LEN 2048
#define TLS_EC_KY_BITS_MIN_LEN 256
#define TLS_KY_NONCE_LEN 48
#define TLS_KY_RSA_LEN 512

#define MAX_TLS_CFG_COUNT 18
#define MAX_TLS_CA_ALIAS_LEN 63
#define TLS_RES_LEN 160

#define MAX_CERT_COUNT 15
#define MAX_SHOW_INFO_COUNT 16
#define MAX_CA_CERT_INDEX 14
#define CA_CERT_BEGIN_INDEX 2

#define CERT_MAX_SIZE 3072
#define OLD_CERT_MAX_SIZE 2048
#define KY_MAX_SIZE 5120
#define PUB_KY_MAX_SIZE 3072
#define CRL_MAX_SIZE (1024 * 20)
#define CERT_NAME_MAX_LEN 64

#define TIME_LEN 26
#define TLS_TYPE_LEN 10

#define TLS_MAGIC_WORDS_LEN 8

#define TLS_SALT_LEN 48
#define IV_LEN 16

#define BLOCK_KY_LEN 32

#define PWD_MIN_LEN 8
#define PWD_MAX_LEN 15
#define PWD_MAX_ENC_LEN 15
#define FGETS_MAX_LEN 32
#define PWD_ENC_LEN 256
#define WORK_KEY_LEN 516
#define TAG_LEN 16

#define ENVELOPE_SYMM_KY_LEN 32
#define ENVELOPE_SYMM_ENC_KY_LEN 512

#define PWD_TPYE_CNT 4
#define PWD_NUM_INDEX 0
#define PWD_LOW_LET_INDEX 1
#define PWD_UP_LET_INDEX 2
#define PWD_SYMBOL_INDEX 3

#define PWD_COMPLEXITY_THR 2

#define TLS_ITER_MAX_NUM 10000

#define START_TIME 0
#define END_TIME 1
#define YEAR_MON_DAY_INDEX 2
#define YEAR_MON_DAY_LEN 8
#define TLS_DEFAULT_ALARM_TIME 60   // day
#define TLS_DAY_TO_S (60 * 60 * 24) // s

#define TLS_PRI_PLAINTEXT 0
#define TLS_PRI_CIPHERTEXT 1
#define TLS_PUB_PLAINTEXT 2

#define TLS_DEC_MODE 0
#define TLS_ENC_MODE 1

#define TLS_VERSION 2

#define TLS_CERT_ILLEGAL (-100)

enum tls_err_num {
    TLS_CERT_LOAD_ERR = -1,
    TLS_CERT_VERIFY_ERR = -2,
    TLS_CERT_KYMATCH_ERR = -3,
    TLS_CERT_LACK_PUB_ERR = -4,
    TLS_CERT_DISCONSEQ_ERR = -5,
    TLS_CERT_CTX_INIT_ERR = -6,
};

#define TLS_CA_CERT (-1)

enum tls_cert_tpye {
    TLS_PUB_CERT = 0,
    TLS_CA1_CERT,
    TLS_CA2_CERT,
    TLS_CA3_CERT,
    TLS_CA4_CERT,
    TLS_CA5_CERT,
    TLS_CA6_CERT,
    TLS_CA7_CERT,
    TLS_CA8_CERT,
    TLS_CA9_CERT,
    TLS_CA10_CERT,
    TLS_CA11_CERT,
    TLS_CA12_CERT,
    TLS_CA13_CERT,
    TLS_CA14_CERT,

    TLS_PRI_KY,

    TLS_CRL,
    TLS_HOST,
};

#define TLS_SAVE_TO_FlASH 0
#define TLS_SAVE_TO_FILE 1
#define TLS_ENABLE_INVALID 0xFFFFFFFF

#define TLS_MNG_INFO_DEFAULT_VERSION 0x0
// flash mode won't use kmc
#define TLS_MNG_INFO_VERSION_1 0x00010000

struct tls_cert_mng_info {
    char magic_words[TLS_MAGIC_WORDS_LEN]; /* 1234567 */
    unsigned int cert_count;               /* num of certs */
    int state;                             /* 0:not ok 1:ok */
    unsigned int ca_wcout;                 /* counts of ca writing flash */
    unsigned int cert_ky_wcout;            /* counts of eqpt and key writing flash */
    unsigned int crl_wcout;                /* counts of crl writing flash */
    unsigned int crl_len;                  /* len of crl */
    unsigned int ky_len;                   /* len of key */
    unsigned int ky_enc_len;               /* len of enc key */
    unsigned char salt[TLS_SALT_LEN];      /* salt */
    unsigned int salt_size;                /* len of salt */
    unsigned int cert_len[MAX_CERT_COUNT];
    unsigned int total_cert_len; /* not include head only len of certs */
    unsigned int tls_enable;
    unsigned int tls_alarm;
    unsigned int pwd_len;     /* len of pwd */
    unsigned int pwd_enc_len; /* len of enc pwd */
    union {
        unsigned char enc_pwd[PWD_ENC_LEN];
        unsigned char pwd[PWD_ENC_LEN];
    };
    unsigned int work_key_len; /* len of work_key */
    unsigned char work_key[WORK_KEY_LEN];
    unsigned char iv[IV_LEN]; /* initial vector */
    unsigned int iv_size;     /* len of initial vector */
    unsigned char tag[TAG_LEN];
    unsigned int tag_len;
    unsigned int save_mode;
    unsigned char envelope_iv[IV_LEN]; /* initial vector for envelope */
    unsigned char envelope_tag[TAG_LEN];
    unsigned int version;
    char res[TLS_RES_LEN];
};

#define TLS_CA_SSL_NEW_CERT_LEN 3072
#define TLS_CA_SSL_MAX_NEW_CERT_NUM 8
#define TLS_CA_SSL_MAX_FLASH_NUM 2 // falsh块cb1与cb2为一组，cb3与cb4为一组
#define TLS_CA_SSL_NEW_CERT_ALIAS_LEN 64
#define TLS_CA_SSL_RSV_LEN 7160

struct tls_ca_new_cert_info {
    char ncert_info[TLS_CA_SSL_NEW_CERT_LEN];
};

struct tls_ca_alias_names {
    char name[TLS_CA_SSL_NEW_CERT_ALIAS_LEN];       // 证书别名
    char thumbprint[TLS_CA_SSL_NEW_CERT_ALIAS_LEN]; // 证书指纹
};

// 910B旧证书格式
struct tls_cert_info {
    char certInfo[OLD_CERT_MAX_SIZE];
};

// 实验局天工款型旧证书格式
struct tls_atlas_9000_cert_info {
    char certInfo[TLS_CA_SSL_NEW_CERT_LEN];
};

// 新证书格式。每个cb里面都需要存储：8个3072字节的证书、8个证书对应别名、证书数量、预留字段大小
struct tls_ca_new_certs {
    struct tls_ca_new_cert_info certs[TLS_CA_SSL_MAX_NEW_CERT_NUM];
    struct tls_ca_alias_names alias[TLS_CA_SSL_MAX_NEW_CERT_NUM];
    unsigned int ncert_count;
    char res[TLS_CA_SSL_RSV_LEN];
};

struct tls_ky_info {
    unsigned char ky_info[KY_MAX_SIZE];
};

struct tls_crl_info {
    unsigned char crl_info[CRL_MAX_SIZE];
};

struct tls_pwd_info {
    unsigned char pwd_info[PWD_MAX_LEN + 1];
};

struct envelope_symm_enc_ky_info {
    unsigned int symm_enc_ky_len;
    unsigned char symm_enc_ky[ENVELOPE_SYMM_ENC_KY_LEN];
};

struct tls_cert_ky_crl_info {
    struct tls_cert_mng_info mng;
    struct tls_ky_info ky;
    struct tls_cert_info certs[MAX_CERT_COUNT];
    struct tls_ca_new_certs ncerts[TLS_CA_SSL_MAX_FLASH_NUM];
    struct tls_crl_info crl;
    struct tls_pwd_info pwd;
    struct envelope_symm_enc_ky_info symm_enc_ky_info;
};

struct tls_alarm_info {
    unsigned int alarm;
    unsigned int save_mode;
};

struct tls_enable_info {
    unsigned int enable;
    unsigned int save_mode;
    int machine_type;
};

struct tls_cert_show_info {
    unsigned int tls_alarm;
    unsigned int tls_enable;
    char issuer[CERT_NAME_MAX_LEN];
    char subject[CERT_NAME_MAX_LEN];
    char start_time[TIME_LEN];
    char end_time[TIME_LEN];
};

struct leaf_cert_info {
    X509* leaf_cert;
    unsigned int leaf_cert_idx;
};

#define HCCP_CERTS_MNG_NAME "hccp_certs_mng_cb"
#define HCCP_CERTS_EQPT_NAME "hccp_certs_eqpt_cb"
#define HCCP_CERTS_EQPT1_NAME "hccp_certs_eqpt_cb1"
#define HCCP_CERTS_EQPT2_NAME "hccp_certs_eqpt_cb2"
#define HCCP_CERTS_EQPT3_NAME "hccp_certs_eqpt_cb3"
#define HCCP_CERTS_EQPT4_NAME "hccp_certs_eqpt_cb4"
#define HCCP_PRI_DATA_NAME "hccp_pri_data_cb"
#define HCCP_CERTS_REVOC_NAME "hccp_certs_revoc_cb"

#define MAGIC_WORD_FOR_TLS "1234567"

#define KMC_SECU_PATH_LEN 64
#define KMC_STORE_PATH_LEN 64

#define TLS_LOCK_FILE_LEN 128
#define TLS_HOST_SAVE_PATH_LEN 128
#define MAX_TLS_LEN 30

struct tls_ky_match_info {
    unsigned char pri_ky_info[KY_MAX_SIZE];
    unsigned char pub_ky_info[PUB_KY_MAX_SIZE];
    unsigned int pri_ky_len;
    unsigned int pub_ky_len;
    unsigned int pub_type;
    unsigned int pri_type;
    uint8_t random[TLS_KY_NONCE_LEN];
    uint8_t sig[TLS_KY_RSA_LEN];
    size_t random_len;
    size_t sig_len;
};

#define ENVELOPE_PUB_CERT 0
#define ENVELOPE_PUB_KY 1

struct digital_envelope_mng_info {
    unsigned int pri_ky_len;     /* len of pri key */
    unsigned int pri_ky_enc_len; /* len of enc pri key */
    unsigned int work_key_len;   /* len of work_key */
    unsigned char work_key[WORK_KEY_LEN];
};

struct digital_envelope_info {
    struct digital_envelope_mng_info mng;
    unsigned char pri_ky_info[KY_MAX_SIZE];
};

struct symmetric_enc_info {
    unsigned char* iv;
    unsigned char* tag;
    unsigned char* out_buf;
    unsigned int* out_len;
};

struct envelope_pub_info {
    unsigned int pub_ky_len;
    unsigned char pub_ky_info[PUB_KY_MAX_SIZE];
};

struct envelope_pri_info {
    unsigned int pri_ky_len;
    unsigned char pri_ky_info[KY_MAX_SIZE];
};

struct envelope_kmc_info {
    unsigned int work_key_len;
    unsigned char work_key[WORK_KEY_LEN];
};

#ifndef CONFIG_LLT
#define TLS_PWD_SECU_PATH "%s/tls_%d.secu"
#define TLS_PWD_STORE_PATH "%s/tls_%d.store"
#define TLS_LOCK_FILE_NAME "%s/tls_file.lock"
#else
#define TLS_PWD_SECU_PATH "/var/log/tls.secu"
#define TLS_PWD_STORE_PATH "/var/log/tls.store"
#define TLS_LOCK_FILE_NAME "/var/log/tls_file.lock"
#endif

#define ENCRYPTED_FLAG "ENCRYPTED"
#define VALID_ENCCRY_ALGO "AES-256"
#endif
