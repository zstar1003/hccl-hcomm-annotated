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

#define SSL_MODE_AUTO_RETRY 0x00000004U

#define SSL_CTRL_MODE 33
#define SSL_set_mode(ssl, op) SSL_ctrl((ssl), SSL_CTRL_MODE, (op), NULL)
#define SSL_OP_NO_RENEGOTIATION 0x40000000U

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

unsigned long long SSL_CTX_set_options(SSL_CTX* ctx, unsigned long long op);
#endif
