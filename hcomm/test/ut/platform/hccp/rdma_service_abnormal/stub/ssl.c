/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stub_ssl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <stdint.h>
#include <stdio.h>

long ASN1_INTEGER_get(const ASN1_INTEGER* a) { return 2; }

void* X509_get_ext_d2i(const X509* x, int nid, int* crit, int* idx) {}

void* sk_value(const OPENSSL_STACK* st, int i)
{
    X509* cert = calloc(1, sizeof(X509));
    return cert;
}

int sk_num(const OPENSSL_STACK* st) { return 1; }

STACK_OF(X509) * SSL_get0_verified_chain(const SSL* s) {}

int RSA_bits(const RSA* r) { return 4096; }

int DSA_bits(const DSA* dsa) { return 4096; }

int DH_bits(const DH* dh) { return 4096; }

int ECDSA_size(const EC_KEY* r) { return 4096; }

int EVP_PKEY_get_bits(const EVP_PKEY* pkey) { return 4096; }

EVP_PKEY* X509_get_pubkey(X509* x)
{
    (x->key).type = 6;
    return &(x->key);
}

int OBJ_obj2nid(const ASN1_OBJECT* a) { return 668; }

void OpenSSL_add_all_algorithms() { return; }

int OPENSSL_add_all_algorithms_noconf(long opts, void* a) { return 1; }

void SSL_library_init() { return; }

void SSL_load_error_strings() { return; }

void SSL_CTX_set_verify(SSL_CTX* ctx, int mode, void* a) { return; }

int SSL_CTX_set_cipher_list(SSL_CTX* ctx, const char* str) { return 1; }

int SSL_CTX_use_certificate_chain_file(SSL_CTX* ctx, const char* file) { return 1; }

int SSL_CTX_set_min_proto_version(SSL_CTX* ctx, int version) { return 1; }

int SSL_CTX_load_verify_locations(SSL_CTX* ctx, const char* CAfile, const char* CApath) { return 1; }

EVP_PKEY* PEM_read_PrivateKey(FILE* fp, void* a, void* b, void* u) {}

int SSL_CTX_check_private_key(const SSL_CTX* ctx) { return 1; }

int SSL_CTX_use_PrivateKey(SSL_CTX* ctx, EVP_PKEY* pkey) { return 1; }

void EVP_PKEY_free(EVP_PKEY* x) { return; }

X509_CRL* PEM_read_X509_CRL(const char* file, void* a, void* b, void* c) { return NULL; }

X509_STORE* SSL_CTX_get_cert_store(const SSL_CTX* ctx)
{
    X509_STORE* store = malloc(sizeof(X509_STORE));
    return store;
}

int X509_STORE_set_flags(X509_STORE* ctx, unsigned long flags) { return 1; }

void X509_STORE_free(X509_STORE* vfy) { free(vfy); }

const SSL_METHOD* TLS_server_method(void) { return NULL; }

const SSL_METHOD* TLS_client_method(void) { return NULL; }

SSL_CTX* SSL_CTX_new(const SSL_METHOD* meth)
{
    SSL_CTX* ctx = malloc(sizeof(SSL_CTX));
    return ctx;
}

void SSL_CTX_free(SSL_CTX* ctx)
{
    free(ctx);
    return;
}

int SSL_shutdown(SSL* s) { return 1; }

void SSL_free(SSL* ssl)
{
    free(ssl);
    return;
}

SSL* SSL_new(SSL_CTX* ctx)
{
    SSL* ssl = malloc(sizeof(SSL_CTX));
    return ssl;
}

int SSL_get_error(const SSL* s, int ret_code) { return 0; }

int SSL_set_fd(SSL* s, int fd)
{
    s->fd = fd;
    return 1;
}

int X509_STORE_add_crl(X509_STORE* ctx, X509_CRL* x) { return 1; }

long SSL_ctrl(SSL* s, int cmd, long larg, void* parg) { return 0; }

void SSL_set_connect_state(SSL* s) { return; }

void SSL_set_accept_state(SSL* s) { return; }

int SSL_do_handshake(SSL* s) { return 1; }

long SSL_get_verify_result(const SSL* ssl) { return 0; }

X509* SSL_get_peer_certificate(const SSL* s)
{
    X509* x509 = malloc(sizeof(X509));
    return x509;
}

X509_NAME* X509_get_issuer_name(const X509* a)
{
    free(a);
    X509_NAME* x509_name = malloc(sizeof(X509_NAME));
    return x509_name;
}

char* X509_NAME_oneline(const X509_NAME* a, char* buf, int len)
{
    free(a);
    return "CA";
}

int SSL_write(SSL* ssl, const void* buf, int num)
{
    int ret;
    ret = send(ssl->fd, buf, num, 0);
    if (ret <= 0) {
        fprintf(stderr, "SSL write size is %d", ret);
    }
    return ret;
}

int SSL_read(SSL* ssl, void* buf, int num)
{
    int ret;
    ret = recv(ssl->fd, buf, num, 0);
    if (ret <= 0) {
        fprintf(stderr, "SSL read size is %d", ret);
    }
    return ret;
}

BIO* BIO_new_mem_buf(const void* buf, int len)
{
    BIO* ret = malloc(sizeof(BIO));

    return ret;
}

int BIO_free(BIO* a)
{
    free(a);
    return 0;
}

#define STACK_OF(type) struct stack_st_##type

X509* d2i_X509_bio(BIO* bp, X509** x509)
{
    X509* cert = malloc(sizeof(X509));

    return cert;
}

X509* PEM_read_bio_X509(BIO* bp, X509** x, pem_password_cb* cb, void* u)
{
    X509* cert = calloc(1, sizeof(X509));

    return cert;
}

X509_STORE* X509_STORE_new(void)
{
    X509_STORE* ret = malloc(sizeof(X509_STORE));

    return ret;
}

X509_STORE_CTX* X509_STORE_CTX_new(void)
{
    X509_STORE_CTX* ret = malloc(sizeof(X509_STORE_CTX));

    return ret;
}

int X509_STORE_CTX_init(X509_STORE_CTX* ctx, X509_STORE* store, X509* x509, STACK_OF(X509) * chain) { return 1; }

int X509_verify_cert(X509_STORE_CTX* ctx) { return 1; }

int X509_STORE_CTX_get_error(X509_STORE_CTX* ctx) { return 1; }

const char* X509_verify_cert_error_string(long n) { return "OK"; }

void X509_STORE_CTX_free(X509_STORE_CTX* ctx)
{
    free(ctx);
    return;
}

OPENSSL_STACK* sk_new_null(void)
{
    OPENSSL_STACK* ret = malloc(sizeof(OPENSSL_STACK));
    ret->data = NULL;

    return ret;
}

int sk_push(OPENSSL_STACK* st, const void* data)
{
    st->data = data;
    return 1;
}

void sk_free(OPENSSL_STACK* buf)
{
    if (buf->data != NULL) {
        free(buf->data);
    }
    free(buf);
    return;
}

void X509_STORE_CTX_trusted_stack(X509_STORE_CTX* ctx, STACK_OF(X509) * sk) { return; }

void X509_free(X509* buf)
{
    free(buf);
    return;
}

unsigned long ERR_peek_last_error(void) { return 1; }

void ERR_clear_error(void) {}

void ERR_error_string_n(unsigned long e, char* buf, size_t len) { return; }

int X509_CRL_free(X509_CRL* crl) { return 0; }

int X509_STORE_add_cert(X509_STORE* store, X509* temp_ca) { return 1; }

unsigned long long SSL_CTX_set_options(SSL_CTX* ctx, unsigned long long op) { return 1; }

long ssl_adp_set_mode(SSL* s, long op) { return 0; }

long ssl_adp_ctrl(SSL* s, int cmd, long larg, void* parg) { return 0; }

SSL* ssl_adp_new(SSL_CTX* ctx) { return NULL; }

void ssl_adp_free(SSL* s) { return; }
int ssl_adp_read(SSL* s, void* buf, int num) { return 0; }

int ssl_adp_write(SSL* s, const void* buf, int num) { return 0; }

int ssl_adp_set_fd(SSL* s, int fd) { return 0; }

void ssl_adp_ctx_free(SSL_CTX* a) { return; }

int ssl_adp_shutdown(SSL* s) { return 0; }

int ssl_adp_get_error(const SSL* s, int i) { return 0; }

int ssl_adp_do_handshake(SSL* s) { return 0; }

void ssl_adp_set_accept_state(SSL* s) { return; }

void ssl_adp_set_connect_state(SSL* s) { return; }

void ssl_adp_clear_error(void) { return; }

void rs_ssl_deinit(struct rs_cb* rscb) { return; };

int rs_ssl_check_cert_chain(
    struct tls_cert_mng_info* mng_info, struct RsCerts* certs, struct tls_ca_new_certs* new_certs)
{
    return 0;
}

int rs_ssl_get_crl_data(struct rs_cb* rscb, FILE* fp, struct tls_cert_mng_info* mng_info, X509_CRL** crl) { return 0; }

int rs_ssl_crl_init(SSL_CTX* ssl_ctx, struct rs_cb* rscb, struct tls_cert_mng_info* mng_info) { return 0; }

int rs_ssl_put_certs(
    struct rs_cb* rscb, struct tls_cert_mng_info* mng_info, struct RsCerts* certs, struct tls_ca_new_certs* new_certs,
    struct CertFile* file_name)
{
    return 0;
}

int rs_ssl_verify_cert_chain(
    X509_STORE_CTX* ctx, X509_STORE* store, struct RsCerts* certs, struct tls_cert_mng_info* mng_info,
    struct tls_ca_new_certs* new_certs)
{
    return 0;
}

X509* tls_load_cert(const uint8_t* inbuf, uint32_t buf_len, int type) { return NULL; }

int rs_remove_certs(const char* end_file, const char* ca_file) { return 0; }

int rs_ssl_skid_get_from_chain(
    struct rs_cb* rscb, struct tls_cert_mng_info* mng_info, struct RsCerts* certs, struct tls_ca_new_certs* new_certs)
{
    return 0;
}

int rs_ssl_init(struct rs_cb* rscb) { return 0; }

int rs_get_pridata(struct rs_cb* rscb, struct RsSecPara* rs_para, struct tls_cert_mng_info* mng_info) { return 0; }

int rs_get_pk(struct rs_cb* rscb, struct tls_cert_mng_info* mng_info, EVP_PKEY** pky) { return 0; }

void rs_ssl_err_string(int fd, int err) { return; }

int rs_tls_peer_cert_verify(SSL* ssl, struct rs_cb* rscb) { return 0; }
