#pragma once

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define CW_OPERATION_MODULE_ABI_V1 "chronowire.operation-module.v1"
#define CW_OPERATION_MODULE_SYMBOL_V1 chronowire_operation_module_v1

/* 全entrypointはC++例外を境界内で捕捉し、error bufferと戻り値へ変換しなければならない。 */

/** 一Emissionのread-only連続float64値とresolved item shape。 */
typedef struct {
    const double* values;
    size_t value_count;
    const size_t* shape;
    size_t rank;
} CwBufferViewV1;

/** Chronowireが確保した単一output buffer。moduleはcapacityを超えて書き込まない。 */
typedef struct {
    double* values;
    size_t value_capacity;
    const size_t* shape;
    size_t rank;
} CwMutableBufferViewV1;

/** process一回の出力件数、status、および任意の単一Diagnostic。 */
typedef struct {
    size_t output_count;
    uint8_t status;
    uint8_t diagnostic_severity;
    const char* diagnostic_code;
    const char* diagnostic_message;
} CwProcessResultV1;

/** immutable float64 Config列からrun-local sessionを生成する。失敗時はNULLを返す。 */
typedef void* (*CwCreateFnV1)(
    const double* parameters,
    size_t parameter_count,
    char* error_message,
    size_t error_capacity
);

/** 名前順に固定されたinput列を処理する。成功は0、契約違反は非0を返す。 */
typedef int (*CwProcessFnV1)(
    void* session,
    const CwBufferViewV1* inputs,
    size_t input_count,
    CwMutableBufferViewV1* output,
    CwProcessResultV1* result,
    char* error_message,
    size_t error_capacity
);

/** EOFで残留出力を生成する任意entry。v0.4 CppExecutorでは未対応。 */
typedef int (*CwFlushFnV1)(
    void* session,
    CwMutableBufferViewV1* output,
    CwProcessResultV1* result,
    char* error_message,
    size_t error_capacity
);

/** 部分初期化を含むsessionを一度だけ安全に解放する。 */
typedef void (*CwDestroyFnV1)(void* session);

/** operation IDとversion付き実装function table。flags bit 0=flush、bit 1=session-local。 */
typedef struct {
    uint32_t struct_size;
    const char* operation_id;
    const char* implementation_id;
    const char* abi_version;
    const char* process_model;
    size_t workspace_size_bytes;
    size_t workspace_alignment_bytes;
    uint32_t flags;
    CwCreateFnV1 create;
    CwProcessFnV1 process;
    CwFlushFnV1 flush;
    CwDestroyFnV1 destroy;
} CwOperationEntryV1;

/** 一共有libraryがexportする重複のないOperation entry列。 */
typedef struct {
    uint32_t struct_size;
    const char* module_abi_version;
    size_t operation_count;
    const CwOperationEntryV1* operations;
} CwOperationModuleV1;

/** `chronowire_operation_module_v1` symbolのfunction型。戻り値はmodule lifetime中有効。 */
typedef const CwOperationModuleV1* (*CwGetOperationModuleFnV1)(void);

#ifdef __cplusplus
}
#endif
