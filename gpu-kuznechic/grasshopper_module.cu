/*
 * grasshopper_module.cu
 * CPython Extension Module — Кузнечик CTR на GPU
 *
 * Компилируется nvcc в grasshopper_gpu.so
 * Импортируется как: import grasshopper_gpu
 *
 * Экспортирует:
 *   grasshopper_gpu.encrypt(data: bytes, rk: bytes, nonce: bytes) -> bytes
 *   grasshopper_gpu.decrypt(data: bytes, rk: bytes, nonce: bytes) -> bytes
 *   grasshopper_gpu.init(tables_path: str) -> None
 *   grasshopper_gpu.gpu_name() -> str
 *
 * Архитектура:
 *   - Всё состояние в статических C-переменных (инициализируется один раз)
 *   - Pinned memory буферы для DMA (два пинга-понга)
 *   - Два CUDA stream для конвейера: пока stream[0] шифрует, stream[1] копирует
 *   - GIL отпускается на время GPU работы
 *   - Нет NumPy, нет CuPy overhead — только CUDA Runtime API
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ─── Параметры конвейера ──────────────────────────────────────────────────── */
#define CHUNK_SIZE     (256 * 1024 * 1024)   /* 256 МБ — размер одного чанка   */
#define NUM_STREAMS    2                      /* двойная буферизация            */
#define THREADS_PER_BLOCK 256
#define T_TABLE_UINTS  16384                  /* 16 * 256 * 4 uint32            */
#define T_TABLE_BYTES  (T_TABLE_UINTS * 4)   /* 64 КБ                          */
#define RK_UINTS       40                     /* 10 ключей * 4 uint32           */

/* ─── Глобальное состояние (инициализируется один раз) ────────────────────── */
static int      g_initialized = 0;

/* Pinned memory — по два буфера вход/выход на каждый stream */
static void*    g_pinned_in [NUM_STREAMS];
static void*    g_pinned_out[NUM_STREAMS];

/* GPU буферы */
static uint4*   g_d_in [NUM_STREAMS];
static uint4*   g_d_out[NUM_STREAMS];

/* T-tables в GPU памяти */
static unsigned int* g_d_tables = NULL;

/* CUDA streams */
static cudaStream_t g_streams[NUM_STREAMS];

/* ─── CUDA kernel (inline) ─────────────────────────────────────────────────── */
/*
 * Компилируем всё одной командой nvcc.
 */

extern __shared__ unsigned int s_all[];

__device__ __forceinline__ unsigned int _swap32(unsigned int x) {
    return ((x << 24) & 0xff000000u) | ((x << 8) & 0x00ff0000u) |
           ((x >> 8) & 0x0000ff00u) | ((x >> 24) & 0x000000ffu);
}

__device__ void _load_t_tables(const unsigned int* global_T) {
    unsigned int* s_ptr = s_all;
    for (int i = threadIdx.x; i < T_TABLE_UINTS; i += blockDim.x)
        s_ptr[i] = global_T[i];
    __syncthreads();
}

struct _Block128 { unsigned int w[4]; };

__device__ __forceinline__ _Block128 _apply_t_layer(_Block128 state) {
    _Block128 next = {{0,0,0,0}};
    unsigned int (*s_T)[256][4] = (unsigned int (*)[256][4])s_all;
    #pragma unroll
    for (int i = 0; i < 16; i++) {
        unsigned char b = (unsigned char)(state.w[i/4] >> ((i%4)*8));
        next.w[0] ^= s_T[i][b][0];
        next.w[1] ^= s_T[i][b][1];
        next.w[2] ^= s_T[i][b][2];
        next.w[3] ^= s_T[i][b][3];
    }
    return next;
}

__global__ void grasshopper_ctr_kernel(
    const uint4* __restrict__ data_in,
    uint4*       __restrict__ data_out,
    const uint4* __restrict__ rk,
    const unsigned char* __restrict__ nonce,
    const unsigned int*  __restrict__ global_T,
    int n_blocks,
    int offset_idx
) {
    _load_t_tables(global_T);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_blocks) return;

    /* Счётчик: nonce[8] || big-endian(offset_idx + idx)[8] */
    _Block128 ctr;
    const unsigned int* n32 = (const unsigned int*)nonce;
    ctr.w[0] = n32[0];
    ctr.w[1] = n32[1];
    unsigned long long b_idx = (unsigned long long)(offset_idx + idx);
    ctr.w[2] = _swap32((unsigned int)(b_idx >> 32));
    ctr.w[3] = _swap32((unsigned int)(b_idx & 0xFFFFFFFFULL));

    /* 9 полных раундов T + XOR */
    #pragma unroll
    for (int r = 0; r < 9; r++) {
        uint4 k = rk[r];
        ctr.w[0] ^= k.x; ctr.w[1] ^= k.y; ctr.w[2] ^= k.z; ctr.w[3] ^= k.w;
        ctr = _apply_t_layer(ctr);
    }
    /* Последний раунд: только XOR с K10 */
    uint4 k9 = rk[9];
    ctr.w[0] ^= k9.x; ctr.w[1] ^= k9.y; ctr.w[2] ^= k9.z; ctr.w[3] ^= k9.w;

    /* XOR с данными */
    uint4 chunk = data_in[idx];
    chunk.x ^= ctr.w[0]; chunk.y ^= ctr.w[1];
    chunk.z ^= ctr.w[2]; chunk.w ^= ctr.w[3];
    data_out[idx] = chunk;
}

/* ─── Вспомогательные функции ──────────────────────────────────────────────── */

#define CUDA_CHECK(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        PyErr_Format(PyExc_RuntimeError, "CUDA error at %s:%d — %s", \
                     __FILE__, __LINE__, cudaGetErrorString(_e)); \
        return NULL; \
    } \
} while(0)

#define CUDA_CHECK_VOID(call) do { \
    cudaError_t _e = (call); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(_e)); \
    } \
} while(0)

/* ─── grasshopper_gpu.init(tables_path) ────────────────────────────────────── */
static PyObject* gh_init(PyObject* self, PyObject* args) {
    const char* tables_path;
    if (!PyArg_ParseTuple(args, "s", &tables_path))
        return NULL;

    if (g_initialized) {
        Py_RETURN_NONE;  /* уже инициализирован — idempotent */
    }

    /* Загружаем T-tables из файла */
    FILE* f = fopen(tables_path, "rb");
    if (!f) {
        PyErr_Format(PyExc_FileNotFoundError, "Cannot open tables file: %s", tables_path);
        return NULL;
    }
    unsigned int* h_tables = (unsigned int*)malloc(T_TABLE_BYTES);
    if (!h_tables) { fclose(f); return PyErr_NoMemory(); }

    size_t read = fread(h_tables, 4, T_TABLE_UINTS, f);
    fclose(f);
    if (read != T_TABLE_UINTS) {
        free(h_tables);
        PyErr_SetString(PyExc_IOError, "tables.bin: unexpected size");
        return NULL;
    }

    /* GPU: T-tables */
    CUDA_CHECK(cudaMalloc((void**)&g_d_tables, T_TABLE_BYTES));
    CUDA_CHECK(cudaMemcpy(g_d_tables, h_tables, T_TABLE_BYTES, cudaMemcpyHostToDevice));
    free(h_tables);

    /* Streams + pinned + GPU буферы */
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&g_streams[i], cudaStreamNonBlocking));
        CUDA_CHECK(cudaHostAlloc(&g_pinned_in[i],  CHUNK_SIZE, cudaHostAllocDefault));
        CUDA_CHECK(cudaHostAlloc(&g_pinned_out[i], CHUNK_SIZE, cudaHostAllocDefault));
        CUDA_CHECK(cudaMalloc((void**)&g_d_in[i],  CHUNK_SIZE));
        CUDA_CHECK(cudaMalloc((void**)&g_d_out[i], CHUNK_SIZE));
    }

    /* Устанавливаем лимит shared memory для kernel */
    CUDA_CHECK(cudaFuncSetAttribute(
        grasshopper_ctr_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        65536
    ));

    g_initialized = 1;
    printf("[grasshopper_gpu] Initialized. Pinned: %d MB × %d streams\n",
           CHUNK_SIZE / (1024*1024), NUM_STREAMS);

    Py_RETURN_NONE;
}

/* ─── Ядро конвейера — вызывается без GIL ────────────────────────────────────
 *
 * Правильный double buffering:
 *
 *   Итерация 0: запускаем stream[0] асинхронно, не ждём
 *   Итерация 1: запускаем stream[1] асинхронно, синхронизируем stream[0], забираем результат
 *   Итерация 2: запускаем stream[0] асинхронно, синхронизируем stream[1], забираем результат
 *   ...
 *   После цикла: синхронизируем последний активный stream
 *
 * Пока stream[i] шифрует, CPU копирует данные в pinned буфер stream[i^1].
 */

static int _run_pipeline(
    const unsigned char* input,
    unsigned char*       output,
    size_t               total_size,
    const unsigned int*  h_rk,       /* 40 uint32 = 160 байт */
    const unsigned char* nonce        /* 8 байт */
) {
    /* Загружаем round keys на GPU — маленькие, быстро */
    uint4* d_rk = NULL;
    if (cudaMalloc((void**)&d_rk, RK_UINTS * 4) != cudaSuccess) return -1;
    if (cudaMemcpy(d_rk, h_rk, RK_UINTS * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_rk); return -1;
    }

    /* Nonce на GPU */
    unsigned char* d_nonce = NULL;
    if (cudaMalloc((void**)&d_nonce, 8) != cudaSuccess) {
        cudaFree(d_rk); return -1;
    }
    if (cudaMemcpy(d_nonce, nonce, 8, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d_rk); cudaFree(d_nonce); return -1;
    }

    int num_chunks = (int)((total_size + CHUNK_SIZE - 1) / CHUNK_SIZE);
    int last_s_idx = -1;

    for (int chunk_i = 0; chunk_i < num_chunks; chunk_i++) {
        size_t offset    = (size_t)chunk_i * CHUNK_SIZE;
        size_t curr_size = (offset + CHUNK_SIZE <= total_size)
                           ? CHUNK_SIZE : (total_size - offset);
        int    s_idx     = chunk_i % NUM_STREAMS;
        int    prev_s    = (chunk_i > 0) ? ((chunk_i - 1) % NUM_STREAMS) : -1;
        size_t prev_offset = (chunk_i > 0) ? ((size_t)(chunk_i-1) * CHUNK_SIZE) : 0;
        size_t prev_size   = (chunk_i > 0)
            ? ((prev_offset + CHUNK_SIZE <= total_size) ? CHUNK_SIZE : (total_size - prev_offset))
            : 0;

        /* Пока запускаем текущий chunk — синхронизируем предыдущий stream и забираем результат */
        /* Это делается ДО memcpy текущего, чтобы не заблокировать DMA */
        if (prev_s >= 0) {
            cudaStreamSynchronize(g_streams[prev_s]);
            memcpy(output + prev_offset, g_pinned_out[prev_s], prev_size);
        }

        /* Копируем текущий chunk в pinned memory (CPU → pinned, без DMA) */
        memcpy(g_pinned_in[s_idx], input + offset, curr_size);

        int n_blocks    = (int)(curr_size / 16);
        int block_offset = (int)(offset / 16);
        int grid = (n_blocks + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        cudaStream_t stream = g_streams[s_idx];

        /* Async H→D */
        cudaMemcpyAsync(
            g_d_in[s_idx], g_pinned_in[s_idx],
            curr_size, cudaMemcpyHostToDevice, stream
        );

        /* Kernel */
        void* kargs[] = {
            &g_d_in[s_idx], &g_d_out[s_idx],
            &d_rk, &d_nonce, &g_d_tables,
            &n_blocks, &block_offset
        };
        cudaLaunchKernel(
            (void*)grasshopper_ctr_kernel,
            dim3(grid), dim3(THREADS_PER_BLOCK),
            kargs, 65536, stream
        );

        /* Async D→H */
        cudaMemcpyAsync(
            g_pinned_out[s_idx], g_d_out[s_idx],
            curr_size, cudaMemcpyDeviceToHost, stream
        );

        last_s_idx = s_idx;
    }

    /* Забираем результат последнего chunk */
    if (last_s_idx >= 0) {
        size_t last_offset = (size_t)(num_chunks - 1) * CHUNK_SIZE;
        size_t last_size   = total_size - last_offset;
        cudaStreamSynchronize(g_streams[last_s_idx]);
        memcpy(output + last_offset, g_pinned_out[last_s_idx], last_size);
    }

    cudaFree(d_rk);
    cudaFree(d_nonce);
    return 0;
}

/* ─── grasshopper_gpu.encrypt(data, rk, nonce) → bytes ─────────────────────── */
static PyObject* gh_encrypt(PyObject* self, PyObject* args) {
    const unsigned char* data;
    Py_ssize_t           data_len;
    const unsigned char* rk;
    Py_ssize_t           rk_len;
    const unsigned char* nonce;
    Py_ssize_t           nonce_len;

    if (!PyArg_ParseTuple(args, "y#y#y#",
            &data,  &data_len,
            &rk,    &rk_len,
            &nonce, &nonce_len))
        return NULL;

    if (rk_len != 160) {
        PyErr_SetString(PyExc_ValueError, "rk must be 160 bytes (10 * 16)");
        return NULL;
    }
    if (nonce_len != 8) {
        PyErr_SetString(PyExc_ValueError, "nonce must be 8 bytes");
        return NULL;
    }
    if (data_len % 16 != 0) {
        PyErr_SetString(PyExc_ValueError, "data length must be multiple of 16 (pad before calling)");
        return NULL;
    }
    if (!g_initialized) {
        PyErr_SetString(PyExc_RuntimeError, "grasshopper_gpu not initialized, call init() first");
        return NULL;
    }

    /* Выделяем PyBytes под результат */
    PyObject* result = PyBytes_FromStringAndSize(NULL, data_len);
    if (!result) return NULL;

    unsigned char* out_buf = (unsigned char*)PyBytes_AS_STRING(result);

    /* Отпускаем GIL — GPU работает независимо */
    Py_BEGIN_ALLOW_THREADS
    _run_pipeline(
        data, out_buf, (size_t)data_len,
        (const unsigned int*)rk, nonce
    );
    Py_END_ALLOW_THREADS

    return result;
}

/* decrypt == encrypt в CTR */
static PyObject* gh_decrypt(PyObject* self, PyObject* args) {
    return gh_encrypt(self, args);
}

/* ─── grasshopper_gpu.gpu_name() ────────────────────────────────────────────── */
static PyObject* gh_gpu_name(PyObject* self, PyObject* args) {
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    return PyUnicode_FromString(props.name);
}

/* ─── Таблица методов модуля ─────────────────────────────────────────────────── */
static PyMethodDef GhMethods[] = {
    {"init",     gh_init,     METH_VARARGS, "init(tables_path) — load tables, alloc GPU resources"},
    {"encrypt",  gh_encrypt,  METH_VARARGS, "encrypt(data, rk, nonce) -> bytes"},
    {"decrypt",  gh_decrypt,  METH_VARARGS, "decrypt(data, rk, nonce) -> bytes"},
    {"gpu_name", gh_gpu_name, METH_NOARGS,  "gpu_name() -> str"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef GhModule = {
    PyModuleDef_HEAD_INIT, "grasshopper_gpu", NULL, -1, GhMethods
};

PyMODINIT_FUNC PyInit_grasshopper_gpu(void) {
    return PyModule_Create(&GhModule);
}
