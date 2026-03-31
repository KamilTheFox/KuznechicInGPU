# Architecture Deep Dive

> Технический разбор реализации ГОСТ Р 34.12-2015 «Кузнечик» на CUDA.  
> Для понимания достаточно базового знания C и общих представлений о GPU.

---

## Содержание

1. [Почему Кузнечик сложен для GPU](#1-почему-кузнечик-сложен-для-gpu)
2. [Алгебраическое преобразование: T-tables](#2-алгебраическое-преобразование-t-tables)
3. [CUDA kernel: архитектура и оптимизации](#3-cuda-kernel-архитектура-и-оптимизации)
4. [Конвейер данных: от FastAPI до GPU и обратно](#4-конвейер-данных-от-fastapi-до-gpu-и-обратно)
5. [CPython C Extension: почему и как](#5-cpython-c-extension-почему-и-как)
6. [Режим CTR и счётчик](#6-режим-ctr-и-счётчик)
7. [Side-channel анализ](#7-side-channel-анализ)
8. [Профиль производительности и узкие места](#8-профиль-производительности-и-узкие-места)

---

## 1. Почему Кузнечик сложен для GPU

GPU эффективен когда тысячи потоков выполняют **одну и ту же операцию над разными данными** (SIMT — Single Instruction Multiple Threads). Проблема Кузнечика в том, что внутри каждого блока данных есть длинная последовательная цепочка зависимостей.

### Линейное преобразование L

По ГОСТ, L — это 16 последовательных применений функции R (шаг LFSR):

```
L(a) = R¹⁶(a)
```

Функция R берёт 128-битный вектор и сдвигает его: новый байт вычисляется как линейная комбинация всех текущих байт в GF(2⁸), остальные байты сдвигаются:

```
R(a₀, a₁, ..., a₁₅) = (x, a₀, a₁, ..., a₁₄)
где x = LC[0]·a₁₅ ⊕ LC[1]·a₀ ⊕ ... ⊕ LC[15]·a₁₄
```

Это создаёт цепочку из 16 зависимых шагов — каждый следующий R зависит от результата предыдущего:

```
block → R → result₁ → R → result₂ → ··· → R → result₁₆
               ↑ зависит     ↑ зависит
```

На 9 раундах шифрования это **144 последовательных операции** внутри одного 128-битного блока. GPU не может их распараллелить — это фундаментальное ограничение алгоритма.

### Сравнение с AES

AES имеет аппаратные инструкции (`VAESENC` на x86, аналоги на ARM). Без них AES тоже медленный. Для Кузнечика таких инструкций нет ни в одном CPU и ни в одном GPU — алгоритм никогда не получил массового применения на Западе где делается железо.

---

## 2. Алгебраическое преобразование: T-tables

### Идея

Поскольку L **линейно** над GF(2⁸), справедливо:

```
L(a ⊕ b) = L(a) ⊕ L(b)
```

Это означает что результат L для всего блока можно получить как XOR результатов L для каждого байта по отдельности:

```
L(S(block)) = L(S(b₀) · e₀) ⊕ L(S(b₁) · e₁) ⊕ ... ⊕ L(S(b₁₅) · e₁₅)
```

где `eᵢ` — единичный вектор с единицей в позиции i.

Каждое слагаемое зависит только от одного байта `bᵢ`. Значит его можно **предвычислить** для всех 256 возможных значений байта:

```
T[i][b] = L( eᵢ · PI[b] )    для i=0..15, b=0..255
```

### Применение

Теперь весь раунд S+L сводится к 16 lookup + XOR:

```c
Block128 result = {0, 0, 0, 0};
for (int i = 0; i < 16; i++) {
    uint8_t b = (state.w[i/4] >> ((i%4)*8)) & 0xFF;
    result.w[0] ^= T[i][b][0];
    result.w[1] ^= T[i][b][1];
    result.w[2] ^= T[i][b][2];
    result.w[3] ^= T[i][b][3];
}
```

Все 16 итераций **независимы** — нет цепочки зависимостей. Компилятор (`nvcc -O3`) разворачивает цикл через `#pragma unroll` и планирует инструкции оптимально.

### Размер таблиц

```
16 позиций × 256 значений × 16 байт (128 бит) = 65536 байт = 64 КБ
```

Именно столько занимает `tables.bin`.

### Генерация таблиц

`t_table_generation.py` вычисляет таблицы через честную реализацию L и S из ГОСТ, упаковывает каждые 16 байт в 4 `uint32` (для удобной работы в CUDA через `uint4`):

```python
for pos in range(16):
    for val in range(256):
        block = [0] * 16
        block[pos] = S_BOX[val]
        res_bytes = l_func(block)          # честный L из ГОСТ
        t_tables[pos, val] = np.frombuffer(bytes(res_bytes), dtype=np.uint32)
```

---

## 3. CUDA kernel: архитектура и оптимизации

### Модель параллелизма

```
Один CUDA thread = один 128-битный блок данных CTR
```

При 64 МБ данных это ~4 миллиона потоков, запущенных одновременно. RTX 3060 имеет 3584 CUDA cores — они обрабатывают потоки группами (warps по 32).

### Shared Memory: загрузка T-tables

64 КБ T-tables загружаются в Shared Memory в начале каждого CUDA block:

```c
__device__ void load_t_tables(const unsigned int* global_T) {
    unsigned int* s_ptr = s_all;
    for (int i = threadIdx.x; i < 16384; i += blockDim.x)
        s_ptr[i] = global_T[i];
    __syncthreads();
}
```

Все 256 потоков блока участвуют в загрузке параллельно: поток 0 загружает элементы 0, 256, 512..., поток 1 — 1, 257, 513... Это **coalesced access** к global memory.

После `__syncthreads()` все потоки в блоке видят заполненную Shared Memory.

**Почему 64 КБ влезает при лимите 48 КБ**: статический лимит Ampere 48 КБ, но driver API позволяет поднять его до 100 КБ через:
```c
cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 65536);
```
Это работает за счёт уменьшения числа одновременно активных блоков на SM — компромисс между occupancy и объёмом памяти на блок.

### uint4 векторизация

Вместо побайтовой работы все данные представлены как `uint4` — 4 × 32-бит = 128 бит за одну транзакцию:

```c
uint4 chunk = data_in[idx];   // читает 16 байт за одну инструкцию
chunk.x ^= ctr.w[0];
chunk.y ^= ctr.w[1];
chunk.z ^= ctr.w[2];
chunk.w ^= ctr.w[3];
data_out[idx] = chunk;        // пишет 16 байт за одну инструкцию
```

Это обеспечивает **coalesced access** к global memory: соседние потоки читают соседние 16-байтовые блоки, что превращается в одну широкую транзакцию памяти.

### Счётчик CTR и swap32

Счётчик занимает старшие 8 байт 128-битного блока и инкрементируется как big-endian uint64 (по ГОСТ Р 34.13-2015):

```c
unsigned long long b_idx = (unsigned long long)(offset_idx + idx);
ctr.w[2] = swap32((unsigned int)(b_idx >> 32));
ctr.w[3] = swap32((unsigned int)(b_idx & 0xFFFFFFFFULL));
```

`swap32` переставляет байты вручную без системных вызовов — это быстрее `__byte_perm` и не требует дополнительных заголовков:

```c
__device__ __forceinline__ unsigned int swap32(unsigned int x) {
    return ((x << 24) & 0xff000000u) | ((x << 8) & 0x00ff0000u) |
           ((x >> 8) & 0x0000ff00u) | ((x >> 24) & 0x000000ffu);
}
```

---

## 4. Конвейер данных: от FastAPI до GPU и обратно

### Проблема наивного подхода

```python
# Медленно: pageable memory, два барьера
d_in = cp.asarray(data_np)          # CPU → системный буфер → GPU
result = kernel(d_in)
return result.get()                  # GPU → системный буфер → CPU
```

Каждая операция синхронна и копирует через промежуточный буфер ОС.

### Pinned Memory

```
Обычная (pageable) память:    CPU RAM → [OS buffer] → GPU VRAM
Pinned (page-locked) память:  CPU RAM ──────────────► GPU VRAM (DMA напрямую)
```

Pinned memory не может быть вытеснена на диск, поэтому CUDA DMA-контроллер работает с ней напрямую без участия CPU. Прирост ~3-4x по скорости копирования.

Мы выделяем pinned буферы **один раз** при старте (512 МБ × 2 = 1 ГБ pinned RAM), и переиспользуем их для каждого запроса.

### Double Buffering с CUDA Streams

Два потока (stream) позволяют перекрывать transfer и вычисления:

```
Время →

Stream 0: [copy chunk 0 H→D] [kernel chunk 0] [copy chunk 0 D→H]
Stream 1:                     [copy chunk 1 H→D] [kernel chunk 1] [copy chunk 1 D→H]
Stream 0:                                         [copy chunk 2 H→D] [kernel chunk 2] ...
```

Пока kernel обрабатывает chunk 0 на GPU, CPU копирует chunk 1 в pinned буфер stream 1. Пока chunk 1 едет по PCIe, GPU уже шифрует chunk 0.

**Критически важно**: синхронизировать stream N нужно **перед** тем как переиспользовать его буферы — не сразу после запуска. Иначе конвейер разрушается:

```c
// Правильно: синхронизируем предыдущий stream пока готовим текущий
if (prev_stream >= 0) {
    cudaStreamSynchronize(streams[prev_stream]);
    memcpy(output + prev_offset, pinned_out[prev_stream], prev_size);
}
memcpy(pinned_in[curr_stream], input + curr_offset, curr_size);  // готовим текущий
// запускаем curr_stream асинхронно...
```

---

## 5. CPython C Extension: почему и как

### Проблема Python-прослойки

При 500 МБ/с данных Python тратит значительное время на:
- создание объектов `bytes` при каждом вызове
- копирование буферов между NumPy/CuPy объектами
- GIL — блокировка интерпретатора не даёт другим запросам обрабатываться пока идёт GPU работа

### Решение: `grasshopper_gpu.so`

CPython C Extension — это `.so` файл с функцией `PyInit_grasshopper_gpu`, которую Python находит при `import grasshopper_gpu`. После этого вызов `_gh.encrypt(data, rk, nonce)` идёт напрямую в C без интерпретатора.

```c
static PyObject* gh_encrypt(PyObject* self, PyObject* args) {
    const unsigned char* data;
    Py_ssize_t data_len;
    // ...
    PyArg_ParseTuple(args, "y#y#y#", &data, &data_len, &rk, &rk_len, &nonce, &nonce_len);

    PyObject* result = PyBytes_FromStringAndSize(NULL, data_len);  // выделяем PyBytes
    unsigned char* out_buf = (unsigned char*)PyBytes_AS_STRING(result);

    Py_BEGIN_ALLOW_THREADS   // ← отпускаем GIL
    _run_pipeline(data, out_buf, data_len, rk, nonce);  // весь GPU pipeline в C
    Py_END_ALLOW_THREADS     // ← берём GIL обратно

    return result;
}
```

`Py_BEGIN_ALLOW_THREADS` / `Py_END_ALLOW_THREADS` — это макросы CPython которые освобождают GIL на время C-кода. FastAPI может обрабатывать другие запросы пока GPU шифрует.

### Сборка через nvcc

Python extensions обычно собираются через `setuptools`, но они не поддерживают `.cu` файлы нативно. Мы используем `nvcc` напрямую:

```bash
nvcc grasshopper_module.cu \
    -O3 -arch=sm_86 --use_fast_math -std=c++11 \
    -Xcompiler -fPIC --shared \
    -I$(python -c "import sysconfig; print(sysconfig.get_path('include'))") \
    -Xlinker --allow-shlib-undefined \   # символы Python резолвятся из интерпретатора
    -lcudart \
    -o grasshopper_gpu.cpython-310-x86_64-linux-gnu.so
```

`--allow-shlib-undefined` — стандартный подход для Python extensions: они не линкуются с libpython явно, а резолвят символы из родительского процесса (интерпретатора) в рантайме.

---

## 6. Режим CTR и счётчик

CTR (Counter Mode) превращает блочный шифр в поточный. Шифруется не сам plaintext, а счётчик — результат XOR-ится с данными:

```
KeyStream[i] = Encrypt(Key, Nonce || Counter[i])
Ciphertext[i] = Plaintext[i] ⊕ KeyStream[i]
```

Два следствия важных для GPU:

**Параллелизм**: каждый блок независим (счётчик разный, но вычисляется из `idx` напрямую). Нет зависимостей между блоками — идеально для GPU.

**Симметричность**: decrypt == encrypt (XOR обратим). Один kernel для обоих операций.

**Структура 128-битного блока** (ГОСТ Р 34.13-2015):

```
[  Nonce (8 байт)  |  Counter big-endian (8 байт)  ]
     фиксированный       0, 1, 2, 3, ...
```

---

## 7. Side-channel анализ

### Timing Attacks (закрыты)

Классическая timing attack использует то, что время выполнения операции зависит от секретных данных — например, условный переход `if (key_bit) { ... }`.

В нашем kernel нет ни одного `if` зависящего от ключа или plaintext. Все ветвления — только по `threadIdx` и `blockIdx` (структурные, не зависящие от данных). Архитектура SIMT гарантирует что все потоки warp выполняют одни и те же инструкции.

### Cache-Timing Attacks (закрыты)

На CPU таблицы S-box и T-tables кэшируются аппаратно (L1/L2/L3). Атака Бернштейна (2005) измеряет время доступа к AES-таблицам через разницу в cache hit/miss и восстанавливает ключ за ~65 миллионов шифрований.

В нашей реализации T-tables находятся в **Shared Memory** — программно-управляемом кэше. Ключевые отличия:
- Shared Memory загружается явно один раз перед шифрованием
- Латентность любого обращения к Shared Memory **фиксирована** и не зависит от того, обращался ли кто-то к этому адресу раньше
- Нет механизма "выбрасывания строки кэша" как в CPU

### Остаточный канал: Bank Conflicts

Shared Memory на Ampere разделена на 32 банка по 4 байта. Если несколько потоков одного warp обращаются к одному банку — происходит **bank conflict** и обращения сериализуются.

Индексы обращений `T[i][b]` зависят от байтов блока данных (которые являются функцией ключа и plaintext). Теоретически это создаёт timing-канал:

```
Время обработки ≈ f(bank_conflicts) = f(data)
```

На практике: разница во времени между 0 и максимальным количеством conflicts составляет единицы наносекунд. Измерение требует физического доступа к GPU и специализированного оборудования (осциллограф с GHz-полосой или EM-зонд). Это за пределами модели угроз для сетевого сервиса.

---

## 8. Профиль производительности и узкие места

### Что мы измеряем

`/benchmark` измеряет **end-to-end** throughput: от принятия `bytes` в Python до возврата зашифрованного `bytes`. Включает:
- вычисление round keys на CPU (~10 мкс, пренебрежимо)
- PKCS7 padding (мгновенно)
- конвейер GPU: H→D transfer + kernel + D→H transfer

### Текущий потолок

RTX 3060 PCIe 4.0 x16 имеет теоретический bandwidth ~30 ГБ/с. Реальный при нашей нагрузке — ~5-8 ГБ/с из-за overhead на синхронизацию и управление потоками.

500 МБ/с = ~0.5 ГБ/с — мы используем ~6-10% от пикового PCIe bandwidth. Остаток уходит на:
- Python overhead при маршалинге `bytes` объектов (~10%)
- Round keys через gostcrypto (~1%, пренебрежимо)
- Overhead CUDA stream management (~5%)
- Kernel latency (L-transform через T-tables, bank conflicts) (~75%)

### Путь к 5+ ГБ/с

Без Python вообще — C++ сервер (например, cpp-httplib + CUDA) убирает все Python-накладные расходы. На этом же железе реально достичь 2-3 ГБ/с.

С GPU-resident обработкой — данные никогда не покидают VRAM (GPU-side HTTP stack, DMA от NVMe напрямую в VRAM). Теоретический потолок — memory bandwidth GPU (~360 ГБ/с для RTX 3060), практически ~10-20 ГБ/с для этого алгоритма.