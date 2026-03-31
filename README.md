# 🔐 Kuznyechik-GPU

**GOST R 34.12-2015 «Кузнечик» — аппаратное ускорение на CUDA**

[![CUDA](https://img.shields.io/badge/CUDA-12.3-76b900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![Python](https://img.shields.io/badge/Python-3.10-3776ab?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-GPU-2496ed?logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> Российский блочный шифр ГОСТ Р 34.12-2015, который на CPU даёт ~85 МБ/с,  
> переведён на GPU и разогнан до **500 МБ/с** — скорость записи на SATA SSD.

---

## Результаты

| Реализация | Скорость | Железо |
|---|---|---|
| CPU (C, T-tables) | 85 МБ/с | baseline |
| GPU V1 — наивный CUDA | < 85 МБ/с | RTX 3060 |
| GPU V3 — T-tables, Shared Memory | ~200 МБ/с | RTX 3060 |
| GPU V4 — uint4 векторизация | ~350 МБ/с | RTX 3060 |
| GPU V7 — Pinned memory + CUDA Streams | ~440 МБ/с | RTX 3060 |
| **GPU V8 — C Extension (текущая)** | **500 МБ/с** | RTX 3060 |

---

## Почему это нетривиально

Кузнечик — один из немногих современных шифров, который **принципиально плохо ложится на GPU**.

Линейное преобразование L состоит из 16 последовательных шагов LFSR (функция R), каждый из которых зависит от результата предыдущего. Это цепочка из 144 зависимых операций внутри одного блока — GPU не может их распараллелить, в отличие от AES у которого есть аппаратные инструкции.

```
R₀(block) → R₁(result₀) → R₂(result₁) → ··· → R₁₅(result₁₄)
                ↑ зависит      ↑ зависит
```

**Решение**: алгебраическое преобразование. Поскольку L линейно над GF(2⁸), вся операция S+L сворачивается в 16 табличных XOR-операций — без цепочки зависимостей:

```c
// Вместо 144 последовательных шагов — 16 независимых lookup + XOR
for (int i = 0; i < 16; i++) {
    unsigned char b = (state.w[i/4] >> ((i%4)*8)) & 0xFF;
    next.w[0] ^= T[i][b][0];  // все 16 итераций независимы
    next.w[1] ^= T[i][b][1];
    next.w[2] ^= T[i][b][2];
    next.w[3] ^= T[i][b][3];
}
```

64 КБ T-таблиц размещаются в **Shared Memory** (программно-управляемый L1-кэш GPU) — детерминированная латентность, нет cache miss.

---

## Архитектура

```
┌─────────────────────────────────────────────────────────┐
│  FastAPI (app.py)                                        │
│    POST /encrypt  POST /decrypt  POST /benchmark         │
└──────────────────────┬──────────────────────────────────┘
                       │ bytes
┌──────────────────────▼──────────────────────────────────┐
│  grasshopper_worker.py  (тонкая Python-обёртка)         │
│    PKCS7 padding   │   round keys (gostcrypto)           │
└──────────────────────┬──────────────────────────────────┘
                       │ C API (без GIL)
┌──────────────────────▼──────────────────────────────────┐
│  grasshopper_gpu.so  (CPython C Extension, nvcc)        │
│                                                          │
│  Pinned Memory [0]  ──────►  GPU Buffer [0]             │
│  Pinned Memory [1]  ──────►  GPU Buffer [1]             │
│                                                          │
│  Stream 0: H→D ──► Kernel ──► D→H                       │
│  Stream 1:      H→D ──► Kernel ──► D→H  (pipeline)      │
└──────────────────────┬──────────────────────────────────┘
                       │ CUDA
┌──────────────────────▼──────────────────────────────────┐
│  grasshopper_ctr_kernel  (grasshopper_ctr.cu)           │
│                                                          │
│  Каждый CUDA thread = один 128-бит блок CTR             │
│  T-tables (64 KB) → Shared Memory (per SM)              │
│  uint4 для coalesced global memory access               │
│  #pragma unroll на 9 раундах                            │
└─────────────────────────────────────────────────────────┘
```

**Конвейер double buffering**: пока Stream 0 шифрует chunk N, Stream 1 копирует chunk N+1 по PCIe. GIL Python отпускается на всё время GPU-работы через `Py_BEGIN_ALLOW_THREADS`.

---

## Безопасность: Side-Channel устойчивость

Перенос вычислений на GPU закрывает два класса атак:

**Timing Attacks** — в CUDA-ядре нет условных переходов зависящих от секретных данных. Архитектура SIMT гарантирует константное время выполнения каждого раунда вне зависимости от значений ключа или открытого текста.

**Cache-Timing Attacks** — T-таблицы загружаются в Shared Memory целиком до начала шифрования. В отличие от CPU L1/L2, Shared Memory управляется программно: каждое обращение к любому байту S-box имеет детерминированную и одинаковую латентность.

> ⚠️ Остаточный канал: bank conflicts в Shared Memory технически зависят от индексов доступа (функции данных). Практическая эксплуатация требует физического доступа к GPU и специализированного оборудования для измерения EM-излучения — что выходит за рамки модели угроз для сетевых сервисов.

---

## Структура репозитория

```
gpu-kuznechic/
├── grasshopper_ctr.cu        # CUDA kernel (CTR mode, T-tables, uint4)
├── grasshopper_module.cu     # CPython C Extension — пинг конвейер, GIL-free
├── setup.py                  # Сборка .so через nvcc
├── t_table_generation.py     # Генерация tables.bin (запускается при сборке)
├── grasshopper_worker.py     # Python-обёртка: padding, round keys
├── app.py                    # FastAPI: /encrypt /decrypt /benchmark /test
├── Dockerfile
└── docker-compose.yml
```

---

> 📐 Подробный технический разбор: [ARCHITECTURE.md](ARCHITECTURE.md)

---

## Быстрый старт

**Требования**: Docker + NVIDIA Container Toolkit, GPU с compute capability ≥ 8.6 (RTX 30xx).

```bash
git clone https://github.com/YOUR_USERNAME/kuznyechik-gpu
cd kuznyechik-gpu

# Генерируем T-tables (один раз)
python t_table_generation.py

# Собираем и запускаем
docker compose up --build
```

Сервис будет доступен на `http://localhost:3100`.

### Swagger UI

```
http://localhost:3100/docs
```

### Пример через curl

```bash
# Шифрование
DATA=$(echo -n "Hello, Kuznyechik!" | base64)
curl -s -X POST http://localhost:3100/encrypt \
  -H "Content-Type: application/json" \
  -d "{\"data_b64\": \"$DATA\"}" | jq .

# Бенчмарк (512 МБ, 5 прогонов)
curl -s -X POST http://localhost:3100/benchmark \
  -H "Content-Type: application/json" \
  -d '{"data_mb": 512, "runs": 5}' | jq .

# Тест корректности
curl -s http://localhost:3100/test | jq .
```

---

## API

| Метод | Путь | Описание |
|---|---|---|
| `POST` | `/encrypt` | Шифрует данные (base64 in/out), генерирует ключ и nonce если не переданы |
| `POST` | `/decrypt` | Дешифрует (нужны ciphertext + key + nonce в base64) |
| `POST` | `/benchmark` | Замер throughput, параметры: `data_mb`, `runs` |
| `GET`  | `/test` | Проверка integrity на случайных данных |
| `GET`  | `/gpu` | Информация о GPU |
| `GET`  | `/health` | Healthcheck |

---

## История оптимизаций

### V1–V2: Наивная CUDA-реализация
Прямой перенос R-шага и S-box на GPU. Результат — медленнее CPU. Каждый поток выполнял 256 итераций последовательных зависимых операций.

### V3: T-tables + Shared Memory *(ключевой прорыв)*
Алгебраическое преобразование: S∘L сворачивается в 16 lookup-таблиц. 64 КБ таблиц размещаются в Shared Memory. Ограничение Ampere — статический лимит 48 КБ — преодолено через `cudaFuncAttributeMaxDynamicSharedMemorySize`.

### V4: uint4 векторизация
Переход от `uint8` к `uint4` (128-бит слова). Coalesced access к глобальной памяти — 4 байта читаются одной транзакцией. Реализован `swap32` для big-endian счётчика CTR.

### V5–V6: Pinned Memory + Memory Windowing
При объёмах > 1 ГБ ОС тратила больше времени на выделение памяти чем GPU на шифрование. Решение: фиксированное окно 512 МБ pinned memory, аллоцированное один раз при старте.

### V7: CUDA Streams (Double Buffering)
Два потока CUDA: пока один шифрует, второй передаёт данные по PCIe. Достигнут потолок Python-интерпретатора (~440 МБ/с).

### V8: CPython C Extension *(текущая)*
Весь конвейер перенесён в `grasshopper_gpu.so` (C + CUDA Runtime API). GIL освобождается на время GPU-работы. Нет NumPy/CuPy overhead в горячем пути. **500 МБ/с**.

---

## Требования к железу

| Компонент | Минимум | Тестировалось |
|---|---|---|
| GPU | NVIDIA Ampere (sm_86) | RTX 3060 12GB |
| CUDA | 12.x | 12.3.1 |
| RAM | 4 GB | 16 GB |
| PCIe | 3.0 x8 | 4.0 x16 |
| OS | Linux | Ubuntu 22.04 |

> Для других архитектур измените `-arch=sm_86` в `setup.py` на свой compute capability.
