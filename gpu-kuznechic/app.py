"""
app.py
FastAPI сервис — Кузнечик CTR на GPU.

Ручки:
    POST /encrypt          — шифрует данные, возвращает base64 шифртекст + nonce
    POST /decrypt          — дешифрует base64 шифртекст
    POST /benchmark        — запускает бенчмарк, возвращает throughput
    GET  /test             — сравнение GPU с эталонной реализацией gostcrypto
    GET  /gpu              — информация о GPU
    GET  /health           — healthcheck
"""

import os
import base64
import secrets
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

import grasshopper_worker as gw

# Эталонная реализация — используется только в /test для сравнения
import gostcrypto.gostcipher as _gostcipher


# ─── Lifespan ────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[startup] Прогрев GPU...")
    t0 = time.perf_counter()
    try:
        gw.benchmark(data_mb=128, runs=1)
        print(f"[startup] GPU прогрет за {time.perf_counter()-t0:.2f} сек.")
    except Exception as e:
        print(f"[startup] Ошибка прогрева: {e}")
    yield


app = FastAPI(
    title="Кузнечик CTR — GPU сервис",
    description="ГОСТ Р 34.12-2015 Grasshopper, режим CTR, ускорение на CUDA",
    version="1.0.0",
    lifespan=lifespan,
)


# ─── Модели ──────────────────────────────────────────────────────────────────

class EncryptRequest(BaseModel):
    data_b64:  str = Field(..., description="Данные (base64)")
    key_b64:   Optional[str] = Field(None, description="256-бит ключ (base64, 32 байта). Если пусто — генерируется.")
    nonce_b64: Optional[str] = Field(None, description="64-бит nonce (base64, 8 байт). Если пусто — генерируется.")

    @field_validator('data_b64')
    @classmethod
    def data_not_empty(cls, v):
        if not v.strip():
            raise ValueError("data_b64 не может быть пустым")
        return v


class EncryptResponse(BaseModel):
    ciphertext_b64:  str   = Field(..., description="Шифртекст (base64)")
    key_b64:         str   = Field(..., description="Использованный ключ (base64)")
    nonce_b64:       str   = Field(..., description="Использованный nonce (base64)")
    original_size:   int   = Field(..., description="Исходный размер данных (байт)")
    encrypted_size:  int   = Field(..., description="Размер шифртекста (байт, с padding)")
    elapsed_ms:      float = Field(..., description="Время шифрования (мс)")
    throughput_mb_s: float = Field(..., description="Throughput (МБ/с)")


class DecryptRequest(BaseModel):
    ciphertext_b64: str = Field(..., description="Шифртекст (base64)")
    key_b64:        str = Field(..., description="256-бит ключ (base64, 32 байта)")
    nonce_b64:      str = Field(..., description="64-бит nonce (base64, 8 байт)")


class DecryptResponse(BaseModel):
    data_b64:        str   = Field(..., description="Расшифрованные данные (base64)")
    size:            int   = Field(..., description="Размер данных (байт)")
    elapsed_ms:      float = Field(..., description="Время дешифрования (мс)")
    throughput_mb_s: float = Field(..., description="Throughput (МБ/с)")


class BenchmarkRequest(BaseModel):
    data_mb: int = Field(default=64,  ge=1,  le=2048, description="Размер тестовых данных (МБ)")
    runs:    int = Field(default=5,   ge=1,  le=20,   description="Количество прогонов")


# ─── Хелперы ─────────────────────────────────────────────────────────────────

def _decode_b64(s: str, name: str) -> bytes:
    try:
        return base64.b64decode(s)
    except Exception:
        raise HTTPException(status_code=400, detail=f"{name}: некорректный base64")

def _encode_b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def _ecb_encrypt_reference(plaintext: bytes, key: bytes) -> bytes:
    """
    Шифрование одного блока через эталонную реализацию gostcrypto (ECB).
    Используется только в /test для сравнения с GPU.
    plaintext должен быть ровно 16 байт.
    """
    cipher = _gostcipher.new(
        'kuznechik',
        bytearray(key),
        _gostcipher.MODE_ECB
    )
    return bytes(cipher.encrypt(bytearray(plaintext)))


# ─── Ручки ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["service"])
def health():
    return {"status": "ok"}


@app.get("/gpu", tags=["service"])
def gpu_info():
    try:
        return gw.gpu_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/test", tags=["service"])
def test_vectors():
    """
    Проверка корректности GPU реализации.
    Три теста: официальный вектор ГОСТ, сравнение GPU vs gostcrypto, CTR round-trip.
    """
    results = []

    # ── Тест 1: Официальный вектор ГОСТ Р 34.12-2015, Приложение А ───────────
    gost_key = bytes([
        0x88,0x99,0xAA,0xBB,0xCC,0xDD,0xEE,0xFF,
        0x00,0x11,0x22,0x33,0x44,0x55,0x66,0x77,
        0xFE,0xDC,0xBA,0x98,0x76,0x54,0x32,0x10,
        0x01,0x23,0x45,0x67,0x89,0xAB,0xCD,0xEF,
    ])
    gost_pt = bytes([
        0x11,0x22,0x33,0x44,0x55,0x66,0x77,0x00,
        0xFF,0xEE,0xDD,0xCC,0xBB,0xAA,0x99,0x88,
    ])
    gost_ct_expected = bytes([
        0x7F,0x67,0x9D,0x90,0xBE,0xBC,0x24,0x30,
        0x5A,0x46,0x8D,0x42,0xB9,0xD4,0xED,0xCD,
    ])
    try:
        ref_ecb = _gostcipher.new("kuznechik", bytearray(gost_key), _gostcipher.MODE_ECB)
        ref_ct  = bytes(ref_ecb.encrypt(bytearray(gost_pt)))
        t1_ok   = (ref_ct == gost_ct_expected)
        results.append({
            "name":        "Официальный вектор ГОСТ Р 34.12-2015 (Приложение А)",
            "passed":      t1_ok,
            "key":         gost_key.hex(" ").upper(),
            "plaintext":   gost_pt.hex(" ").upper(),
            "expected_ct": gost_ct_expected.hex(" ").upper(),
            "got_ct":      ref_ct.hex(" ").upper(),
            "note":        "ECB, эталонная библиотека gostcrypto",
        })
    except Exception as e:
        results.append({"name": "Официальный вектор ГОСТ", "passed": False, "error": str(e)})

    # ── Тест 2: GPU vs gostcrypto — побайтовое сравнение ─────────────────────
    # CTR блок 0: keystream = Encrypt(key, nonce||counter_0), ciphertext = keystream XOR plaintext
    # Воспроизводим это через ECB вручную (не зависим от CTR API gostcrypto)
    try:
        cmp_key   = secrets.token_bytes(32)
        cmp_nonce = secrets.token_bytes(8)
        cmp_plain = secrets.token_bytes(16)   # ровно один блок — нет вопросов с padding

        # GPU: шифруем через нашу реализацию
        t0     = time.perf_counter()
        gpu_ct = gw.encrypt_ctr(cmp_plain, cmp_key, cmp_nonce)
        gpu_ms = round((time.perf_counter() - t0) * 1000, 3)

        # Эталон: keystream блока 0 = ECB(key, nonce || 0x0000000000000000)
        # Это точное определение CTR из ГОСТ Р 34.13-2015
        counter_block = cmp_nonce + b"\x00" * 8
        t0      = time.perf_counter()
        ref_ecb = _gostcipher.new("kuznechik", bytearray(cmp_key), _gostcipher.MODE_ECB)
        keystream = bytes(ref_ecb.encrypt(bytearray(counter_block)))
        ref_ct    = bytes(a ^ b for a, b in zip(cmp_plain, keystream))
        ref_ms    = round((time.perf_counter() - t0) * 1000, 3)

        # gpu_ct содержит padding (17 байт для 16-байтного plaintext)
        # первые 16 байт — это шифртекст нашего блока
        t2_ok = (gpu_ct[:16] == ref_ct)
        results.append({
            "name":            "GPU vs gostcrypto — побайтовое сравнение",
            "passed":          t2_ok,
            "plaintext":       cmp_plain.hex(" ").upper(),
            "counter_block":   counter_block.hex(" ").upper(),
            "keystream":       keystream.hex(" ").upper(),
            "gpu_ciphertext":  gpu_ct[:16].hex(" ").upper(),
            "ref_ciphertext":  ref_ct.hex(" ").upper(),
            "gpu_elapsed_ms":  gpu_ms,
            "ref_elapsed_ms":  ref_ms,
            "note":            "CTR блок 0: Encrypt(key, nonce||0) XOR plaintext",
        })
    except Exception as e:
        results.append({"name": "GPU vs gostcrypto", "passed": False, "error": str(e)})

    # ── Тест 3: CTR round-trip ────────────────────────────────────────────────
    try:
        rt_key   = secrets.token_bytes(32)
        rt_nonce = secrets.token_bytes(8)
        rt_plain = secrets.token_bytes(1024)
        ct       = gw.encrypt_ctr(rt_plain, rt_key, rt_nonce)
        pt_back  = gw.decrypt_ctr(ct, rt_key, rt_nonce)
        results.append({
            "name":      "CTR round-trip: decrypt(encrypt(data)) == data",
            "passed":    pt_back == rt_plain,
            "data_size": len(rt_plain),
        })
    except Exception as e:
        results.append({"name": "CTR round-trip", "passed": False, "error": str(e)})

    all_passed = all(r["passed"] for r in results)
    response   = {"all_passed": all_passed, "tests": results, "gpu": gw.gpu_info()}

    if not all_passed:
        return JSONResponse(status_code=500, content=response)
    return response

@app.post("/encrypt", response_model=EncryptResponse, tags=["crypto"])
def encrypt(req: EncryptRequest):
    """
    Шифрует данные Кузнечиком в режиме CTR.

    - Данные принимаются в base64.
    - Если ключ или nonce не переданы — генерируются случайные.
    - Возвращает шифртекст, ключ и nonce в base64 — сохрани их для дешифрования.
    """
    data = _decode_b64(req.data_b64, "data_b64")
    if not data:
        raise HTTPException(status_code=400, detail="Данные пустые")

    if req.key_b64:
        key = _decode_b64(req.key_b64, "key_b64")
        if len(key) != 32:
            raise HTTPException(status_code=400, detail=f"Ключ должен быть 32 байта, получено {len(key)}")
    else:
        key = secrets.token_bytes(32)

    if req.nonce_b64:
        nonce = _decode_b64(req.nonce_b64, "nonce_b64")
        if len(nonce) != 8:
            raise HTTPException(status_code=400, detail=f"Nonce должен быть 8 байт, получено {len(nonce)}")
    else:
        nonce = secrets.token_bytes(8)

    try:
        t0      = time.perf_counter()
        ct      = gw.encrypt_ctr(data, key, nonce)
        elapsed = time.perf_counter() - t0
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка шифрования: {e}")

    size_mb    = len(data) / (1024 * 1024)
    throughput = size_mb / elapsed if elapsed > 0 else 0

    return EncryptResponse(
        ciphertext_b64  = _encode_b64(ct),
        key_b64         = _encode_b64(key),
        nonce_b64       = _encode_b64(nonce),
        original_size   = len(data),
        encrypted_size  = len(ct),
        elapsed_ms      = round(elapsed * 1000, 3),
        throughput_mb_s = round(throughput, 2),
    )


@app.post("/decrypt", response_model=DecryptResponse, tags=["crypto"])
def decrypt(req: DecryptRequest):
    """
    Дешифрует шифртекст Кузнечиком в режиме CTR.

    - Принимает шифртекст, ключ и nonce в base64.
    - CTR симметричен — тот же kernel что и для шифрования.
    """
    ct    = _decode_b64(req.ciphertext_b64, "ciphertext_b64")
    key   = _decode_b64(req.key_b64,        "key_b64")
    nonce = _decode_b64(req.nonce_b64,      "nonce_b64")

    if len(key) != 32:
        raise HTTPException(status_code=400, detail=f"Ключ должен быть 32 байта, получено {len(key)}")
    if len(nonce) != 8:
        raise HTTPException(status_code=400, detail=f"Nonce должен быть 8 байт, получено {len(nonce)}")
    if len(ct) % 16 != 0:
        raise HTTPException(status_code=400, detail=f"Длина шифртекста должна быть кратна 16, получено {len(ct)}")

    try:
        t0      = time.perf_counter()
        pt      = gw.decrypt_ctr(ct, key, nonce)
        elapsed = time.perf_counter() - t0
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Ошибка padding: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка дешифрования: {e}")

    size_mb    = len(ct) / (1024 * 1024)
    throughput = size_mb / elapsed if elapsed > 0 else 0

    return DecryptResponse(
        data_b64        = _encode_b64(pt),
        size            = len(pt),
        elapsed_ms      = round(elapsed * 1000, 3),
        throughput_mb_s = round(throughput, 2),
    )


@app.post("/benchmark", tags=["benchmark"])
def run_benchmark(req: BenchmarkRequest):
    """
    Замер throughput GPU шифрования.

    - data_mb: объём случайных данных (1–2048 МБ)
    - runs: количество прогонов (медиана как итог)

    ⚠️ Большие data_mb (512+) занимают несколько секунд.
    """
    try:
        return gw.benchmark(data_mb=req.data_mb, runs=req.runs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))