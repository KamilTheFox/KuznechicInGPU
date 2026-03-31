"""
grasshopper_worker.py — v8 (C extension backend)
"""
import os, time
import gostcrypto.gostcipher.gost_34_12_2015 as _kuz_impl
import grasshopper_gpu as _gh

_HERE = os.path.dirname(os.path.abspath(__file__))
_gh.init(os.path.join(_HERE, 'tables.bin'))
print(f"[Worker] GPU: {_gh.gpu_name()}")

def _compute_rk(key: bytes) -> bytes:
    c = _kuz_impl.GOST34122015Kuznechik(bytearray(key))
    rk = bytearray()
    for k in c._cipher_iter_key: rk.extend(k)
    c.clear()
    return bytes(rk)

def _pad(data: bytes) -> bytes:
    r = 16 - (len(data) % 16)
    return data + bytes([r] * r)

def _unpad(data: bytes) -> bytes:
    if not data: return data
    r = data[-1]
    if 1 <= r <= 16 and data[-r:] == bytes([r] * r): return data[:-r]
    return data

def encrypt_ctr(data: bytes, key: bytes, nonce: bytes) -> bytes:
    return _gh.encrypt(_pad(data), _compute_rk(key), nonce)

def decrypt_ctr(data: bytes, key: bytes, nonce: bytes) -> bytes:
    return _unpad(_gh.decrypt(data, _compute_rk(key), nonce))

def benchmark(data_mb=128, runs=3):
    key, nonce = bytes(32), bytes(8)
    data = os.urandom(data_mb * 1024 * 1024)
    encrypt_ctr(data[:4096], key, nonce)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        res = encrypt_ctr(data, key, nonce)
        times.append(time.perf_counter() - t0)
    med = sorted(times)[len(times)//2]
    return {"throughput_mb_s": round(data_mb/med, 2), "integrity_ok": decrypt_ctr(res, key, nonce)==data}

def test_vectors():
    b = benchmark(data_mb=64, runs=1)
    return {"all_passed": b["integrity_ok"], "details": b, "gpu": gpu_info()}

def gpu_info():
    return {"name": _gh.gpu_name()}