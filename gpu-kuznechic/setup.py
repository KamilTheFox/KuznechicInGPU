"""
setup.py — сборка CPython extension grasshopper_gpu.so
"""

import subprocess
import sys
import os
import sysconfig
import glob

def find_python_lib(libdir, ver):
    """
    В Ubuntu 22.04 libpython3.11 бывает:
      libpython3.11.so
      libpython3.11.so.1
      libpython3.11.so.1.0
      libpython3.11-dev -> устанавливается отдельно
    Ищем реальный файл и возвращаем точное имя для -l.
    """
    # Сначала ищем в стандартных местах
    search_dirs = [
        libdir,
        f"/usr/lib/python{ver}",
        f"/usr/lib/x86_64-linux-gnu",
        f"/usr/lib",
        f"/usr/local/lib",
    ]
    patterns = [
        f"libpython{ver}.so",
        f"libpython{ver}.so.*",
        f"libpython{ver}m.so",
        f"libpython{ver}m.so.*",
    ]
    for d in search_dirs:
        for pat in patterns:
            matches = glob.glob(os.path.join(d, pat))
            if matches:
                # Берём первый найденный, извлекаем имя без lib и .so
                fname = os.path.basename(matches[0])
                # fname = "libpython3.11.so.1.0" -> "python3.11.so.1.0"
                # Нам нужно имя для -l: убираем "lib" и всё после первого .so
                name = fname[3:]  # убираем "lib"
                # Для -l нужно имя до расширения
                if ".so" in name:
                    name = name[:name.index(".so")]
                return d, name
    return libdir, f"python{ver}"

def build():
    py_include = sysconfig.get_path('include')
    py_libdir  = sysconfig.get_config_var('LIBDIR') or '/usr/lib/x86_64-linux-gnu'
    py_ver     = f"{sys.version_info.major}.{sys.version_info.minor}"
    ext_suffix = sysconfig.get_config_var('EXT_SUFFIX')
    output     = f"grasshopper_gpu{ext_suffix}"

    found_dir, lib_name = find_python_lib(py_libdir, py_ver)
    print(f"[build] Python include: {py_include}")
    print(f"[build] Python lib dir: {found_dir}")
    print(f"[build] Python lib name: -{lib_name}")

    # CPython extensions не обязаны линковаться с libpython —
    # символы резолвятся из интерпретатора в рантайме.
    # Используем -Xlinker --allow-shlib-undefined чтобы не требовать libpython при сборке.
    cmd = [
        "nvcc",
        "grasshopper_module.cu",
        "-O3",
        f"-arch=sm_86",
        "--use_fast_math",
        "-std=c++11",
        "-Xcompiler", "-fPIC",
        "--shared",
        f"-I{py_include}",
        f"-L{found_dir}",
        # Не линкуем libpython явно — Python extension резолвит символы из
        # родительского процесса (интерпретатора) в рантайме
        "-Xlinker", "--allow-shlib-undefined",
        "-lcudart",
        "-o", output,
    ]

    print(f"[build] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        # Fallback: пробуем найти и слинковать явно
        print("[build] Retrying with explicit python link...")
        cmd2 = [
            "nvcc",
            "grasshopper_module.cu",
            "-O3",
            f"-arch=sm_86",
            "--use_fast_math",
            "-std=c++11",
            "-Xcompiler", "-fPIC",
            "--shared",
            f"-I{py_include}",
            f"-L{found_dir}",
            f"-l{lib_name}",
            "-lcudart",
            "-o", output,
        ]
        print(f"[build] Running: {' '.join(cmd2)}")
        result2 = subprocess.run(cmd2)
        if result2.returncode != 0:
            print(f"[build] FAILED")
            sys.exit(1)

    print(f"[build] OK → {output}")
    return output

if __name__ == "__main__":
    build()