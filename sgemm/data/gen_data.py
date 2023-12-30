import os
import sys
import numpy as np


def gen_data(m: int, k:int, n: int):
    x = np.random.randn(m, k).astype(np.float32)
    y = np.random.randn(k, n).astype(np.float32)
    z = np.matmul(x, y)
    return x, y, z


def save_bin(x: np.array, y:np.array, z:np.array, saved_file: str):
    x.tofile(f"{saved_file}_x.bin")
    y.tofile(f"{saved_file}_y.bin")
    z.tofile(f"{saved_file}_z.bin")


def gen_and_save(m: int, k:int, n: int, saved_dir: str):
    x, y, z = gen_data(m, k, n)
    save_bin(x, y, z, f"{saved_dir}/sgemm_m_{m}_k_{k}_n_{n}")
    

if __name__ == '__main__':
    saved_dir = sys.path[0]
    gen_and_save(512, 512, 512, saved_dir)
    gen_and_save(1024, 1024, 1024, saved_dir)
    gen_and_save(2048, 2048, 2048, saved_dir)
    gen_and_save(4096, 4096, 4096, saved_dir)