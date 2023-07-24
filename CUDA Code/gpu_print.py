from numba import cuda

# 查看GPU设备信息
print(cuda.gpus)
"""
一般使用CUDA_VISIBLE_DEVICES这个环境变量来选择某张卡。如选择0号GPU卡运行你的程序
CUDA_VISIBLE_DEVICES='0' python example.py
如果手头暂时没有GPU设备，Numba提供了一个模拟器，供用户学习和调试，只需要在命令行里添加一个环境变量
export NUMBA_ENABLE_CUDASIM=1
"""

def cpu_print():
    print("print by cpu.")

# 在GPU函数上添加@cuda.jit装饰符，表示该函数是一个在GPU设备上运行的函数，GPU函数又被称为核函数
@cuda.jit
def gpu_print():
    # GPU核函数
    print("print by gpu.")

def main():
    # 主函数调用GPU核函数时，需要添加如[1, 2]这样的执行配置，这个配置是在告知GPU以多大的并行粒度同时进行计算
    # gpu_print[1, 2]()表示同时开启2个线程并行地执行gpu_print函数，函数将被并行地执行2次
    gpu_print[1, 2]()
    # 调用cuda.synchronize()函数，等待GPU上的所有线程执行完毕
    cuda.synchronize()
    cpu_print()

if __name__ == "__main__":
    main()

# 执行代码: CUDA_VISIBLE_DEVICES='0' python gpu_print.py