from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from gpu import barrier
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from math import ceildiv, exp, inf
from random import random_float64
from time import perf_counter_ns
from testing import assert_almost_equal

alias dtype = DType.float32
alias ROWS = 128
alias COLS = 1024
alias BLOCK_SIZE = 256
alias ELEMENTS_PER_THREAD = ceildiv(COLS, BLOCK_SIZE)

alias static_layout = Layout.row_major(ROWS, COLS)


fn row_softmax_kernel(
    input: LayoutTensor[mut=False, dtype, static_layout],
    output: LayoutTensor[mut=True, dtype, static_layout],
) -> None:
    var shared_max = tb[dtype]().row_major[BLOCK_SIZE]().shared().alloc()
    var shared_sum = tb[dtype]().row_major[BLOCK_SIZE]().shared().alloc()

    var row = block_idx.x
    var tid = thread_idx.x

    var thread_max = -inf[dtype]()
    for i in range(ELEMENTS_PER_THREAD):
        var col = tid + i * BLOCK_SIZE
        if col < COLS:
            thread_max = max(thread_max, input[row, col][0])

    shared_max[tid] = thread_max
    barrier()

    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shared_max[tid] = max(shared_max[tid], shared_max[tid + stride])
        barrier()
        stride //= 2

    var row_max = shared_max[0]
    barrier()

    var thread_sum: Float32 = 0.0
    for i in range(ELEMENTS_PER_THREAD):
        var col = tid + i * BLOCK_SIZE
        if col < COLS:
            thread_sum += exp(input[row, col][0] - row_max[0])

    shared_sum[tid] = thread_sum
    barrier()

    stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shared_sum[tid] += shared_sum[tid + stride]
        barrier()
        stride //= 2

    var row_sum = shared_sum[0]
    barrier()

    for i in range(ELEMENTS_PER_THREAD):
        var col = tid + i * BLOCK_SIZE
        if col < COLS:
            var exp_val = exp(input[row, col][0] - row_max[0])
            output[row, col] = exp_val / row_sum[0]


def main():
    print("Batched Row-wise Softmax")
    print("Matrix size:", ROWS, "x", COLS)
    print("Threads per block:", BLOCK_SIZE)
    print("Elements per thread:", ELEMENTS_PER_THREAD)

    with DeviceContext() as ctx:
        var input_buffer = ctx.enqueue_create_buffer[dtype](ROWS * COLS)
        var output_buffer = ctx.enqueue_create_buffer[dtype](ROWS * COLS)

        with input_buffer.map_to_host() as h_input:
            for i in range(ROWS * COLS):
                h_input[i] = Float32(random_float64(-1.0, 1.0))

        var input_tensor = LayoutTensor[mut=False, dtype, static_layout](
            input_buffer.unsafe_ptr()
        )
        var output_tensor = LayoutTensor[mut=True, dtype, static_layout](
            output_buffer.unsafe_ptr()
        )

        print("\nLaunching row_softmax_kernel...")
        var start_time = perf_counter_ns()
        ctx.enqueue_function[row_softmax_kernel](
            input_tensor,
            output_tensor,
            grid_dim=(ROWS, 1),
            block_dim=(BLOCK_SIZE, 1),
        )
        ctx.synchronize()
        var end_time = perf_counter_ns()
        print(
            "Kernel execution time:", (end_time - start_time) / 1_000_000, "ms"
        )

        print("\nVerifying results...")
        with output_buffer.map_to_host() as h_output:
            var gpu_tensor = LayoutTensor[mut=False, dtype, static_layout](
                h_output.unsafe_ptr()
            )
            for r in range(ROWS):
                var row_sum: Float32 = 0.0
                for c in range(COLS):
                    row_sum += gpu_tensor[r, c][0]
                assert_almost_equal(row_sum, 1.0, rtol=1e-3)

        print("Verification PASSED: All rows sum to 1.0.")
