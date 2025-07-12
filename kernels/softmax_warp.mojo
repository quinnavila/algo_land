from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx, lane_id
from memory import UnsafePointer
from gpu import barrier
from gpu.warp import sum as warp_sum, shuffle_xor, WARP_SIZE
from math import ceildiv, exp, inf
from random import random_float64
from time import perf_counter_ns
from layout.tensor_builder import LayoutTensorBuild as tb

alias dtype = DType.float32
alias BLOCK_SIZE = 256
alias ELEMENTS_PER_THREAD = 8
alias ELEMENTS_PER_BLOCK = BLOCK_SIZE * ELEMENTS_PER_THREAD


fn reduce_sum_kernel(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int,
) -> None:
    alias num_warps = BLOCK_SIZE // WARP_SIZE
    var shared = tb[dtype]().row_major[num_warps]().shared().alloc().ptr

    var tid = thread_idx.x
    var start_idx = block_idx.x * ELEMENTS_PER_BLOCK

    var acc: Float32 = 0.0
    for i in range(ELEMENTS_PER_THREAD):
        var global_idx = start_idx + tid + i * BLOCK_SIZE
        if global_idx < N:
            acc += input[global_idx]

    var warp_total = warp_sum(acc)

    if lane_id() == 0:
        shared[tid // WARP_SIZE] = warp_total

    barrier()

    if tid == 0:
        var block_total: Float32 = 0.0
        for i in range(num_warps):
            block_total += shared[i]
        output[block_idx.x] = block_total


fn reduce_max_kernel(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int,
) -> None:
    alias num_warps = BLOCK_SIZE // WARP_SIZE
    var shared = tb[dtype]().row_major[num_warps]().shared().alloc().ptr

    var tid = thread_idx.x
    var start_idx = block_idx.x * ELEMENTS_PER_BLOCK

    var acc: Float32 = -inf[dtype]()
    for i in range(ELEMENTS_PER_THREAD):
        var global_idx = start_idx + tid + i * BLOCK_SIZE
        if global_idx < N:
            acc = max(acc, input[global_idx])

    var offset = WARP_SIZE // 2
    while offset > 0:
        acc = max(acc, shuffle_xor(acc, offset))
        offset //= 2

    if lane_id() == 0:
        shared[tid // WARP_SIZE] = acc

    barrier()

    if tid == 0:
        var block_max = shared[0]
        for i in range(1, num_warps):
            block_max = max(block_max, shared[i])
        output[block_idx.x] = block_max


def reduce_sum(
    ctx: DeviceContext,
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int,
):
    if N <= 0:
        return
    if N == 1:
        ctx.enqueue_copy(output, input, 1)
        return

    var current_N = N
    var d_current_input = input

    while current_N > 1:
        var num_blocks = ceildiv(current_N, ELEMENTS_PER_BLOCK)
        var d_output_ptr: UnsafePointer[Float32]

        if num_blocks == 1:
            d_output_ptr = output
        else:
            var d_buffer = ctx.enqueue_create_buffer[dtype](num_blocks)
            d_output_ptr = d_buffer.unsafe_ptr()

        ctx.enqueue_function[reduce_sum_kernel](
            d_current_input,
            d_output_ptr,
            current_N,
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE,
        )
        d_current_input = d_output_ptr
        current_N = num_blocks


def reduce_max(
    ctx: DeviceContext,
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int,
):
    if N <= 0:
        return
    if N == 1:
        ctx.enqueue_copy(output, input, 1)
        return

    var current_N = N
    var d_current_input = input

    while current_N > 1:
        var num_blocks = ceildiv(current_N, ELEMENTS_PER_BLOCK)
        var d_output_ptr: UnsafePointer[Float32]

        if num_blocks == 1:
            d_output_ptr = output
        else:
            var d_buffer = ctx.enqueue_create_buffer[dtype](num_blocks)
            d_output_ptr = d_buffer.unsafe_ptr()

        ctx.enqueue_function[reduce_max_kernel](
            d_current_input,
            d_output_ptr,
            current_N,
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE,
        )
        d_current_input = d_output_ptr
        current_N = num_blocks


fn exp_reduce_to_block_sum_kernel(
    input: UnsafePointer[Float32],
    max_val_ptr: UnsafePointer[Float32],
    block_sums: UnsafePointer[Float32],
    N: Int,
) -> None:
    alias num_warps = BLOCK_SIZE // WARP_SIZE
    var shared = tb[dtype]().row_major[num_warps]().shared().alloc().ptr

    var tid = thread_idx.x
    var start_idx = block_idx.x * ELEMENTS_PER_BLOCK
    var max_val = max_val_ptr[0]

    var acc: Float32 = 0.0
    for i in range(ELEMENTS_PER_THREAD):
        var global_idx = start_idx + tid + i * BLOCK_SIZE
        if global_idx < N:
            acc += exp(input[global_idx] - max_val)

    var warp_total = warp_sum(acc)

    if lane_id() == 0:
        shared[tid // WARP_SIZE] = warp_total

    barrier()

    if tid == 0:
        var block_total: Float32 = 0.0
        for i in range(num_warps):
            block_total += shared[i]
        block_sums[block_idx.x] = block_total


fn normalize_fused_kernel(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    max_val_ptr: UnsafePointer[Float32],
    global_sum_ptr: UnsafePointer[Float32],
    N: Int,
) -> None:
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if idx < N:
        var max_val = max_val_ptr[0]
        var global_sum = global_sum_ptr[0]
        output[idx] = exp(input[idx] - max_val) / global_sum


def main():
    alias N = 500_000

    print("Running Improved Softmax with Idiomatic Allocation for N =", N)

    with DeviceContext() as ctx:
        var input_buffer = ctx.enqueue_create_buffer[dtype](N)
        var output_buffer = ctx.enqueue_create_buffer[dtype](N)

        with input_buffer.map_to_host() as h_input:
            for i in range(N):
                h_input[i] = Float32(random_float64(-1.0, 1.0))

        ctx.synchronize()
        var start_time_ns = perf_counter_ns()

        var max_buf = ctx.enqueue_create_buffer[dtype](1)
        reduce_max(ctx, input_buffer.unsafe_ptr(), max_buf.unsafe_ptr(), N)

        var sum_grid_size = ceildiv(N, ELEMENTS_PER_BLOCK)
        var block_sums_buf = ctx.enqueue_create_buffer[dtype](sum_grid_size)
        ctx.enqueue_function[exp_reduce_to_block_sum_kernel](
            input_buffer.unsafe_ptr(),
            max_buf.unsafe_ptr(),
            block_sums_buf.unsafe_ptr(),
            N,
            grid_dim=sum_grid_size,
            block_dim=BLOCK_SIZE,
        )

        var sum_buf = ctx.enqueue_create_buffer[dtype](1)
        reduce_sum(
            ctx,
            block_sums_buf.unsafe_ptr(),
            sum_buf.unsafe_ptr(),
            sum_grid_size,
        )

        var normalize_grid_size = ceildiv(N, BLOCK_SIZE)
        ctx.enqueue_function[normalize_fused_kernel](
            input_buffer.unsafe_ptr(),
            output_buffer.unsafe_ptr(),
            max_buf.unsafe_ptr(),
            sum_buf.unsafe_ptr(),
            N,
            grid_dim=normalize_grid_size,
            block_dim=BLOCK_SIZE,
        )

        ctx.synchronize()
        var end_time_ns = perf_counter_ns()
        var duration_ms = (end_time_ns - start_time_ns) / 1_000_000

        var total_sum: Float32 = 0.0
        with output_buffer.map_to_host() as h_output:
            for i in range(N):
                total_sum += h_output[i]

        print("Softmax computation complete.")
        print("Execution time:", duration_ms, "ms")
        print("Sum of all elements in the output:", total_sum)

        if total_sum > 0.999 and total_sum < 1.001:
            print("Verification PASSED.")
        else:
            print("Verification FAILED.")
