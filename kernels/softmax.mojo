from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx
from gpu import barrier
from layout import Layout, LayoutTensor
from layout.tensor_builder import LayoutTensorBuild as tb
from memory import UnsafePointer
from math import ceildiv, exp, inf
from random import random_float64
from time import perf_counter_ns

alias dtype = DType.float32
alias BLOCK_SIZE = 256
alias ELEMENTS_PER_THREAD = 8
alias ELEMENTS_PER_BLOCK = BLOCK_SIZE * ELEMENTS_PER_THREAD

fn reduction_kernel[op: fn(Float32, Float32) -> Float32](
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int,
    initial_value: Float32,
) -> None:
    var shared = tb[dtype]().row_major[BLOCK_SIZE]().shared().alloc()
    var tid = thread_idx.x
    var start_idx = block_idx.x * ELEMENTS_PER_BLOCK

    var acc = initial_value
    # review for simd
    for i in range(ELEMENTS_PER_THREAD):
        var global_idx = start_idx + tid + i * BLOCK_SIZE
        if Int32(global_idx) < N:
            acc = op(acc, input[global_idx])

    shared[tid] = acc
    barrier()

    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if tid < stride:
            shared[tid] = op(shared[tid][0], shared[tid + stride][0])
        barrier()
        stride //= 2

    if tid == 0:
        output[block_idx.x] = shared[0][0]

def reduce[op: fn(Float32, Float32) -> Float32](
    ctx: DeviceContext,
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    N: Int,
    initial_value: Float32,
):
    if N <= 0: return
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
            var d_intermediate_buffer = ctx.enqueue_create_buffer[dtype](num_blocks)
            d_output_ptr = d_intermediate_buffer.unsafe_ptr()

        ctx.enqueue_function[reduction_kernel[op]](
            d_current_input, d_output_ptr, current_N, initial_value,
            grid_dim=num_blocks, block_dim=BLOCK_SIZE
        )
        d_current_input = d_output_ptr
        current_N = num_blocks

fn exp_and_block_sum_kernel(
    input: UnsafePointer[Float32],
    output: UnsafePointer[Float32],
    max_val_ptr: UnsafePointer[Float32],
    block_sums_ptr: UnsafePointer[Float32],
    N: Int,
) -> None:
    var idx = block_idx.x * block_dim.x + thread_idx.x
    var max_val = max_val_ptr[0]

    var exp_val: Float32 = 0.0
    if Int32(idx) < N:
        exp_val = exp(input[idx] - max_val)
        output[idx] = exp_val

    var shared = tb[dtype]().row_major[BLOCK_SIZE]().shared().alloc()
    shared[thread_idx.x] = exp_val if Int32(idx) < N else 0.0
    barrier()

    var stride = BLOCK_SIZE // 2
    while stride > 0:
        if thread_idx.x < stride:
            shared[thread_idx.x] += shared[thread_idx.x + stride]
        barrier()
        stride //= 2

    if thread_idx.x == 0:
        block_sums_ptr[block_idx.x] = shared[0][0]

fn normalize_kernel(
    output: UnsafePointer[Float32],
    global_sum_ptr: UnsafePointer[Float32],
    N: Int,
) -> None:
    var idx = block_idx.x * block_dim.x + thread_idx.x
    if Int32(idx) < N:
        output[idx] /= global_sum_ptr[0]

def main():
    alias N = 500_000
    alias data_layout = Layout.row_major(N)

    print("Running Idiomatic Softmax for N =", N)

    with DeviceContext() as ctx:
        var input_buffer = ctx.enqueue_create_buffer[dtype](N)
        var output_buffer = ctx.enqueue_create_buffer[dtype](N)
        
        with input_buffer.map_to_host() as h_input:
            for i in range(N):
                h_input[i] = Float32(random_float64(-1.0, 1.0))

        var input_tensor = LayoutTensor[dtype, data_layout](input_buffer)
        var output_tensor = LayoutTensor[dtype, data_layout](output_buffer)
        
        fn add(a: Float32, b: Float32) -> Float32: return a + b
        fn max_op(a: Float32, b: Float32) -> Float32: 
            if a > b: return a
            else: return b

        ctx.synchronize()
        var start_time_ns = perf_counter_ns()

        var max_buf = ctx.enqueue_create_buffer[dtype](1)
        reduce[max_op](ctx, input_tensor.ptr, max_buf.unsafe_ptr(), N, -inf[dtype]())
        
        var elementwise_grid_size = ceildiv(N, BLOCK_SIZE)
        var block_sums_buf = ctx.enqueue_create_buffer[dtype](elementwise_grid_size)
        
        ctx.enqueue_function[exp_and_block_sum_kernel](
            input_tensor.ptr, output_tensor.ptr, max_buf.unsafe_ptr(), block_sums_buf.unsafe_ptr(), N,
            grid_dim=elementwise_grid_size,
            block_dim=BLOCK_SIZE
        )
        
        var sum_buf = ctx.enqueue_create_buffer[dtype](1)
        reduce[add](ctx, block_sums_buf.unsafe_ptr(), sum_buf.unsafe_ptr(), elementwise_grid_size, 0.0)
        
        ctx.enqueue_function[normalize_kernel](
            output_tensor.ptr, sum_buf.unsafe_ptr(), N,
            grid_dim=elementwise_grid_size,
            block_dim=BLOCK_SIZE
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