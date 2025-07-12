from gpu.host import DeviceContext
from gpu.id import block_dim, block_idx, thread_idx, lane_id
from gpu import barrier
from gpu.warp import sum as warp_sum, shuffle_xor
from layout import Layout, LayoutTensor, UNKNOWN_VALUE, RuntimeLayout
from layout.tensor_builder import LayoutTensorBuild as tb
from math import ceildiv, inf
from random import random_float64
from time import perf_counter_ns
from testing import assert_almost_equal
from memory import UnsafePointer
from utils.index import Index

alias dtype = DType.float32
alias BLOCK_SIZE = 256
alias ELEMENTS_PER_THREAD = 8
alias ELEMENTS_PER_BLOCK = BLOCK_SIZE * ELEMENTS_PER_THREAD
alias NUM_WARPS = BLOCK_SIZE // 32


fn reduction_sum_warp_kernel(
    input: LayoutTensor[mut=False, dtype, Layout.row_major(UNKNOWN_VALUE, 1)],
    output: LayoutTensor[mut=True, dtype, Layout.row_major(UNKNOWN_VALUE, 1)],
    initial_value: Float32,
) -> None:
    var shared = tb[dtype]().row_major[NUM_WARPS, 1]().shared().alloc()
    var tid = thread_idx.x
    var start_idx = block_idx.x * ELEMENTS_PER_BLOCK

    var acc = initial_value
    for i in range(ELEMENTS_PER_THREAD):
        var global_idx = start_idx + tid + i * BLOCK_SIZE
        if global_idx < input.runtime_layout.dim(0):
            var val = input[global_idx, 0][0]
            acc += val

    var warp_total = warp_sum(acc)

    if lane_id() == 0:
        shared[tid // 32, 0] = warp_total

    barrier()

    if tid == 0:
        var block_total: Float32 = 0.0
        for i in range(NUM_WARPS):
            block_total += shared[i, 0][0]
        output[0, block_idx.x] = block_total


fn reduction_max_warp_kernel(
    input: LayoutTensor[mut=False, dtype, Layout.row_major(UNKNOWN_VALUE, 1)],
    output: LayoutTensor[mut=True, dtype, Layout.row_major(UNKNOWN_VALUE, 1)],
    initial_value: Float32,
) -> None:
    var shared = tb[dtype]().row_major[NUM_WARPS, 1]().shared().alloc()
    var tid = thread_idx.x
    var start_idx = block_idx.x * ELEMENTS_PER_BLOCK

    var acc = initial_value
    for i in range(ELEMENTS_PER_THREAD):
        var global_idx = start_idx + tid + i * BLOCK_SIZE
        if global_idx < input.runtime_layout.dim(0):
            var val = input[global_idx, 0][0]
            acc = max(acc, val)

    var offset = 16
    while offset > 0:
        acc = max(acc, shuffle_xor(acc, offset))
        offset //= 2

    if lane_id() == 0:
        shared[tid // 32, 0] = acc

    barrier()

    if tid == 0:
        var block_max = -inf[dtype]()
        for i in range(NUM_WARPS):
            block_max = max(block_max, shared[i, 0][0])
        output[0, block_idx.x] = block_max


alias KernelType = fn (
    LayoutTensor[mut=False, dtype, Layout.row_major(UNKNOWN_VALUE, 1)],
    LayoutTensor[mut=True, dtype, Layout.row_major(UNKNOWN_VALUE, 1)],
    Float32,
) -> None


def reduce[
    kernel: KernelType
](
    ctx: DeviceContext,
    input_ptr: UnsafePointer[Float32],
    N: Int,
    initial_value: Float32,
) -> Float32:
    if N <= 0:
        return initial_value
    if N == 1:
        var result_buffer = ctx.enqueue_create_buffer[dtype](1)
        ctx.enqueue_copy(result_buffer.unsafe_ptr(), input_ptr, 1)
        ctx.synchronize()
        with result_buffer.map_to_host() as h:
            return h[0]

    alias static_layout = Layout.row_major(UNKNOWN_VALUE, 1)
    var result_buffer = ctx.enqueue_create_buffer[dtype](1)
    var current_N = N
    var d_current = input_ptr

    while current_N > 1:
        var num_blocks = ceildiv(current_N, ELEMENTS_PER_BLOCK)
        var d_next: UnsafePointer[Float32]

        if num_blocks == 1:
            d_next = result_buffer.unsafe_ptr()
        else:
            var temp_buf = ctx.enqueue_create_buffer[dtype](num_blocks)
            d_next = temp_buf.unsafe_ptr()

        var runtime_in_layout = RuntimeLayout[static_layout].row_major(
            Index(current_N, 1)
        )
        var runtime_out_layout = RuntimeLayout[static_layout].row_major(
            Index(num_blocks, 1)
        )

        var input_tensor = LayoutTensor[mut=False, dtype, static_layout](
            d_current, runtime_in_layout
        )
        var output_tensor = LayoutTensor[mut=True, dtype, static_layout](
            d_next, runtime_out_layout
        )

        ctx.enqueue_function[kernel](
            input_tensor,
            output_tensor,
            initial_value,
            grid_dim=num_blocks,
            block_dim=BLOCK_SIZE,
        )

        d_current = d_next
        current_N = num_blocks

    ctx.synchronize()
    var final_result: Float32
    with result_buffer.map_to_host() as h:
        final_result = h[0]
    return final_result


def main():
    alias N_initial = 1_000_000

    print("Step 1: Building a Reusable, Warp-Based Reduction Function")
    print("Array size (N):", N_initial)
    print("-" * 60)

    with DeviceContext() as ctx:
        var input_buffer = ctx.enqueue_create_buffer[dtype](N_initial)
        var h_input_data = List[Float32]()
        h_input_data.reserve(N_initial)

        with input_buffer.map_to_host() as h_input:
            for i in range(N_initial):
                var val = Float32(random_float64(-10.0, 10.0))
                h_input[i] = val
                h_input_data.append(val)

        print("Test Case 1: Sum Reduction (Warp-Based)")
        var start_time_sum = perf_counter_ns()
        var gpu_sum = reduce[reduction_sum_warp_kernel](
            ctx, input_buffer.unsafe_ptr(), N_initial, 0.0
        )
        var end_time_sum = perf_counter_ns()

        var cpu_sum: Float32 = 0.0
        for i in range(N_initial):
            cpu_sum += h_input_data[i]

        print("GPU Sum:", gpu_sum)
        print("CPU Sum:", cpu_sum)
        print("Time:", (end_time_sum - start_time_sum) / 1_000_000, "ms")
        assert_almost_equal(gpu_sum, cpu_sum, rtol=1e-3)
        print("Verification PASSED.\n")

        print("Test Case 2: Max Reduction (Warp-Based)")
        var start_time_max = perf_counter_ns()
        var gpu_max = reduce[reduction_max_warp_kernel](
            ctx, input_buffer.unsafe_ptr(), N_initial, -inf[dtype]()
        )
        var end_time_max = perf_counter_ns()

        var cpu_max: Float32 = -inf[dtype]()
        for i in range(N_initial):
            if h_input_data[i] > cpu_max:
                cpu_max = h_input_data[i]

        print("GPU Max:", gpu_max)
        print("CPU Max:", cpu_max)
        print("Time:", (end_time_max - start_time_max) / 1_000_000, "ms")
        assert_almost_equal(gpu_max, cpu_max)
        print("Verification PASSED.")
