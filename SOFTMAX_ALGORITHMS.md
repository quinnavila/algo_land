
---

### GPU Softmax Algorithms: A Comparison

Choosing a GPU softmax algorithm balances simplicity, performance, and memory. This guide covers four key strategies.

### At a Glance

*   **1. Standard 3-Pass:**
    *   **How:** Three passes: find max, compute `exp(x - max)` and sum, then normalize.
    *   **Memory:** High (needs a large intermediate buffer).
    *   **Best For:** Small vectors.

*   **2. 2-Pass Online Normalizer:**
    *   **How:** Two passes: find final max and sum in one pass, normalize in the second.
    *   **Memory:** Low (no large buffer, but reads input twice).
    *   **Best For:** Medium vectors.

*   **3. Tiled Online Softmax:**
    *   **How:** Processes input in tiles, tracks a running max/sum, and rescales past results as needed.
    *   **Memory:** Low (only needs a small state buffer).
    *   **Best For:** Very large vectors (the basis for FlashAttention).

*   **4. Fused Online Softmax:**
    *   **How:** Fuses the Tiled Online logic with the preceding operation (e.g., MatMul).
    *   **Memory:** Minimal (computes and consumes tiles in fast on-chip SRAM).
    *   **Best For:** Maximum performance in attention mechanisms.

---

### Detailed Breakdown

#### 1. Standard 3-Pass Parallel

The most direct parallel method.
*   **Pass 1:** Find global `max`.
*   **Pass 2:** Compute `exp(x - max)`, store in a new buffer, and find its `sum`.
*   **Pass 3:** Divide the stored `exp` values by the `sum`.

#### 2. 2-Pass with Online Normalizer

Avoids the large temporary buffer by reading the input twice.
*   **Pass 1:** Computes the final `max` and `sum` in one pass using a special formula.
*   **Pass 2:** Re-reads input to compute `exp(x - final_max) / final_sum`.

#### 3. Tiled Online Softmax

Handles inputs too large for fast memory by processing in sequential blocks.
*   Maintains a running `max` and `sum`.
*   The key is **rescaling** previous results if a new tile contains a larger max.

#### 4. Fused Online Softmax (True FlashAttention)

Fuses the Tiled Online logic with the preceding operation (e.g., matrix multiplication).
*   A single kernel computes a score tile (`QK^T`) and immediately processes it.
*   The score tile lives only in fast on-chip SRAM, avoiding slow global DRAM.

### Summary Table

| Approach | Memory Use | Key Trait | Best For |
| :--- | :--- | :--- | :--- |
| **Standard (3-Pass)** | High | Simple, memory-heavy | Small vectors |
| **2-Pass Online** | Low | Efficient, two passes | Medium vectors |
| **Tiled Online** | Low | Tile-based, scalable | Large vectors |
| **Fused Online** | Minimal | Fused, avoids global memory writes | High-performance attention |