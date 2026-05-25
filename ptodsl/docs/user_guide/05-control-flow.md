# 5. Control Flow

PTODSL uses a **tracing** compilation model. When you call `kernel.compile(...)`, PTODSL executes your Python function body once to record every PTO instruction — this pass is called *tracing*. The recorded program is then lowered and optimized into device code. Once compiled, launching the kernel runs the already-built device code directly on the NPU.

This has one critical implication for how you write loops and branches:

- **Python native `for`/`if`** runs at trace time. A `for i in range(4)` loop gets unrolled — the device code contains four copies of the body, not a loop instruction. An `if` condition is evaluated at trace time, and only the taken branch is recorded.
- **`pto.for_` / `pto.if_`** produce device-side control flow. The loop bound or branch condition can be a runtime value, and the hardware will execute the loop or take the branch dynamically.

**Simple rule: Python control flow = trace time (compile-time). `pto.*` control flow = device-side (runtime).**

## 5.1 Python native `for` — trace-time unrolling

When you write a plain Python `for` loop inside a kernel body, Python executes it immediately during tracing. Each iteration records its instructions separately, so the device code gets a linear sequence with the body repeated:

```python
@pto.jit(target="a5")
def unrolled_kernel(A, O, *, N: pto.constexpr):
    a_view = pto.make_tensor_view(A, shape=[N], strides=A.strides)
    o_view = pto.make_tensor_view(O, shape=[N], strides=O.strides)

    # N is constexpr, so range(N) is known at trace time.
    # The loop unrolls: the device gets N copies of the body.
    for i in range(N):
        a_part = pto.partition_view(a_view, offsets=[i], sizes=[1])
        o_part = pto.partition_view(o_view, offsets=[i], sizes=[1])
        a_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)
        o_tile = pto.alloc_tile(shape=[1], dtype=pto.f32)
        pto.tile.load(a_part, a_tile)
        pto.tile.add(a_tile, a_tile, o_tile)
        pto.tile.store(o_tile, o_part)
```

This works when the loop bound is a compile-time constant (like a `constexpr` parameter). But if `N` comes from a tensor shape and varies per launch, `range(N)` would trace a different number of iterations each time — you would get a cache miss and recompilation on every new value. For dynamic bounds, use `pto.for_`.

## 5.2 `pto.for_` — device-side loops

`pto.for_` records a structured loop that executes on the device. Its bound can be any expression involving runtime values (tensor shapes, scalar computations, block indices), and the compiler may optimize it further — unrolling when the bound is known at compile time, or keeping it as a runtime loop otherwise.

### Basic form

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"control_flow.basic_for","symbol":"control_flow_basic_for_probe","compile":{"BLOCK":8}} -->
```python
with pto.for_(start, stop, step=step) as iv:
    pto.tile.load(pto.partition_view(a_view, offsets=[iv, 0], sizes=[1, cols]), tile)
```

- `start`, `stop`, `step` are PTO scalar expressions. They are evaluated on the device.
- The loop body executes `(stop - start + step - 1) // step` times.
- Use with `step=1` unless you need a strided iteration.

Compare the two approaches:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"control_flow.compare_loops","symbol":"control_flow_compare_loops_probe","compile":{"BLOCK":8}} -->
```python
# Trace-time unrolling — BLOCK must be constexpr
for i in range(BLOCK):
    pto.tile.load(pto.partition_view(a_view, offsets=[0, 0], sizes=[1, cols]), tile)

# Device-side loop — num_blocks can be dynamic
with pto.for_(0, num_blocks, step=1) as i:
    pto.tile.load(pto.partition_view(a_view, offsets=[i, 0], sizes=[1, cols]), tile)
```

### Nested loops

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"control_flow.nested_loops","symbol":"control_flow_nested_loops_probe","compile":{"BLOCK":8}} -->
```python
with pto.for_(0, rows, step=1) as r:
    with pto.for_(0, cols, step=1) as c:
        val = scalar.load(tile[r, c])
```

Both loops execute on the device. The outer loop bound `rows` and inner loop bound `cols` can be runtime values.

### Loop with carry state

When a loop needs to propagate state from one iteration to the next, use the `.carry(...)` method. This is the PTODSL equivalent of a loop that accumulates or updates variables across iterations. The following self-contained kernel is the smallest compileable carry example used by the docs-as-test harness:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"carry_loop_probe","compile":{"BLOCK":128}} -->
```python
@pto.jit(target="a5")
def carry_loop_probe(*, BLOCK: pto.constexpr = 128):
    m_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    l_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    m_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    l_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    m_prev.fill(0.0)
    l_prev.fill(0.0)
    o_prev.fill(0.0)

    kv_loop = pto.for_(0, 4, step=1).carry(m=m_prev, l=l_prev, o=o_prev)
    with kv_loop:
        kv_loop.m.fill(1.0)
        kv_loop.l.fill(2.0)
        kv_loop.o.fill(3.0)
        kv_loop.update(m=m_next, l=l_next, o=o_next)

    final_o = kv_loop.final("o")
    final_o.fill(4.0)
```

`.carry(name=initial_value)` declares named state variables that are passed from one iteration to the next. Inside the loop body, access the current value with `loop.name`. At the end of the body, call `loop.update(name=new_value)` to set what the next iteration receives. After the loop exits, `loop.final("name")` retrieves the value from the last iteration.

This pattern is central to algorithms like online softmax, where each KV block updates running statistics (row max, sum, output accumulator). The ping-pong tile pattern — allocating two tiles and swapping them each iteration — is the idiomatic way to manage this state:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"control_flow.carry_pingpong","symbol":"control_flow_carry_pingpong_probe","compile":{"Br":16,"num_blocks":4}} -->
```python
# Allocate ping-pong state tiles
m_prev = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
m_next = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
l_prev = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")
l_next = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")

# Initialize prev tiles
m_prev.fill(float("-inf"))
l_prev.fill(0.0)

loop = pto.for_(0, num_blocks, step=1).carry(m=m_prev, l=l_prev)
with loop:
    m_cur = loop.m
    l_cur = loop.l

    m_next.fill(1.0)
    l_next.fill(2.0)

    loop.update(m=m_next, l=l_next)
```

### Chunked inner loop with carry (tail handling)

For SIMD kernels that process data in vector-width chunks, use a carry loop to track the remaining element count across column iterations:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"tail.chunked_inner_loop","symbol":"tail_chunked_inner_loop_probe","compile":{"BLOCK":128}} -->
```python
VEC = pto.elements_per_vreg(pto.f32)
col_loop = pto.for_(0, cols, step=VEC).carry(remained=cols)
with col_loop:
    c = col_loop.iv
    remained = col_loop.remained
    mask, remained = pto.make_mask(pto.f32, remained)
    vec = pto.vlds(tile[r, c:])
    # ... operate under mask ...
    pto.vsts(vec, out_tile[r, c:], mask)
    col_loop.update(remained=remained)
```

`make_mask(dtype, n)` returns two values: the predicate mask for the current chunk and the updated remaining count. Passing the updated count back via `col_loop.update(remained=...)` feeds it into the next iteration, so each chunk correctly computes how many elements are left. If `n` is an `index`, the updated remaining count stays an `index`; PTODSL hides the hardware `i32` tail-mask bookkeeping internally.

## 5.3 `pto.if_` — device-side conditionals

`pto.if_` records a device-side conditional branch. Unlike a Python `if`, the condition can be a runtime PTO scalar, and both branches are recorded into the program so the hardware can choose at runtime.

The condition must be a PTO scalar value (e.g., the result of a comparison like `a > b` or a value loaded from a tile). Python booleans evaluated at trace time should use a plain `if` instead.

### Recommended block structure

PTODSL should treat one device-side conditional as one explicit branch object.
The recommended surface is:

```python
with pto.if_(cond) as br:
    with br.then_:
        ...
    with br.else_:
        ...
```

This keeps the `if` / `else` pairing explicit. The `else_` branch is optional
for side-effect-only conditionals.

### Automatic named merge across branches

When a value must flow out of both branches, PTODSL should merge by explicit
name. Each branch assigns the same output names with `br.assign(...)`, and the
merged results are read back from the branch handle after the conditional:

```python
@pto.simt
def conditional_scale(
    tile: pto.Tile,
    threshold: pto.f32,
    scale: pto.f32,
    rows: pto.i32,
    cols: pto.i32,
):
    with pto.for_(0, rows, step=1) as r:
        with pto.for_(0, cols, step=1) as c:
            val = scalar.load(tile[r, c])
            big = val > threshold

            with pto.if_(big) as br:
                with br.then_:
                    br.assign(val=val * scale)
                with br.else_:
                    br.assign(val=val)

            val = br.val
            scalar.store(val, tile[r, c])
```

In this example, both branches define the merged value named `val`. After the
conditional closes, `br.val` is the SSA-merged result seen by downstream code.
This surface avoids explicit result-type declarations and explicit
`pto.yield_(...)` in user code while still keeping the merge contract explicit.

## 5.4 `pto.constexpr` and tracing

`pto.constexpr` parameters (Section 3.8) are compile-time constants. They are fixed at `.compile()` time and cannot change between launches of the same compiled kernel. Because their values are known during tracing, they interact naturally with Python control flow:

```python
@pto.jit(target="a5")
def kernel(
    A,
    *,
    BLOCK: pto.constexpr = 128,
    NUM_BLOCKS: pto.constexpr = 8,
    UNROLL: pto.constexpr = False,
):
    N = A.shape[0]
    num_blocks = (N + BLOCK - 1) // BLOCK

    # N and num_blocks are runtime values derived from tensor metadata.
    # They can drive device-side control flow such as pto.for_(...),
    # but they are not Python integers and cannot be used in range(...).
    with pto.for_(0, num_blocks, step=1) as i:
        ...

    if UNROLL:
        # Trace-time: UNROLL and NUM_BLOCKS are both known during tracing.
        # Each iteration records separately, so the loop is fully unrolled.
        for i in range(NUM_BLOCKS):
            ...
    else:
        # The non-unrolled path can still use a device-side loop whose bound
        # is a constexpr value captured into the traced program.
        with pto.for_(0, NUM_BLOCKS, step=1) as i:
            ...
```

This lets you write a single kernel that specializes into different strategies based on constexpr knobs, while still using runtime tensor metadata for device-side control flow.

## 5.5 Summary

| Construct | When evaluated | Use for |
|-----------|---------------|---------|
| Python `for` | Trace time | Bounds known at compile time (constexpr), deliberate unrolling |
| Python `if` | Trace time | Conditions known at compile time, variant selection |
| `pto.for_` | Device-side | Dynamic bounds, runtime loop counts |
| `pto.for_(...).carry(...)` | Device-side | Loops with accumulated state across iterations |
| `pto.if_` | Device-side | Runtime conditions, data-dependent branching |
