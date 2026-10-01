# quantile_window

A rolling quantile over `f64` slices: for every position of a sliding window, the quantile of the values inside it is calculated. Single-threaded, no runtime dependencies, and no heap allocation once the window is set up.

## Motivation

This project grew out of my interest in processing data as efficiently as possible, in the context of rolling windows. Rolling quantiles are a special case among them, because they require access by rank. A rolling sum or mean, by contrast, can be computed by adding the incoming value and subtracting the outgoing one.

## Why this approach?

One note in advance: all complexities below are stated with respect to rolling windows. The variable `w` therefore stands for the window size as the number of elements an operation works on. In a complexity bound this is usually written as `n`.

The most obvious approach to computing rolling quantiles is to sort every window. That costs `O(w log w)` per position, and therefore scales worse and worse as the window size grows. So alternatives are needed, and one of them is the order statistic tree.

An order statistic tree is well suited for maintaining the ranking within a window. Also it has a time complexity of `O(log w)` for inserting, searching and deleting elements. That complexity, however, does not take the reality of hardware into account. The nodes of such a tree are normally connected to one another by references. The rebalancing that takes place, depending on the concrete implementation of the tree, causes those references to change continuously. Nodes that logically belong together therefore tend to lie scattered across memory, while the arrangement keeps changing as the tree is
used.

As long as related nodes lie close to one another, or are already in the CPU's cache, this costs little. But as soon as the tree grows and consequently claims more memory, the probability rises that more and more nodes lie outside the caches. Since memory accesses are expensive, the cost of the tree rises with the window size. This happens in a way the time complexity cannot show, because that bound is tied to the comparisons alone and not to the cache misses.

## Implementation

This implementation keeps the window in **fixed-size sorted blocks** of `BLOCK_SIZE` values each and maintains two `8`-ary **tournament trees** over them. One is a **successor tree**, whose root is the global successor across all blocks. The other is a **predecessor tree**, whose root is the global predecessor across all blocks. It also keeps a queue of the values in the order they were inserted. This tells the window which value is leaving, so callers do not have to keep track of it themselves. In addition, the window holds a global floor value corresponding to the value at the rank of the currently searched quantile — or the lower of the two interpolated values, when the searched rank is not an integer.

Each block carries a tracker that divides its values into the part lying at or below the global searched rank and the part above it. A tracker moves only when a value at or below the position it points to enters or leaves the block. If, for instance, the outgoing value was on the left side of the tracker while the incoming one is inserted on the right, the tracker effectively moves one position to the left. This means that the block now holds one fewer element below the searched rank.

These local movements are accumulated into a single global delta. A delta of one step to the left means the window is one value short of the searched rank. Consequently, the smallest value above the current searched rank becomes the new floor value, which is the root of the successor tree. A delta in the other direction is settled the same way, through the root of the predecessor tree.

The dominant work per step is:

* linear scans and a shift inside a single block, bounded by the length of the block.
* possible leaf-to-root updates in the trees, bounded by the tree depth.

Both bounds depend on the window size only, never on the length of the input. The decisive difference from an order statistic tree is what the trees actually hold. An order statistic tree stores every value in the window, so it grows with `w`. The tournament trees store one entry per *block* — in other words `w / BLOCK_SIZE` of them — which reduces their depth to `O(log(w / BLOCK_SIZE))`. As a result, the tree structure that has to be traversed at each step is small enough that the chance of it staying in cache is higher.

The price is paid inside the block: locating the outgoing value and inserting the incoming value are linear scans and a shift over up to `BLOCK_SIZE` values. Hence the cost per step is `O(BLOCK_SIZE)` + `O(log(w / BLOCK_SIZE))` per tree update rather than `O(log w)`. This means more raw work, but in exchange that work happens on contiguous memory instead of chasing references through a structure that keeps rearranging itself.

In addition, the blocks are arranged in the form of a circular buffer on contiguous memory. Every time a block updates its number of values (mostly equal to `BLOCK_SIZE`, only exception is the last block), the process advances to the next block (the immediate neighbor of the previous block). Only when all existing blocks have been updated does the cycle repeat, continuing until all elements of the input data have passed through the window. The queue is updated in the exact same manner.

## Features

**`NaN` means missing, not "a value".** A `NaN` never contributes to the quantile. The rank is always computed over the number of valid values currently in the window. The result will be `NaN` if and only if the window contains no valid value at all. `NaN` always maps to the largest key of a total order over `f64` (the same bit trick as `f64::total_cmp`). Therefore missing values collect at the right edge of every sorted block and can never end up between two valid values.

**Zero runtime dependencies.** Only `rand` as a development dependency, used only by the benchmarks and tests.

## Correctness

* **Property tests** generate inputs and compare against a naive reference implementation that sorts each window. Every test runs against four `BLOCK_SIZE` instantiations (16, 32, 64 and 128) that are also used in the benchmarks. A bug that only appears at a specific `BLOCK_SIZE` cannot pass.
* **Regression tests** cover bugs that actually occurred, each with its specific description.
* **API tests** cover the boundary cases: window equal to the input length, an all-`NaN` window, and window size 1 including `NaN`.
* **An allocation test** verifies the allocation behavior of the implementation. All memory allocations occur at the setup. Once the window is filled, nothing is allocated again. `tests/allocations.rs` proves this by installing a global counting allocator.
* **Unsafe code** carries a `SAFETY` comment for every block. Clippy's `undocumented_unsafe_blocks` lint is enabled in `lib.rs` and reports nothing.

## Measurements

All numbers below were measured under the following conditions:

* Rust 1.98.1 (2026-09-01)
* quantile_window version 0.1.0
* release build
* single core
* MacBook Air M3
* using benchmark suite in `benches/rolling.rs`

Run the suite yourself with:

```txt
cargo bench --bench rolling -- std_bench
```

This suite performs measurements regarding the influence of different input lengths, different distributions, and various quantiles.

The following reported performance metrics are the fastest of four repetitions.

First things first: The three measurements below share a comparable base line. Each one of them holds a cell with a measure of an input length of 20 000 000, a window size of 1000, a quantile of 0.5, and a continuous distribution. Consequently, this base line is measured three times independently, with results of 30.0 M/s, 30.0 M/s and 29.9 M/s. Therefore the resulting spread between those measurements can be interpreted as a measure of run-to-run
reproducibility.

### Length dependence performance

Configuration: window 1000, quantile 0.5, continuous between `[-1000.0..1000.0]`

| input length | 5 000 000 | 10 000 000 | 20 000 000 |
|---|---|---|---|
| throughput | 30.1 M/s | 30.1 M/s | 30.0 M/s |

As can be seen, the length of the input data is irrelevant to the throughput of the window. Despite a fourfold increase in input length across the columns, no big effect is measurable.

### NaN handling performance

Configuration: input length 20 000 000, quantile 0.5, continuous between `[-1000.0..1000.0]`, `NaN` ratio 0.2 and `NaN` ratio 0.4, without `NaN` matches a continuous distribution between `[-1000.0..1000.0]`

| window | with nan (0.2) | with nan (0.4) | without nan |
|---|---|---|---|
| 100 | 29.3 M/s | 31.4 M/s | 31.7 M/s |
| 1000 | 28.9 M/s | 31.1 M/s | 30.0 M/s |
| 10000 | 27.0 M/s | 29.1 M/s | 27.6 M/s |
| 100000 | 26.3 M/s | 28.4 M/s | 26.9 M/s |
| 1000000 | 26.2 M/s | 28.2 M/s | 26.5 M/s |
| 2000000 | 25.0 M/s | 27.1 M/s | 25.5 M/s |

As can be seen, a higher number of missing values in the form of `NaN`s accelerates computation. However, it is also apparent that this only holds true once the percentage of `NaN`s relative to the input data reaches a certain threshold. In contrast to the distribution without `NaN`s, a `NaN` ratio of 0.2 results in a slower processing speed, whereas a `NaN` ratio of 0.4 shows a performance gain over it. The only exception is the window with a size of 100, where the without `NaN` distribution is the fastest.

### Different quantiles performance

Configuration: window 1000, input length 20 000 000, continuous between `[-1000.0..1000.0]`

| quantile | throughput |
|---|---|
| 0.0 | 39.5 M/s |
| 0.25 | 30.7 M/s |
| 0.5 | 29.9 M/s |
| 0.75 | 30.7 M/s |
| 1.0 | 40.3 M/s |

As can be seen, the computation gains performance toward the outer quantiles. The median therefore represents the **worst-case** scenario across the entire spectrum of possible quantiles.

## Usage

### Standard dispatcher

In this case, the `BLOCK_SIZE` is chosen for you automatically. A larger `BLOCK_SIZE` makes the scan inside a block more expensive but produces fewer blocks and therefore shallower trees. Which side dominates depends on the window size, so `rolling_quantile_window` derives the block size from the metadata that the window size implies and selects between sizes of 16, 32 and 64.

```rust
use quantile_window::rolling_quantile_window;

let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

// median of every window of length 3
let medians = rolling_quantile_window(&input, 3, 0.5).unwrap();
assert_eq!(&medians, &[2.0, 3.0, 4.0, 5.0]);

// NaN counts as missing, not as a value
let with_nan = [1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0];
let medians = rolling_quantile_window(&with_nan, 3, 0.5).unwrap();
assert_eq!(&medians, &[2.0, 2.5, 4.0, 5.5]);
```

### Choose your own `BLOCK_SIZE`

`rolling_quantile_window_generic` exposes the `BLOCK_SIZE` as a const generic. Use it for benchmarking or for a workload whose window size is known in advance:

```rust
use quantile_window::rolling_quantile_window_generic;

let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
let medians = rolling_quantile_window_generic::<64>(&input, 3, 0.5).unwrap();
assert_eq!(&medians, &[2.0, 3.0, 4.0, 5.0]);
```

Any positive multiple of 16 works, powers of two are not required, and there is no upper limit for `BLOCK_SIZE`.

Note that with a `BLOCK_SIZE` above 512, the internal sort needs an allocated buffer. Therefore, the complete computation needs one additional allocation per block. The update phase remains allocation-free either way, because sorting only happens while the window is being filled.

### Handling errors

The computation can fail if an invalid input is given. In those cases a `WindowError` is returned. There are three distinct variants:

1. **`InputArrayIsEmptyError`** — occurs when the given array is empty.
2. **`SizingError`** — occurs when the given window size is zero, or exceeds the length of the input data.
3. **`InvalidQuantileError`** — occurs when the given quantile is not valid. A valid quantile is not `NaN`, is finite, lies within the inclusive range of `0.0..=1.0`, and has no more than two decimal places.

## License

Licensed under either of:

* Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or
  <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
