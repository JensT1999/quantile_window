mod common;

use rand::{SeedableRng, rngs::StdRng};
use std::{alloc::{GlobalAlloc, System}, sync::atomic::AtomicUsize};

static ALLOCATIONS_COUNT: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;

// SAFETY: a global allocator is not allowed to unwind: `fetch_add` is not able to panic and `System`
// wont unwind, because it is also a global allocator and is subject to the same rules. In this
// `CountingAllocator` all parameters will always be passed through unchanged.
unsafe impl GlobalAlloc for CountingAllocator {

    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCATIONS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // SAFETY: `System.alloc` requires non zero sized `layout`. The safety contract for `alloc`
        // must be upheld by the caller.
        unsafe {
            System.alloc(layout)
        }
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCATIONS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // SAFETY: `System.alloc_zeroed` requires non zero sized `layout`. The safety contract for `alloc_zeroed`
        // must be upheld by the caller.
        unsafe {
            System.alloc_zeroed(layout)
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // SAFETY: `System.realloc` requires that `ptr` is allocated with this allocator,
        // that `layout` is the same as when allocated, `new_size` is greater than zero and
        // `new_size` is not allowed to overflow `isize`, when rounded up to the nearest multiple
        // of `layout.align()`. The safety contract for `realloc` must be upheld by the caller.
        unsafe {
            System.realloc(ptr, layout, new_size)
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
        // SAFETY: `System.dealloc` requires that `ptr` is allocated with this allocator,
        // that `layout` is the same as when allocated. The safety contract for `dealloc`
        // must be upheld by the caller.
        unsafe {
            System.dealloc(ptr, layout);
        }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

const RANDOM_DATA_SEED: u64 = 42;
const LOWEST_POSSIBLE_VALUE: f64 = -1000.0;
const HIGHEST_POSSIBLE_VALUE: f64 = 1000.0;

const TESTED_WINDOW_SIZES: [usize; 3] = [
    128, 512, 1000
];

const TESTED_QUANTILES: [f64; 1] = [
    0.0
];

const SHORT_INPUT_LENGTH: usize = 10000;
const LONG_INPUT_LENGTH: usize = 100000;

fn count_allocations(
    input_data: &[f64],
    window_size: usize,
    quantile: f64
) -> usize {
    let before_computation = ALLOCATIONS_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    let result_quantiles = quantile_window::rolling_quantile_window(
        input_data,
        window_size,
        quantile
    ).unwrap();
    let after_computation = ALLOCATIONS_COUNT.load(std::sync::atomic::Ordering::Relaxed);
    std::hint::black_box(&result_quantiles);
    after_computation - before_computation
}

#[test]
fn test_allocation_count_while_computation() {
    let mut rng = StdRng::seed_from_u64(RANDOM_DATA_SEED);
    let short_input_data = common::gen_continous(
        SHORT_INPUT_LENGTH,
        &mut rng,
        LOWEST_POSSIBLE_VALUE,
        HIGHEST_POSSIBLE_VALUE
    );
    let long_input_data = common::gen_continous(
        LONG_INPUT_LENGTH,
        &mut rng,
        LOWEST_POSSIBLE_VALUE,
        HIGHEST_POSSIBLE_VALUE
    );

    for window_size in TESTED_WINDOW_SIZES {
        for quantile in TESTED_QUANTILES {
            let setup_run_allocations = count_allocations(
                &short_input_data[..window_size],
                window_size,
                quantile
            );

            let short_run_allocations = count_allocations(
                &short_input_data,
                window_size,
                quantile
            );

            let long_run_allocations = count_allocations(
                &long_input_data,
                window_size,
                quantile
            );

            let assertion_failed_msg = format!(
                "The assert of the allocations failed, while calculating window size: {} and quantile: {}",
                window_size,
                quantile
            );

            // At this point we just check if the allocations of the window setup are the same as the
            // allocations on the short run. All allocations should happen inside the window setup,
            // never inside the update phase. Therefore the allocations between setup and short run should
            // be the same.
            // In addition there should be no allocation per update step. This is proven through the comparison
            // with a long run, because if the update phase allocated memory, the long run should allocate
            // more than the short run.
            // Also a comparison with a hard coded allocation count would be useless, because this would
            // force us to determine it before this test.
            // The goal of this test is just to show that the update phase is free from allocations.

            assert_eq!(
                setup_run_allocations,
                short_run_allocations,
                "{}",
                assertion_failed_msg
            );

            assert_eq!(
                short_run_allocations,
                long_run_allocations,
                "{}",
                assertion_failed_msg
            );
        }
    }
}
