mod common;

use common::{
    gen_continous
};

use rand::{SeedableRng, rngs::StdRng};
use std::{alloc::{GlobalAlloc, System}, sync::atomic::AtomicUsize};

static ALLOCATIONS_COUNT: AtomicUsize = AtomicUsize::new(0);

struct CountingAllocator;
unsafe impl GlobalAlloc for CountingAllocator {

    unsafe fn alloc(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCATIONS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            System.alloc(layout)
        }
    }

    unsafe fn alloc_zeroed(&self, layout: std::alloc::Layout) -> *mut u8 {
        ALLOCATIONS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            System.alloc_zeroed(layout)
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: std::alloc::Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS_COUNT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        unsafe {
            System.realloc(ptr, layout, new_size)
        }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: std::alloc::Layout) {
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
    let short_input_data = gen_continous(
        SHORT_INPUT_LENGTH,
        &mut rng,
        LOWEST_POSSIBLE_VALUE,
        HIGHEST_POSSIBLE_VALUE
    );
    let long_input_data = gen_continous(
        LONG_INPUT_LENGTH,
        &mut rng,
        LOWEST_POSSIBLE_VALUE,
        HIGHEST_POSSIBLE_VALUE
    );

    for window_size in TESTED_WINDOW_SIZES {
        for quantile in TESTED_QUANTILES {
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

            assert_eq!(
                short_run_allocations,
                long_run_allocations,
                "The assert of the allocations failed, while calculating window size: {} and quantile: {}",
                window_size,
                quantile
            );
        }
    }
}
