mod common;

use common::{
    gen_continous,
    quantize_data,
    nan_fill_data,
    sparse_data,
    gen_monotonous,
    count_unique_values,
    process_and_assert_results
};
use rand::{SeedableRng, rngs::StdRng};

const TESTED_WINDOW_SIZES: [usize; 16] = [
    1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 1000
];

const TESTED_QUANTILES: [f64; 6] = [
    0.0, 0.01, 0.25, 0.5, 0.75, 1.0
];

// Those numbers are just an orientation, mostly it will generate a number of unique values nearby those numbers.
const NUM_UNIQUE_VALUES_IN_WINDOW: [usize; 3] = [
    200, 20, 4
];

const RATIO_OF_NAN_IN_WINDOW: [f64; 3] = [
    0.1, 0.3, 0.6
];

const GRADIENTS_TO_TEST: [f64; 6] = [
    1.0, -1.0, 2.0, -2.0, 10.0, -10.0
];

const TESTED_DATA_SEED: u64 = 42;
const TESTED_INPUT_LENGTH: usize = 2000;
const LOWEST_POSSIBLE_VALUE: f64 = -1000.0;
const HIGHEST_POSSIBLE_VALUE: f64 = 1000.0;

#[test]
fn test_continous_data() {
    let mut rng = StdRng::seed_from_u64(TESTED_DATA_SEED);
    let test_data = gen_test_data(&mut rng);

    for window_size in TESTED_WINDOW_SIZES {
        for quantile in TESTED_QUANTILES {
            process_and_assert_results(&test_data, "continous data", window_size, quantile);
        }
    }
}

#[test]
fn test_quantized_data() {
    let mut rng = StdRng::seed_from_u64(TESTED_DATA_SEED);
    let test_data = gen_test_data(&mut rng);

    let span = HIGHEST_POSSIBLE_VALUE - LOWEST_POSSIBLE_VALUE;
    for unique_values in NUM_UNIQUE_VALUES_IN_WINDOW {
        let step = span / unique_values as f64;
        let quantized_data = quantize_data(&test_data, step);

        let unique_values_count = count_unique_values(&quantized_data);
        assert!(
            (unique_values / 2) <= unique_values_count &&
            (unique_values + 2) >= unique_values_count
        );

        let test_label = format!("quantized data - step: {}", step);
        for window_size in TESTED_WINDOW_SIZES {
            for quantile in TESTED_QUANTILES {
                process_and_assert_results(&quantized_data, &test_label, window_size, quantile);
            }
        }
    }
}

#[test]
fn test_full_nan_data() {
    let mut rng = StdRng::seed_from_u64(TESTED_DATA_SEED);
    let test_data = gen_test_data(&mut rng);

    let nan_test_data = nan_fill_data(&test_data, &mut rng, 1.0);
    for window_size in TESTED_WINDOW_SIZES {
        for quantile in TESTED_QUANTILES {
            process_and_assert_results(&nan_test_data, "full of nan data", window_size, quantile);
        }
    }
}

#[test]
fn test_partial_nan_data() {
    let mut rng = StdRng::seed_from_u64(TESTED_DATA_SEED);
    let test_data = gen_test_data(&mut rng);

    for nan_ratio in RATIO_OF_NAN_IN_WINDOW {
        let nan_test_data = nan_fill_data(&test_data, &mut rng, nan_ratio);
        let test_label = format!("nan data - ratio: {}", nan_ratio);

        for window_size in TESTED_WINDOW_SIZES {
            for quantile in TESTED_QUANTILES {
                process_and_assert_results(&nan_test_data, &test_label, window_size, quantile);
            }
        }
    }
}

#[test]
fn test_nan_and_quantized_data() {
    let mut rng = StdRng::seed_from_u64(TESTED_DATA_SEED);
    let test_data = gen_test_data(&mut rng);

    let span = HIGHEST_POSSIBLE_VALUE - LOWEST_POSSIBLE_VALUE;
    for unique_values in NUM_UNIQUE_VALUES_IN_WINDOW {
        let step = span / unique_values as f64;
        let quantized_data = quantize_data(&test_data, step);

        let unique_values_count = count_unique_values(&quantized_data);
        assert!(
            (unique_values / 2) <= unique_values_count &&
            (unique_values + 2) >= unique_values_count
        );

        for nan_ratio in RATIO_OF_NAN_IN_WINDOW {
            let nan_test_data = nan_fill_data(&quantized_data, &mut rng, nan_ratio);
            let test_label = format!(
                "quantized data - step: {} - nan data - ratio: {}",
                step,
                nan_ratio
            );

            for window_size in TESTED_WINDOW_SIZES {
                for quantile in TESTED_QUANTILES {
                    process_and_assert_results(&nan_test_data, &test_label, window_size, quantile);
                }
            }
        }
    }
}

#[test]
fn test_sparse_data() {
    let mut rng = StdRng::seed_from_u64(TESTED_DATA_SEED);
    let test_data = gen_test_data(&mut rng);

    for window_size in TESTED_WINDOW_SIZES {
        let sparse_test_data = sparse_data(&test_data, window_size);
        let test_label = format!("sparse data - spacing {}", window_size);

        for quantile in TESTED_QUANTILES {
            process_and_assert_results(&sparse_test_data, &test_label, window_size, quantile);
        }
    }
}

#[test]
fn test_gradient_data() {
    let mut rng = StdRng::seed_from_u64(TESTED_DATA_SEED);

    for gradient in GRADIENTS_TO_TEST {
        for window_size in TESTED_WINDOW_SIZES {
            let test_data = gen_monotonous(
                TESTED_INPUT_LENGTH,
                &mut rng,
                gradient,
                window_size,
                LOWEST_POSSIBLE_VALUE,
                HIGHEST_POSSIBLE_VALUE
            );

            let test_label = format!("gradient data - gradient: {}", gradient);
            for quantile in TESTED_QUANTILES {
                process_and_assert_results(&test_data, &test_label, window_size, quantile);
            }
        }
    }
}

fn gen_test_data(
    rng: &mut StdRng
) -> Vec<f64> {
    gen_continous(
        TESTED_INPUT_LENGTH,
        rng,
        LOWEST_POSSIBLE_VALUE,
        HIGHEST_POSSIBLE_VALUE
    )
}
