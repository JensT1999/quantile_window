#![allow(dead_code)]
use std::{cmp::Ordering};
use rand::{RngExt, rngs::StdRng};

pub fn naive_quantile_gen(
    input_vec: &[f64],
    window_size: usize,
    quantile: f64
) -> Vec<f64> {
    assert!(window_size <= input_vec.len());
    let num_windows = input_vec.len() - window_size + 1;
    let mut result_vec = Vec::with_capacity(num_windows);
    for windows in 0..num_windows {
        let window_slice = &input_vec[windows..windows + window_size];
        let mut nan_handled_array = window_slice.iter()
            .copied()
            .filter(|x| !x.is_nan())
            .collect::<Vec<f64>>();

        if nan_handled_array.is_empty() {
            result_vec.push(f64::NAN);
            continue;
        }
        nan_handled_array.sort_by(f64::total_cmp);

        let searched_rank = quantile * (nan_handled_array.len() - 1) as f64;
        let floor_rank = searched_rank.floor() as usize;
        if searched_rank % 1.0 == 0.0 {
            result_vec.push(nan_handled_array[floor_rank]);
        } else {
            let floor_value = nan_handled_array[floor_rank];
            let ceil_rank = floor_rank + 1;
            let ceil_value = nan_handled_array[ceil_rank];
            let interpolated_quantile = floor_value + (ceil_value - floor_value) *
                (searched_rank - searched_rank.floor());
            result_vec.push(interpolated_quantile);
        }
    }

    result_vec
}

pub fn assert_quantile_results(
    vec1: &[f64],
    vec2: &[f64],
    test_label: &str,
    window: usize,
    quantile: f64
) {
    assert_eq!(vec1.len(), vec2.len());

    let panic_message = |
        index: usize,
        item1: f64,
        item2: f64,
        window: usize,
        quantile: f64
    | {
        panic!("Panicked at index: {} {} {} while processing {} and calculating window: {} with quantile: {}",
            index,
            item1,
            item2,
            test_label,
            window,
            quantile
        );
    };

    let iterator = vec1.iter().zip(vec2.iter());
    for (index, item) in iterator.enumerate() {
        let (i1, i2) = item;
        if !i1.is_nan() && !i2.is_nan() {
            if *i1 != *i2 {
                panic_message(
                    index,
                    *i1,
                    *i2,
                    window,
                    quantile
                );
            }
        } else {
            if i1.is_nan() && i2.is_nan() {
                continue;
            }

            panic_message(
                index,
                *i1,
                *i2,
                window,
                quantile
            );
        }
    }
}

pub fn gen_continous(
    length: usize,
    rng: &mut StdRng,
    lowest_possible_value: f64,
    highest_possible_value: f64
) -> Vec<f64> {
    (0..length)
        .map(|_| rng.random_range(lowest_possible_value..highest_possible_value))
        .collect::<Vec<f64>>()
}

pub fn gen_monotonous(
    length: usize,
    rng: &mut StdRng,
    gradient: f64,
    window_size: usize,
    lowest_possible_value: f64,
    highest_possible_value: f64
) -> Vec<f64> {
    assert!(gradient.abs() >= 1.0);
    let span = highest_possible_value - lowest_possible_value;
    let trend_per_step = gradient * (span / window_size as f64);
    (0..length)
        .map(|index| {
            let rand_value = rng.random_range(lowest_possible_value..highest_possible_value);
            rand_value + (index as f64 * trend_per_step)
        })
        .collect::<Vec<f64>>()
}

pub fn nan_fill_data(
    data: &[f64],
    rng: &mut StdRng,
    nan_ratio: f64
) -> Vec<f64> {
    assert!(nan_ratio > 0.0 && nan_ratio <= 1.0);
    data
        .iter()
        .map(|x| {
            if rng.random_bool(nan_ratio) {
                f64::NAN
            } else {
                *x
            }
        })
        .collect::<Vec<f64>>()
}

pub fn quantize_data(
    data: &[f64],
    step: f64,
) -> Vec<f64> {
    assert!(step.is_finite() && step > 0.0);
    data
        .iter()
        .map(|x| {
            (x / step).round() * step
        })
        .collect::<Vec<f64>>()
}

pub fn sparse_data(
    data: &[f64],
    spacing: usize
) -> Vec<f64> {
    data
        .iter()
        .enumerate()
        .map(|item| {
            if item.0 % spacing == 0 {
                *item.1
            } else {
                f64::NAN
            }
        })
        .collect::<Vec<f64>>()
}

pub fn count_unique_values(
    data: &[f64]
) -> usize {
    let mut copied_vector = data.to_vec();
    copied_vector.sort_by(f64::total_cmp);
    copied_vector.dedup_by(|a, b| a.total_cmp(b) == Ordering::Equal);
    copied_vector.len()
}

pub fn process_and_assert_results(
    test_data: &[f64],
    test_label: &str,
    window_size: usize,
    quantile: f64
) {
    let naive_result = naive_quantile_gen(
        test_data,
        window_size,
        quantile
    );

    process_results_and_assert_naive_results::<16>(
        test_data,
        test_label,
        window_size,
        quantile,
        &naive_result
    );

    process_results_and_assert_naive_results::<32>(
        test_data,
        test_label,
        window_size,
        quantile,
        &naive_result
    );

    process_results_and_assert_naive_results::<64>(
        test_data,
        test_label,
        window_size,
        quantile,
        &naive_result
    );

    process_results_and_assert_naive_results::<128>(
        test_data,
        test_label,
        window_size,
        quantile,
        &naive_result
    );
}

fn process_results_and_assert_naive_results<const BLOCK_SIZE: usize>(
    test_data: &[f64],
    test_label: &str,
    window_size: usize,
    quantile: f64,
    naive_result: &[f64]
) {
    let test_result = quantile_window::rolling_quantile_window_generic::<BLOCK_SIZE>(
        test_data,
        window_size,
        quantile
    ).unwrap();

    let test_label = format!("{} - block size: {}", test_label, BLOCK_SIZE);
    assert_quantile_results(
        naive_result,
        &test_result,
        &test_label,
        window_size,
        quantile
    );
}
