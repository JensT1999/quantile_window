mod common;

use common::{
    process_and_assert_results
};

const TESTED_QUANTILES: [f64; 3] = [
    0.0, 0.5, 1.0
];

const VALID_TEST_INPUT: [f64; 10] = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0
];

#[test]
fn test_window_size_equals_input_length() {
    for quantile in TESTED_QUANTILES {
        let test_label = format!("window size equals input length - quantile: {}", quantile);
        process_and_assert_results(
            &VALID_TEST_INPUT,
            &test_label,
            VALID_TEST_INPUT.len(),
            quantile
        );
    }
}

#[test]
fn test_window_full_of_nan_at_beginning() {
    let nan_input = [f64::NAN; 10];
    let test_input = [nan_input, VALID_TEST_INPUT].concat();

    process_and_assert_results(
        &test_input,
        "window full of nans at beginning",
        10,
        0.5
    );
}

#[test]
fn test_window_size_one() {
    let mut test_input = VALID_TEST_INPUT.to_vec();
    test_input[3] = f64::NAN;
    test_input[7] = f64::NAN;

    let result_quantiles = quantile_window::rolling_quantile_window(
        &test_input,
        1,
        0.5
    ).unwrap();

    assert_eq!(result_quantiles.len(), test_input.len());
    let zipped_iter = test_input
        .iter()
        .zip(result_quantiles.iter());
    for (expected_result, calculated_result) in zipped_iter {
        assert!(
            (expected_result.is_nan() && calculated_result.is_nan()) ||
                expected_result == calculated_result
        );
    }
}
