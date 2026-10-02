mod common;

use common::{
    gen_continous,
    sparse_data,
    process_and_assert_results
};

use rand::{SeedableRng, rngs::StdRng};

const TESTED_DATA_SEED: u64 = 42;
const TESTED_INPUT_SIZE: usize = 1000;
const TESTED_WINDOW_SIZE: usize = 128;
const LOWEST_POSSIBLE_VALUE: f64 = -1000.0;
const HIGHEST_POSSIBLE_VALUE: f64 = 1000.0;
const TESTED_QUANTILES: [f64; 4] = [
    0.0, 0.01, 0.5, 1.0
];

// Edge quantile - in case there is no successor, we need to shift to our global predeccessor.
// Otherwise our floor value would get f64::NAN and per definiton not valid.
#[test]
fn test_no_valid_successor() {
    let test_data = [2.0, 1.0, 0.0];
    let result = quantile_window::rolling_quantile_window(
        &test_data,
        2,
        1.0
    ).unwrap();
    assert_eq!(&result, &[2.0, 1.0]);

    let test_data = [3.0, 2.0, 1.0, 0.0];
    let result = quantile_window::rolling_quantile_window(
        &test_data,
        3,
        1.0
    ).unwrap();
    assert_eq!(&result, &[3.0, 2.0]);

    let test_data = [1.0, 2.0, 3.0, 4.0, 5.0];
    let result = quantile_window::rolling_quantile_window(
        &test_data,
        3,
        1.0
    ).unwrap();
    assert_eq!(&result, &[3.0, 4.0, 5.0]);
}

// Wingow got only one valid element or got/was empty. In case the last valid element was replaced by a other valid
// (case offset 0) element the window should have no valid predeccessor or successor and just fill in the new
// element. In case the window was completely empty (offset 1) - the window should get f64::NAN and get
// reinitialized by the next window step.
#[test]
fn test_no_valid_successor_or_predeccessor() {
    let mut rng = StdRng::seed_from_u64(TESTED_DATA_SEED);
    let test_data = gen_continous(
        TESTED_INPUT_SIZE,
        &mut rng,
        LOWEST_POSSIBLE_VALUE,
        HIGHEST_POSSIBLE_VALUE
    );

    for offset in 0usize..=1 {
        let sparse_test_data: Vec<f64> = sparse_data(&test_data, TESTED_WINDOW_SIZE + offset);
        let test_label = format!("no succ or pred test - offset: {}", offset);
        for quantile in TESTED_QUANTILES {
            process_and_assert_results(
                &sparse_test_data,
                &test_label,
                TESTED_WINDOW_SIZE,
                quantile
            );
        }
    }
}

// Window leaves ran out right state. While the tracker points to a invalid value e.g. f64::NAN when ran out right,
// it should point to a valid value when returning to a not ran out right state e.g. tracker should be
// length of the values block minus one.
#[test]
fn test_minimal_duplicates() {
    let test_data = [
        3.0, 1.0, 2.0, 0.0, 2.0, 1.0, 1.0, 3.0, 1.0, 1.0, 3.0, 0.0, 1.0, 3.0, 2.0, 0.0, 1.0, 1.0,
        3.0, 3.0, 2.0, 2.0, 0.0, 3.0, 3.0, 0.0, 0.0, 2.0, 1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 1.0
    ];

    for quantile in TESTED_QUANTILES {
        process_and_assert_results(
            &test_data,
            "minimal duplicate data",
            18,
            quantile
        );
    }
}

// The interpolation originally failed when floor value and successor value were f64::INFINITY both. This test
// checks if f64::INFINITY and f64::INFINITY result in f64::INFINITY, as well as f64::NEG_INFINITY. The expected
// result for the interpolation between f64::NEG_INFINITY and f64::INFINITY is f64::NAN.
#[test]
fn test_inf_input() {
    let test_data = [
        f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY
    ];

    let result = quantile_window::rolling_quantile_window(
        &test_data,
        2,
        0.5
    ).unwrap();

    [f64::NAN, f64::NAN, f64::INFINITY, f64::NAN, f64::NEG_INFINITY]
        .iter()
        .zip(result.iter())
        .for_each(|(value_a, value_b)| {
            if value_a.is_nan() || value_b.is_nan() {
                assert!(
                    value_a.is_nan() && value_b.is_nan()
                );
            } else {
                assert_eq!(value_a, value_b)
            }
        });

}
