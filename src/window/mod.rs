use std::fmt::Display;

use crate::window::quantile_window::block_size_dispatcher::{
    StdBlockSizes,
    get_suitable_std_block_size
};

mod quantile_window;
mod utils;

const QUANTILE_EPSILON: f64 = 1e-9;

#[derive(Debug, PartialEq, Eq)]
pub enum WindowError {
    InputArrayIsEmptyError,
    SizingError,
    InvalidQuantileError,
}

impl Display for WindowError {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowError::InputArrayIsEmptyError => write!(f, "InputArrayIsEmptyError: It seems the \
                input array is empty"),
            WindowError::SizingError => write!(f, "SizingError: It seems you entered a wrong size"),
            WindowError::InvalidQuantileError => write!(f, "InvalidQuantileError: \
                It seems you entered an invalid quantile"),
        }
    }
}

impl std::error::Error for WindowError {}

/// Computes a rolling quantile over an input slice/array.
///
/// Uses an optimized rolling window mechanism to return a vector containing the calculated quantile
/// for every possible window position.
/// Please note: This window will always step one element to the right! The steps are not parameterizable.
///
/// Also note: [`f64::NAN`] will be counted as missing and does not enter the quantile.
/// In this case the quantile will be calculated from the remaining valid elements.
/// In explanation: When there are two of one hundred elements equals to [`f64::NAN`] the quantile will
/// be calculated from the remaining ninety eight elements.
///
/// # Block size
/// Internally the window is split into fixed-size blocks. The block size is a trade-off:
/// larger blocks mean fewer blocks and therefore shallower tournament trees, but a longer
/// linear scan inside each block. Which side dominates depends on the window size.
///
/// This function makes that choice for you. It derives the appropriate block size by
/// comparing the metadata each candidate produces for the given `window_size`. In this context
/// it selects between possible block sizes of 16, 32 and 64; 64 is the largest useful size,
/// larger ones gained nothing in any benchmark.
/// Across all measured window sizes the selection picks the fastest of those three
/// (`cargo bench --bench rolling -- block_sizes_bench`).
/// For more information on the implementation, take a look at `quantile_window::block_size_dispatcher`.
///
/// Use [`rolling_quantile_window_generic`] if you want to choose the block size yourself.
///
/// # Returns
/// A vector containing the calculated quantiles.
/// Please note: The vector always will be of the size [(`input_array.len()` - `window_size`) + 1].
///
/// # Errors
/// Returns a [`WindowError`] if:
/// * `input_array` is empty ([`WindowError::InputArrayIsEmptyError`]).
/// * `window_size` is 0 or larger than `input_array.len()` ([`WindowError::SizingError`]).
/// * `quantile` is outside the valid range ([`0.0, 1.0`]) ([`WindowError::InvalidQuantileError`]).
///
/// # Example
/// ```
/// use quantile_window::rolling_quantile_window;
///
/// // 0.5 quantile
/// let test_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let result_quantiles = rolling_quantile_window(&test_input, 3, 0.5).unwrap();
/// assert_eq!(&result_quantiles, &[2.0, 3.0, 4.0, 5.0]);
///
/// // 0.0 quantile
/// let result_quantiles = rolling_quantile_window(&test_input, 3, 0.0).unwrap();
/// assert_eq!(&result_quantiles, &[1.0, 2.0, 3.0, 4.0]);
///
/// // 1.0 quantile
/// let result_quantiles = rolling_quantile_window(&test_input, 3, 1.0).unwrap();
/// assert_eq!(&result_quantiles, &[3.0, 4.0, 5.0, 6.0]);
///
/// // NaN example
/// let test_input = [1.0, 2.0, 3.0, f64::NAN, 5.0, 6.0];
/// let result_quantiles = rolling_quantile_window(&test_input, 3, 0.5).unwrap();
/// assert_eq!(&result_quantiles, &[2.0, 2.5, 4.0, 5.5]);
///
/// // Full of NaN example
/// let test_input = [1.0, f64::NAN, f64::NAN, f64::NAN, 5.0, 6.0];
/// let result_quantiles = rolling_quantile_window(&test_input, 3, 0.5).unwrap();
/// assert!(!result_quantiles[0].is_nan());
/// assert!(result_quantiles[1].is_nan());
/// assert!(!result_quantiles[2].is_nan());
/// assert!(!result_quantiles[3].is_nan());
/// ```
pub fn rolling_quantile_window(
    input_array: &[f64],
    window_size: usize,
    quantile: f64
) -> Result<Vec<f64>, WindowError> {
    let suitable_block_size = get_suitable_std_block_size(window_size);
    match suitable_block_size {
        StdBlockSizes::B16 => rolling_quantile_window_generic::<16>(
            input_array,
            window_size,
            quantile
        ),

        StdBlockSizes::B32 => rolling_quantile_window_generic::<32>(
            input_array,
            window_size,
            quantile
        ),

        StdBlockSizes::B64 => rolling_quantile_window_generic::<64>(
            input_array,
            window_size,
            quantile
        )
    }
}

/// Computes a rolling quantile over an input slice/array.
///
/// Uses an optimized rolling window mechanism to return a vector containing the calculated quantile
/// for every possible window position.
/// Please note: This window will always step one element to the right! The steps are not parameterizable.
///
/// Also note: [`f64::NAN`] will be counted as missing and does not enter the quantile.
/// In this case the quantile will be calculated from the remaining valid elements.
/// In explanation: When there are two of one hundred elements equals to [`f64::NAN`] the quantile will
/// be calculated from the remaining ninety eight elements.
///
/// This function includes one special parameter: `BLOCK_SIZE` as a const generic. If `BLOCK_SIZE` does not
/// match the following constraints compilation will fail.
///
/// # Constraints
/// - `BLOCK_SIZE` must be greater than zero.
/// - `BLOCK_SIZE` must be a value that is divisible by the size of the underlying sorting network. In this
///   implementation, the sorting network has a size of 16.
///
/// `BLOCK_SIZE` is a trade-off rather than a "smaller is better" choice. A larger block makes the linear search
/// inside a block more expensive, but reduces the number of blocks and therefore the depth of the underlying
/// tournament trees. Which side wins depends on the window size, so there is no single best value.
///
/// [`rolling_quantile_window`] makes that choice automatically; use this function only to override it - to
/// benchmark a particular size, or to pin one for a workload whose window size is known in advance.
///
/// # Returns
/// A vector containing the calculated quantiles.
/// Please note: The vector always will be of the size [(`input_array.len()` - `window_size`) + 1].
///
/// # Errors
/// Returns a [`WindowError`] if:
/// * `input_array` is empty ([`WindowError::InputArrayIsEmptyError`]).
/// * `window_size` is 0 or larger than `input_array.len()` ([`WindowError::SizingError`]).
/// * `quantile` is outside the valid range ([`0.0, 1.0`]) ([`WindowError::InvalidQuantileError`]).
///
/// # Example
/// ```
/// use quantile_window::rolling_quantile_window_generic;
///
/// // 0.5 quantile
/// let test_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let result_quantiles = rolling_quantile_window_generic::<16>(&test_input, 3, 0.5).unwrap();
/// assert_eq!(&result_quantiles, &[2.0, 3.0, 4.0, 5.0]);
///
/// // 0.5 quantile
/// let test_input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let result_quantiles = rolling_quantile_window_generic::<64>(&test_input, 3, 0.5).unwrap();
/// assert_eq!(&result_quantiles, &[2.0, 3.0, 4.0, 5.0]);
/// ```
pub fn rolling_quantile_window_generic<const BLOCK_SIZE: usize>(
    input_array: &[f64],
    window_size: usize,
    quantile: f64
) -> Result<Vec<f64>, WindowError> {
    if input_array.is_empty() {
        return Err(WindowError::InputArrayIsEmptyError);
    }

    if window_size == 0 || window_size > input_array.len() {
        return Err(WindowError::SizingError);
    }

    if !valid_quantile(quantile) {
        return Err(WindowError::InvalidQuantileError);
    }

    Ok(
        quantile_window::rolling_window::<BLOCK_SIZE>(
            input_array,
            window_size,
            quantile
        )
    )
}

#[inline(always)]
fn valid_quantile(quantile: f64) -> bool {
    if quantile.is_nan() || quantile.is_infinite() {
        return false;
    }

    if !(0.0..=1.0).contains(&quantile) {
        return false;
    }

    let scaled_up = quantile * 100.0;
    let computed = scaled_up - scaled_up.round();
    computed.abs() < QUANTILE_EPSILON
}

#[cfg(test)]
mod test {
    use crate::window::{
        rolling_quantile_window_generic,
        valid_quantile,
        WindowError
    };

    const TESTED_QUANTILES: [(f64, bool); 12] = [
        (0.01, true), (0.5, true), (0.001, false), (1.0, true), (1.01, false),
        (0.0, true), (-0.5, false), (-0.01, false), (0.75, true), (0.123, false),
        (f64::NAN, false), (f64::INFINITY, false)
    ];

    #[test]
    fn test_valid_quantile() {
        TESTED_QUANTILES
            .iter()
            .for_each(|(value, valid)| {
                assert!(valid_quantile(*value) == *valid);
            });
    }

    #[test]
    fn test_rolling_quantile_invalid_input() {
        let test_input: [f64; 0] = [];
        let call_result = rolling_quantile_window_generic::<16>(
            &test_input,
            128,
            0.5
        );
        assert!(call_result.err().unwrap() == WindowError::InputArrayIsEmptyError);

        let test_input = [1.0, 2.0, 3.0, 4.0, 5.0];
        let call_result = rolling_quantile_window_generic::<16>(
            &test_input,
            0,
            0.5
        );
        assert!(call_result.err().unwrap() == WindowError::SizingError);

        let call_result = rolling_quantile_window_generic::<16>(
            &test_input,
            16,
            0.5
        );
        assert!(call_result.err().unwrap() == WindowError::SizingError);

        let call_result = rolling_quantile_window_generic::<16>(
            &test_input,
            3,
            1.1
        );
        assert!(call_result.err().unwrap() == WindowError::InvalidQuantileError);
    }
}
