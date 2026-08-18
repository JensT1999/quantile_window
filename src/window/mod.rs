use std::fmt::Display;

mod quantile_window;
mod utils;

const QUANTILE_EPSILON: f64 = 1e-9;

#[derive(Debug)]
pub enum WindowError {
    InputArrayIsEmptyError,
    SizingError,
    InvalidQuantileError,
}

impl Display for WindowError {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowError::InputArrayIsEmptyError => write!(f, "InputArrayIsEmptyError: It seems the
                input array is empty"),
            WindowError::SizingError => write!(f, "SizingError: It seems you entered a wrong size"),
            WindowError::InvalidQuantileError => write!(f, "InvalidQuantileError:
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
/// # Returns
/// A vector containing the calculated quantiles.
/// Please note: The vector always will be of the size [(input_array.len() - window_size) + 1].
///
/// # Errors
///
/// Returns a ['WindowError'] if:
/// * 'input_array' is empty (['WindowError::InputArrayIsEmptyError']).
/// * 'window_size' is 0 or larger than 'input_array.len()' (['WindowError::SizingError']).
/// * 'quantile' is outside the valid range (['0.0, 1.0']) (['WindowError::InvalidQuantileError'])
///
/// # Example
/// ```
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let res = rolling_quantile_window(&data, 3, 0.5);
/// assert!(res.is_ok());
/// ```
pub fn rolling_quantile_window(
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

    let result_vec = quantile_window::rolling_window(input_array, window_size, quantile);
    Ok(result_vec)
}

fn valid_quantile(quantile: f64) -> bool {
    if quantile.is_nan() || quantile.is_infinite() {
        return false;
    }

    if quantile < 0.0 || quantile > 1.0 {
        return false;
    }

    let scaled_up = quantile * 100.0;
    let computed = scaled_up - scaled_up.round();
    computed.abs() < QUANTILE_EPSILON
}
