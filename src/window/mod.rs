use std::fmt::Display;

mod quantile_window;
mod utils;

// Needs to be documented
const WINDOW_SIZE_THRESHHOLD_FOR_SIZE_32: usize = 10000;
const WINDOW_SIZE_THRESHHOLD_FOR_SIZE_64: usize = 1500000;
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
pub fn rolling_quantile_window(
    input_array: &[f64],
    window_size: usize,
    quantile: f64
) -> Result<Vec<f64>, WindowError> {
    if window_size <= WINDOW_SIZE_THRESHHOLD_FOR_SIZE_32 {
        rolling_quantile_window_generic::<16,1>(
            input_array,
            window_size,
            quantile
        )
    } else if window_size <= WINDOW_SIZE_THRESHHOLD_FOR_SIZE_64 {
        rolling_quantile_window_generic::<32, 2>(
            input_array,
            window_size,
            quantile
        )
    } else {
        rolling_quantile_window_generic::<64, 4>(
            input_array,
            window_size,
            quantile
        )
    }
}

pub fn rolling_quantile_window_generic<const BLOCK_SIZE: usize, const SLICES_PER_BLOCK: usize>(
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

    let result_vec = quantile_window::rolling_window::<BLOCK_SIZE, SLICES_PER_BLOCK>(
        input_array,
        window_size,
        quantile
    );
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
