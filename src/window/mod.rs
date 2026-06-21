use std::fmt::Display;

mod quantile_window;

const QUANTILE_EPSILON: f64 = 1e-9;

#[derive(Debug)]
pub enum WindowError {
    SizingError,
    InvalidQuantileError,
}

impl Display for WindowError {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowError::SizingError => write!(f, "SizingError: It seems you entered a wrong size"),
            WindowError::InvalidQuantileError => write!(f, "InvalidQuantileError:
                It seems you entered an invalid quantile"),
        }
    }
}

impl std::error::Error for WindowError {}

pub fn rolling_quantile_window(input_array: &[f64], window_size: usize, quantile: f64) ->
    Result<Vec<f64>, WindowError> {
    if input_array.len() == 0 || window_size == 0 {
        return Err(WindowError::SizingError);
    }

    if !valid_quantile(quantile) {
        return Err(WindowError::InvalidQuantileError);
    }

    let result_vec = quantile_window::rolling_window(input_array, window_size, quantile);
    Ok(result_vec)
}

fn valid_quantile(quantile: f64) -> bool {
    if quantile <= 0.0 || quantile > 1.0 {
        return false;
    }

    let scaled_up = quantile * 100.0;
    let computed = scaled_up - scaled_up.round();
    computed.abs() < QUANTILE_EPSILON
}
