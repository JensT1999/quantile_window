use std::fmt::Display;

mod quantile_window;

#[derive(Debug)]
pub enum WindowError {
    SizingError,
}

impl Display for WindowError {

    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WindowError::SizingError => write!(f, "Error"),
        }
    }
}

impl std::error::Error for WindowError {}

pub fn rolling_quantile_window(input_array: &[f64], window_size: usize, quantile: f64) ->
    Result<Vec<f64>, WindowError> {
    if window_size == 0 {
        return Err(WindowError::SizingError);
    }

    let mut window = quantile_window::QuantileWindow::new(window_size, quantile);

    // Size of result vec = (input_length - window_size) / (steps (1)) + 1
    let mut result_vec = Vec::with_capacity((input_array.len() - window_size) + 1);

    // Initial fill
    let input_slice = &input_array[0..window_size];
    for value in input_slice {
        window.add(*value);
    }

    window.prepare();
    window.result_quantile(&mut result_vec);

    // update loop
    let input_slice = &input_array[window_size..];
    for (index, input) in input_slice.iter().enumerate() {
        if index == 28 {
            println!("test");
        }

        window.update_window(*input);
        window.adjust_and_result_quantile(&mut result_vec);
    }

    Ok(result_vec)
}
