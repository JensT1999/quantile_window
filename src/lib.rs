#![warn(clippy::undocumented_unsafe_blocks)]
mod window;
pub use window::{
    rolling_quantile_window,
    rolling_quantile_window_generic,
    WindowError
};
