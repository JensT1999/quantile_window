#![warn(clippy::undocumented_unsafe_blocks)]
#![warn(missing_docs)]
#![doc = include_str!("../README.md")]
mod window;
pub use window::{
    rolling_quantile_window,
    rolling_quantile_window_generic,
    WindowError,
    SizingErrorType
};
