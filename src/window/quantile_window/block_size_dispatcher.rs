//! Selects the `BLOCK_SIZE` used by [`crate::rolling_quantile_window`], depending on the
//! requested window size.
//!
//! The block sizes below were determined by benchmarking; 16, 32 and 64
//! performed best. The benchmarks can be found in `benches/rolling.rs` and run
//! on your own machine with `cargo bench --bench rolling -- block_sizes_bench`.
//! Results vary between machines - the sizes chosen here were all measured on a
//! MacBook Air M3.
//!
//! The important point is that neither larger or smaller blocks are better in
//! general. The reasons are:
//!
//! 1. The linear searches and the shift performed inside a block, to locate the
//!    outgoing value and move the incoming one to its sorted position. Larger
//!    blocks mean more operations here.
//! 2. The depth of the tournament trees - the deeper they are, the more
//!    operations each lookup takes.
//! 3. Growing cache unfriendliness caused by a rising number of blocks, and with
//!    it a rising number of leaves in the tournament trees. Smaller blocks mean
//!    more blocks, and therefore more leaves, for the same window size.
//!
//! As described in the module header, the number of leaves is
//! rounded up to the next power of `K_ARY` so that the tournament
//! trees stay perfectly balanced. The calculation starts from the
//! number of blocks, which follows from the window size and the
//! given `BLOCK_SIZE`. Because of the rounding, the leaf count of a
//! tree always stays within a fixed range. In general the range between
//! two consecutive powers of `K_ARY`. Crossing such a range therefore
//! adds an entire new level at once. In terms of the window size
//! this gives:
//!
//! ```text
//!     window_size = K_ARY^k * B
//!
//!     K_ARY  the number of children per node
//!     k      the exponent, i.e. the number of levels
//!     B      the chosen BLOCK_SIZE
//! ```
//!
//! Since the three block sizes divide the same window differently -
//! `B16` produces twice as many blocks as `B32` - they cross into a
//! new range at different window sizes.
//!
//! The benchmarks show that point 2, the depth of the tournament
//! trees, is the decisive variable for performance. At equal depth,
//! however, point 1 alone determines which block size performs best,
//! which favours the smaller one. That effect does not hold without
//! limit: smaller blocks also mean more blocks, and correspondingly
//! more leaves in the tournament trees. Point 3 therefore gains
//! weight as the trees grow, until it outweighs the advantage of the
//! shorter distances inside a block. The point where this tips over
//! is described by the two constants [`BORDER_FOR_B16`] and
//! [`BORDER_FOR_B32`]. Each marks the point at which the
//! corresponding block size starts to lose against the next larger
//! one, even at equal depth. 64 turned out to be the largest useful
//! block size. [`get_suitable_std_block_size`] implements what is
//! described here.

use crate::window::quantile_window::{
    K_ARY,
    quantilewindow_tree_utils::tree_calculate_metadata,
    utils::calculate_needed_blocks
};

const BORDER_FOR_B16: usize = K_ARY.pow(5);
const BORDER_FOR_B32: usize = K_ARY.pow(6);

pub fn get_suitable_std_block_size(
    window_size: usize
) -> StdBlockSizes {
    let b16_metadata = StdBlockSizes::B16.calculate_metadata(window_size);
    let b32_metadata = StdBlockSizes::B32.calculate_metadata(window_size);
    let b64_metadata = StdBlockSizes::B64.calculate_metadata(window_size);

    if b16_metadata.needed_blocks > BORDER_FOR_B16 {
        if b32_metadata.needed_blocks > BORDER_FOR_B32 ||
            b64_metadata.needed_tree_buffer_length < b32_metadata.needed_tree_buffer_length {
            return StdBlockSizes::B64;
        }

        return StdBlockSizes::B32;
    }

    if b16_metadata.needed_tree_buffer_length <= b32_metadata.needed_tree_buffer_length {
        StdBlockSizes::B16
    } else {
        StdBlockSizes::B32
    }
}

struct BlockSizeMetaData {
    needed_blocks: usize,
    needed_tree_buffer_length: usize
}

impl BlockSizeMetaData {

    fn generate<const BLOCK_SIZE: usize>(
        window_size: usize
    ) -> Self {
        let needed_blocks = calculate_needed_blocks::<BLOCK_SIZE>(window_size);
        let needed_tree_buffer_length = tree_calculate_metadata(needed_blocks).1;

        Self {
            needed_blocks,
            needed_tree_buffer_length
        }
    }
}

pub enum StdBlockSizes {
    B16,
    B32,
    B64
}

impl StdBlockSizes {

    fn calculate_metadata(
        &self,
        window_size: usize
    ) -> BlockSizeMetaData {
        match self {
            StdBlockSizes::B16 => BlockSizeMetaData::generate::<16>(window_size),
            StdBlockSizes::B32 => BlockSizeMetaData::generate::<32>(window_size),
            StdBlockSizes::B64 => BlockSizeMetaData::generate::<64>(window_size)
        }
    }
}
