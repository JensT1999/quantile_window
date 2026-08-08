//! Internal implementation of the rolling quantile window.
//!
//! The public API consists solely of [`rolling_window`].
//! Everything else in this file is just an implementation detail.
//!
//! # Implementation Overview
//!
//! ## Block layout
//!
//! The input window is partitioned into fixed-size blocks of [`BLOCK_SIZE`]
//! elements. By default, [`BLOCK_SIZE`] is 64, although other values are
//! supported. The chosen block size must be a multiple of
//! [`SORTING_NETWORK_SIZE`], since blocks are initially sorted using a sorting
//! network.
//!
//! The default value of 64 was determined experimentally. Increasing
//! [`BLOCK_SIZE`] may reduce performance, since
//! [`QuantileWindow::update_block_elements`] performs a linear search within a
//! block to locate the element that has to be removed.
//!
//! ## Internal state
//!
//! The internal state is represented by [`QuantileWindow`], which owns
//! - an array of [`QuantileWindowBlock`]s,
//! - a queue storing the elements in insertion order, and
//! - two tournament trees used to maintain the global predecessor and successor
//!   candidates.
//!
//! The internal data structures are fully determined by the `window_size` passed
//! to [`rolling_window`]. This includes the number of blocks, the queue length,
//! and the size of the tournament trees. Everything is preallocated - No dynamic
//! allocations occur during the execution of [`rolling_window`].
//!
//! ## Tournament trees
//!
//! Every node of both tournament trees stores two pieces of information:
//! the candidate value contributed by a block and the index of the
//! corresponding [`QuantileWindowBlock`] within the internal block array.
//!
//! Two dedicated dummy values are used throughout the tournament trees:
//! [`PRED_DUMMY_VALUE`] (equal to [`OrderedDouble::MAX`]) and
//! [`SUCC_DUMMY_VALUE`] (equal to [`OrderedDouble::MIN`]). These values ensure
//! that invalid blocks are always propagated to the bottom of the predecessor
//! and successor trees, respectively.
//!
//! In addition, [`TREE_INVALID_BLOCK_IDX`] is used to distinguish valid block
//! references from invalid ones. A block is considered invalid by the
//! tournament trees if it is in the `ran_out_right` state or if the node
//! corresponds to one of the padding nodes required because the tournament
//! trees are always represented as perfect k-ary trees.
//!
//! Both tournament trees are implemented as `K_ARY`-ary trees. Consequently,
//! the number of leaf nodes is always rounded up to the next power of
//! `K_ARY`, ensuring that both trees remain perfectly balanced.
//!
//! ## Execution phases
//!
//! The [`rolling_window`] computation consists of two phases:
//!   1. The initial fill phase
//!   2. The update phase
//!
//! ## Initial fill phase
//!
//! During the initial fill phase, incoming elements are distributed block by
//! block. Each block is filled until it contains [`BLOCK_SIZE`] elements before
//! the next block is used. Once the last block has been filled, the preparation
//! for the update phase begins.
//!
//! The preparation phase begins by sorting every [`QuantileWindowBlock`]. This
//! is achieved by combining the previously mentioned sorting network with a
//! k-way merge algorithm.
//!
//! Once all blocks are sorted, the initial global `floor_value` is determined
//! by [`QuantileWindow::find_value_by_rank`]. The `floor_value` is defined as
//! the lower interpolation value, i.e. the value corresponding to the floored
//! rank of the requested quantile. If the requested rank is an integer, the
//! `floor_value` is equal to the exact global quantile. The requested rank is
//! computed as `quantile * (current_size - 1)`. Since the window size is fixed
//! during computation, `current_size` is equal to the configured window size
//! once the preparation phase has completed.
//!
//! The initial `floor_value` is found using an algorithm inspired by the
//! median-of-medians selection algorithm, operating directly on the sorted
//! [`QuantileWindowBlock`] instances.
//!
//! After determining the `floor_value`, together with the block containing it
//! and its local index, the block trackers are initialized. A block tracker
//! marks the local lower bound of the global `floor_value`. In other words, it
//! separates the values of a block into those that would appear to the left of
//! the global `floor_value` in the globally sorted order and those that would
//! appear to its right.
//!
//! Next, both tournament trees are initialized. Every block contributes one
//! successor candidate and one predecessor candidate:
//!
//! - The successor tree receives the value at the block's tracker position.
//!   The only exception is the block containing the current `floor_value`,
//!   which contributes the value at `tracker + 1`.
//! - The predecessor tree always receives the value at `tracker - 1`.
//!
//! Finally, duplicate values are handled. Since the median-of-medians based
//! selection may return a `floor_value` located in the middle of a sequence of
//! duplicate values, the global position has to be shifted to the first valid
//! occurrence.
//!
//! During [`QuantileWindow::initialize_block_tracker`], the tracker of the
//! floor block is therefore moved to the beginning of the local sequence of
//! duplicates. In addition, [`QuantileWindow::find_value_by_rank`] returns the
//! number of duplicate positions that still have to be skipped relative to the
//! requested rank and the global lower bound of the `floor_value`.
//!
//! This shift is then performed iteratively using the successor tree until the
//! remaining number of duplicate positions to skip reaches zero. In each
//! iteration, the root of the successor tree yields the next occurrence of the
//! global `floor_value`. The metadata stored alongside the successor value
//! identifies the corresponding block, which becomes the new floor block.
//!
//! During each shift, the tournament trees are updated for both the previous
//! (`old_floor_block`) and the new (`new_floor_block`) floor block. In
//! particular, the tracker of the `old_floor_block` advances by one position,
//! causing its successor and predecessor candidates to be updated accordingly.
//!
//! //! ## Update phase
//!
//! During the update phase, old values are continuously replaced by newly
//! incoming values. Similar to the initial fill phase, the update process
//! starts with the first [`QuantileWindowBlock`].
//!
//! Each [`QuantileWindowBlock`] is updated for exactly the number of elements
//! defined by its internal length before the next block is processed. Once the
//! last block has been completely updated, the process starts again with the
//! first block. Therefore, the update process behaves similarly to a circular
//! buffer.
//!
//! The removed value is obtained from the insertion-order queue. Its position is
//! determined by the current block index multiplied by [`BLOCK_SIZE`] plus the
//! internal update index of the block. This update index tracks how many
//! elements of the corresponding block have already been replaced.
//!
//! The main challenge during the update phase is maintaining the global
//! quantile position after local block modifications. The block tracker
//! separates each block into a left and right partition relative to the global
//! `floor_value`.
//!
//! Changes in the distribution of values between these partitions cause the
//! tracker to move. For example, removing a value from the left partition and
//! inserting a value into the right partition shifts the tracker one position
//! to the left. The opposite operation shifts the tracker one position to the
//! right.
//!
//! Every local tracker movement is translated into a global window-level delta.
//! This delta tracking represents the cumulative effect of all local block
//! modifications on the global quantile position. The resulting shift depends
//! on the relative positions of the removed and inserted values within the
//! block.
//!
//! The exact tracker adjustments performed during a block update are implemented
//! in [`QuantileWindow::update_block`].
//!
//! After the local block update has completed, the previously accumulated global
//! shift has to be compensated. This is performed by
//! [`QuantileWindow::global_right_shift`] and
//! [`QuantileWindow::global_left_shift`].
//!
//! These functions shift the global `floor_value` until its current rank matches
//! the original target rank calculated during the preparation phase again. The
//! performed shifts are conceptually similar to the duplicate correction shifts
//! described previously.
//!
//! The difference is that [`QuantileWindow::global_left_shift`] traverses the
//! roots of the predecessor tree, while [`QuantileWindow::global_right_shift`]
//! traverses the roots of the successor tree.
//!
//! The implementation maintains several invariants:
//!
//! 1. The number of elements stored in each [`QuantileWindowBlock`] is immutable.
//! 2. All data buffers of [`QuantileWindowBlock`] instances are sorted and remain
//!    sorted during the complete computation.
//! 3. Every [`QuantileWindowBlock`] contains a tracker that points to the local
//!    floor position of the block with respect to the global `floor_value`.
//! 4. A tracker can move beyond the right boundary of its block if all values
//!    contained in the block are smaller than the global `floor_value`. This
//!    state is called `ran_out_right`.
//! 5. A tracker never points to memory outside the allocated block buffer.
//!    Out-of-range tracker positions are represented by sentinel states.
//! 6. Each block contributes a successor candidate to the global successor tree.
//!    For every block, this candidate is the value at the block's tracker position.
//!    The only exception is the block containing the current global `floor_value`,
//!    where the candidate is taken from the position immediately after the tracker
//!    (`tracker + 1`).
//! 7. Each block contributes a predecessor candidate to the global predecessor
//!    tree. This candidate always corresponds to the value at the position
//!    immediately before the block's tracker (`tracker - 1`).
//! 8. The global successor tree maintains the minimum of all successor values
//!    provided by the individual blocks.
//! 9. The global predecessor tree maintains the maximum of all predecessor values
//!    provided by the individual blocks.

use std::{marker::PhantomData, vec};

use crate::window::utils::{
    ordered_double::OrderedDouble,
    sorting_networks,
};

const SORTING_NETWORK_SIZE: usize = 16;
const BLOCK_SIZE: usize = 64;
const SLICES_PER_BLOCK: usize = BLOCK_SIZE / SORTING_NETWORK_SIZE;
const K_ARY: usize = 8;

const PLACE_HOLDER_VALUE: OrderedDouble = OrderedDouble::MAX;
const PRED_DUMMY_VALUE: OrderedDouble = OrderedDouble::MIN;
const SUCC_DUMMY_VALUE: OrderedDouble = OrderedDouble::MAX;
const TREE_INVALID_BLOCK_IDX: usize = usize::MAX;

struct QuantileWindow {
    current_size: usize,

    quantile: f64,
    searched_rank: f64,
    global_floor_rank: usize,
    actual_floor_rank: usize,
    actual_floor_value: OrderedDouble,
    actual_floor_block_index: usize,
    interpolation: bool,

    actual_block: usize,
    block_data: Vec<QuantileWindowBlock>,
    queue_data: Vec<OrderedDouble>,

    pred_tree: QuantileWindowTree<QuantileWindowPredeccessorTree>,
    succ_tree: QuantileWindowTree<QuantileWindowSuccessorTree>,
}

#[derive(Clone)]
struct QuantileWindowBlock {
    data: [OrderedDouble; BLOCK_SIZE],
    length: usize,
    update_index: usize,
    tracker: usize,
    actual_floor_block: bool,
    ran_out_right: bool,
}

impl QuantileWindowBlock {

    fn sort(&mut self) {
        let mut current_slice = 0;
        while current_slice < SLICES_PER_BLOCK {
            let slice_start_index = current_slice * SORTING_NETWORK_SIZE;
            let slice_end_index = slice_start_index + SORTING_NETWORK_SIZE;
            let block_slice = &mut self.data[slice_start_index..slice_end_index];

            sorting_networks::sorting_network_16(block_slice);
            current_slice += 1;
        }

        self.k_way_merge_slices();
    }

    fn k_way_merge_slices(&mut self) {
        let mut temp_block_data = [OrderedDouble::from_f64(0.0); BLOCK_SIZE];
        temp_block_data.copy_from_slice(&self.data);

        let mut slices_ptr = [0; SLICES_PER_BLOCK];
        slices_ptr
            .iter_mut()
            .enumerate()
            .for_each(|item| {
                *item.1 = item.0 * SORTING_NETWORK_SIZE;
            });

        let mut k = 0;
        loop {
            let mut smallest = OrderedDouble::MAX;
            let mut target_slice = 0;
            let mut found = false;

            for (index, slice_ptr) in slices_ptr.iter().enumerate() {
                let max_ptr = (index * SORTING_NETWORK_SIZE) + SORTING_NETWORK_SIZE;
                if *slice_ptr == max_ptr {
                    continue;
                }

                let slice_value = temp_block_data[*slice_ptr];
                if slice_value <= smallest {
                    smallest = slice_value;
                    target_slice = index;
                    found = true;
                }
            }

            if !found {
                break;
            }

            self.data[k] = smallest;
            slices_ptr[target_slice] += 1;
            k += 1;
        }
    }

    #[inline(always)]
    fn tracker_forwards(&mut self) {
        if (self.tracker + 1) == self.length {
            self.ran_out_right = true;
        } else {
            self.tracker += 1;
        }
    }

    #[inline(always)]
    fn tracker_backwards(&mut self) {
        if self.ran_out_right {
            self.tracker = self.length - 1;
            self.ran_out_right = false;
        } else {
            self.tracker -= 1;
        }
    }

    // SAFETY: First of all the predeccessor marks the value of `current_block_tracker` - 1.
    // So there could be two potential cases where the tracker runs out of the underlying data buffer:
    // 1. `current_block_tracker` is equal to zero. In this case `PRED_DUMMY_VALUE` will be returned
    // and no access on the underlying data buffer occurs.
    // 2. If the block is `ran_out_right`: `self.length` is guaranteed to be >= 1.
    // After the initial fill of the window the length of the blocks is immutable. The number of elements
    // in the block is completely deterministic. Even if the block is in `ran_out_right` state there will
    // be always one element minimum. Otherwise the block wont even exist, because in the construction of the
    // window the needed blocks are calculated depending on the deterministic size of the window.
    #[inline(always)]
    fn get_predeccessor_value(&self) -> OrderedDouble {
        let current_block_tracker = self.tracker;
        if self.ran_out_right {
            unsafe {
                *self.data.get_unchecked(self.length - 1)
            }
        } else if current_block_tracker == 0 {
            PRED_DUMMY_VALUE
        } else {
            unsafe {
                *self.data.get_unchecked(current_block_tracker - 1)
            }
        }
    }

    #[inline(always)]
    fn is_predeccessor_out_of_range(&self) -> bool {
        let current_block_tracker = self.tracker;
        current_block_tracker == 0
    }

    // SAFETY: First of all the successor marks the value of `current_block_tracker`. If the focused block
    // is equal to the global floor block (e.g. the block that contains the actual floor value that is
    // being tracker) the successor marks the value of `current_block_tracker` + 1.
    // Additionally it is important to understand that the `actual_floor_block` wont ever turn into
    // `ran_out_right` state. The `ran_out_right` state marks a block where all elements contained by this block
    // are sorted on the left side of the current global floor value.
    // So there are only two cases where the tracker could be outside of the valid range of the underlying
    // data buffer:
    // 1. We are focusing the floor block and our tracker is equals to `self.length` - 1:
    // In this case the following if clause would return false, because the `succ_index` would be equal
    // to `self.length`. So `SUCC_DUMMY_VALUE` would be returned and no access on the underlying data buffer
    // occurs.
    // 2. We are focusing a block, which is in `ran_out_right` state:
    // As you can see in the function `get_successor_index` length of the block will be returned if the block
    // is in `ran_out_right` state. Also in this case the if clause would return false, because `succ_index` would
    // be equal to `self.length` and `SUCC_DUMMY_VALUE` would be returned.
    #[inline(always)]
    fn get_successor_value(&self) -> OrderedDouble {
        let succ_index = self.get_successor_index();
        if succ_index < self.length {
            unsafe {
                *self.data.get_unchecked(succ_index)
            }
        } else {
            SUCC_DUMMY_VALUE
        }
    }

    #[inline(always)]
    fn is_successor_out_of_range(&self) -> bool {
        let succ_index = self.get_successor_index();
        if succ_index < self.length {
            return false;
        }

        true
    }

    #[inline(always)]
    fn get_successor_index(&self) -> usize {
        let current_block_tracker = self.tracker;
        if self.actual_floor_block {
            current_block_tracker + 1
        } else if self.ran_out_right {
            self.length
        } else {
            current_block_tracker
        }
    }

    #[inline(always)]
    fn set_actual_floor_block(&mut self, is_actual_floor_block: bool) {
        self.actual_floor_block = is_actual_floor_block;
    }
}

struct QuantileWindowSelectionHelper<'a> {
    block_index: usize,
    tracker: usize,
    tracker_low: usize,
    tracker_high: usize,
    data_slice: &'a [OrderedDouble],
    tracker_value: OrderedDouble,
    invalid: bool,
}

#[derive(Clone, Copy)]
struct QuantileWindowSelectionCandidate {
    block_index: usize,
    block_tracker: usize,
    block_value: OrderedDouble,
}

struct QuantileWindowSelectionResult {
    selection_candidate: QuantileWindowSelectionCandidate,
    duplicates_to_skip: usize,
}

struct QuantileWindowUpdateResult {
    new_tracker: usize,
    ran_out_right: bool,
    update_pred_tree: bool,
    update_succ_tree: bool,
}

impl QuantileWindow {

    fn new(window_size: usize, quantile: f64) -> QuantileWindow {
        let needed_blocks = quantilewindow_utils::calculate_needed_blocks(window_size);
        let needed_queue_size = window_size;
        let needed_trees_metadata = quantilewindow_tree_utils::
            tree_calculate_metadata(needed_blocks);

        let result_window = QuantileWindow {
            current_size: 0,
            quantile,
            searched_rank: 0.0,
            global_floor_rank: 0,
            actual_floor_rank: 0,
            actual_floor_value: OrderedDouble::default(),
            actual_floor_block_index: 0,
            interpolation: false,
            actual_block: 0,
            block_data: vec![QuantileWindowBlock {
                data: [PLACE_HOLDER_VALUE; BLOCK_SIZE],
                length: 0,
                update_index: 0,
                tracker: 0,
                actual_floor_block: false,
                ran_out_right: false
            }; needed_blocks],
            queue_data: vec![OrderedDouble::default(); needed_queue_size],
            pred_tree: QuantileWindowTree::<QuantileWindowPredeccessorTree>::new(needed_trees_metadata),
            succ_tree: QuantileWindowTree::<QuantileWindowSuccessorTree>::new(needed_trees_metadata),
        };

        result_window
    }

    fn add(&mut self, value: OrderedDouble) {
        let current_block = if self.block_data[self.actual_block].length == BLOCK_SIZE {
            self.actual_block += 1;
            &mut self.block_data[self.actual_block]
        } else {
            &mut self.block_data[self.actual_block]
        };

        current_block.data[current_block.length] = value;
        self.queue_data[self.current_size] = value;
        current_block.length += 1;
        self.current_size += 1;
    }

    // Prepare
    fn prepare(&mut self) {
        self.initial_sort();

        let searched_rank = self.quantile * ((self.current_size - 1) as f64);
        let floor_rank = searched_rank.floor() as usize;

        let selection_result = self.find_value_by_rank(floor_rank);
        let global_floor_candidate = selection_result.selection_candidate;
        self.initialize_block_tracker( global_floor_candidate);

        self.pred_tree.initialize_tree(&self.block_data);
        self.succ_tree.initialize_tree(&self.block_data);

        // Initializing window metadata
        self.searched_rank = searched_rank;
        self.global_floor_rank = floor_rank;
        self.actual_floor_rank = floor_rank;
        self.actual_floor_value = global_floor_candidate.block_value;
        self.actual_floor_block_index = global_floor_candidate.block_index;
        self.interpolation = !((searched_rank % 1.0) == 0.0);

        let duplicates_to_skip = selection_result.duplicates_to_skip;
        self.skip_duplicates(duplicates_to_skip);

        // Setting floor block
        let target_floor_block = unsafe {
            self.block_data.get_unchecked_mut(self.actual_floor_block_index)
        };
        target_floor_block.set_actual_floor_block(true);

        self.actual_block = 0;
    }

    fn initial_sort(&mut self) {
        self.block_data
            .iter_mut()
            .for_each(|block| block.sort());
    }

    fn find_value_by_rank(&self, searched_rank: usize) -> QuantileWindowSelectionResult {
        let blocks_len = self.block_data.len();

        let mut selection_helpers: Vec<QuantileWindowSelectionHelper> = Vec::with_capacity(blocks_len);
        for (index, block) in self.block_data.iter().enumerate() {
            selection_helpers.push(QuantileWindowSelectionHelper {
                block_index: index,
                tracker: 0,
                tracker_high: block.length,
                tracker_low: 0,
                data_slice: &block.data,
                tracker_value: OrderedDouble::from_f64(0.0),
                invalid: false,
            });
        }

        let mut pivot_candidates: Vec<QuantileWindowSelectionCandidate> = vec![
            QuantileWindowSelectionCandidate {
                block_index: 0,
                block_tracker: 0,
                block_value: OrderedDouble::from_f64(0.0)
            }; blocks_len];

        loop {
            let mut active_blocks = 0;
            for selection_helper in &mut selection_helpers {
                if selection_helper.invalid {
                    continue;
                }

                let tracker_calculation =
                    |low, high| (low + (high - 1)) / 2;
                let tracker_index =
                    tracker_calculation(selection_helper.tracker_low, selection_helper.tracker_high);
                selection_helper.tracker = tracker_index;

                let tracker_value = selection_helper.data_slice[tracker_index];
                selection_helper.tracker_value = tracker_value;

                pivot_candidates[active_blocks].block_index = selection_helper.block_index;
                pivot_candidates[active_blocks].block_tracker = tracker_index;
                pivot_candidates[active_blocks].block_value = tracker_value;
                active_blocks += 1;
            }

            let candidates_slice = &mut pivot_candidates[0..active_blocks];
            let pivot_element =
                candidates_slice.select_nth_unstable_by(active_blocks / 2,
                |a, b| a.block_value.cmp(&b.block_value)).1;

            let mut global_lower_bound = 0;
            let mut global_upper_bound = 0;
            for selection_helper in &selection_helpers {
                if selection_helper.invalid {
                    continue;
                }

                let target_slice =
                    &selection_helper.data_slice[selection_helper.tracker_low..selection_helper.tracker_high];

                let pivot_lower_bound = target_slice
                    .partition_point(|&x| x < pivot_element.block_value);
                global_lower_bound += pivot_lower_bound;

                let pivot_upper_bound = target_slice
                    .partition_point(|&x| x <= pivot_element.block_value);
                global_upper_bound += pivot_upper_bound;
            }

            for selection_helper in &selection_helpers {
                global_lower_bound += selection_helper.tracker_low;
                global_upper_bound += selection_helper.tracker_low;
            }

            if searched_rank >= global_lower_bound && searched_rank < global_upper_bound {
                let duplicates_to_skip = searched_rank - global_lower_bound;
                return QuantileWindowSelectionResult {
                    selection_candidate: *pivot_element,
                    duplicates_to_skip
                };
            } else if global_lower_bound < searched_rank {
                for selection_helper in &mut selection_helpers {
                    if selection_helper.invalid {
                        continue;
                    }

                    let tracker_value = selection_helper.tracker_value;
                    if tracker_value <= pivot_element.block_value {
                        let new_lower = selection_helper.tracker + 1;
                        selection_helper.tracker_low = new_lower;

                        if new_lower == selection_helper.tracker_high {
                            selection_helper.invalid = true;
                        }
                    }
                }
            } else if global_lower_bound > searched_rank {
                for selection_helper in &mut selection_helpers {
                    if selection_helper.invalid {
                        continue;
                    }

                    let tracker_value = selection_helper.tracker_value;
                    if tracker_value >= pivot_element.block_value {
                        let new_high = selection_helper.tracker;
                        selection_helper.tracker_high = new_high;

                        if new_high == selection_helper.tracker_low {
                            selection_helper.invalid = true;
                        }
                    }
                }
            }
        }
    }

    fn initialize_block_tracker(&mut self, floor_value_canidate: QuantileWindowSelectionCandidate) {
        for (index, block) in self.block_data.iter_mut().enumerate() {
            // This if clause seems to be necessary for performance.
            // Gemini told me that without the if clause the compiler seems to generate to complex code.
            if index == floor_value_canidate.block_index {
                let mut block_tracker = floor_value_canidate.block_tracker;
                let tracker_value = block.data[block_tracker];
                // Duplicate skipping intern
                while block_tracker > 0 && tracker_value == block.data[block_tracker - 1] {
                    block_tracker -= 1;
                }

                block.tracker = block_tracker;
                block.set_actual_floor_block(true);
                continue;
            }

            let lower_bound_of_floor_value =
                block.data.partition_point(|&x| x < floor_value_canidate.block_value);
            if lower_bound_of_floor_value == block.length {
                block.ran_out_right = true;
            }

            block.tracker = lower_bound_of_floor_value;
        }
    }

    // SAFETY: First of all this function belongs to the funcion named `prepare`. So this marks one
    // step of the initial phase of the window. There could append only one case where the unsafe
    // blocks would lead to undefined behavior:
    // `old_floor_block_index` or `new_floor_block_index` are out of range of the `self.block_data` buffer:
    // If the function is called the very first time `old_floor_block_index` will be the result of the function
    // `find_value_by_rank`. Short explanation: This function uses a median of medians approach on the sorted blocks
    // to determine the initial floor value. In conclusion the first `old_floor_block_index` will always be a valid
    // index.
    // The `new_floor_block_index` comes from the root of the so called `succ_tree`. Short explanation: The
    // `succ_tree` is a tournament tree containing the minimum value of all successors of the current floor value.
    // It tracks down the value and the `block_index` of the block where the values comes from.
    // The tree is initialized after the initial floor value determination on all blocks with the
    // corresponding block indexes.
    fn skip_duplicates(&mut self, duplicates_to_skip: usize) {
        let mut duplicates_to_skip = duplicates_to_skip;
        while duplicates_to_skip > 0 {
            let new_floor_data = self.succ_tree.get_root();

            // Old floor block adjustment
            let old_floor_block_index = self.actual_floor_block_index;
            let old_floor_block = unsafe {
                self.block_data.get_unchecked_mut(old_floor_block_index)
            };
            old_floor_block.tracker_forwards();
            self.pred_tree.update_tree(old_floor_block,
                old_floor_block_index);
            self.succ_tree.update_tree(old_floor_block,
                old_floor_block_index);

            // New floor block adjustment
            let new_floor_block_index = new_floor_data.block_index;
            let new_floor_block = unsafe {
                self.block_data.get_unchecked(new_floor_block_index)
            };
            self.succ_tree.update_tree(new_floor_block,
                new_floor_block_index);

            self.actual_floor_block_index = new_floor_block_index;
            duplicates_to_skip -= 1;
        }
    }

    fn update_window(&mut self, new_value: OrderedDouble) {
        let actual_block_index = self.actual_block;
        let old_value = self.update_queue_get_old_value(new_value);
        let insertion_indizes = self.update_block_elements(new_value, old_value);

        let update_result = if actual_block_index == self.actual_floor_block_index {
            self.update_block::<true>(new_value, insertion_indizes)
        } else {
            self.update_block::<false>(new_value, insertion_indizes)
        };

        let actual_block = unsafe {
            self.block_data.get_unchecked_mut(actual_block_index)
        };

        actual_block.tracker = update_result.new_tracker;
        actual_block.ran_out_right = update_result.ran_out_right;

        if update_result.update_pred_tree {
            self.pred_tree.update_tree(actual_block,
                actual_block_index);
        }

        if update_result.update_succ_tree {
            self.succ_tree.update_tree(actual_block,
                actual_block_index);
        }

        if (actual_block.update_index + 1) == actual_block.length {
            actual_block.update_index = 0;
            self.actual_block = if (self.actual_block + 1) == self.block_data.len() {
                0
            } else {
                self.actual_block + 1
            }
        } else {
            actual_block.update_index += 1;
        }
    }

    fn update_queue_get_old_value(&mut self, new_value: OrderedDouble) -> OrderedDouble {
        let actual_block_index = self.actual_block;
        let actual_block = unsafe {
            self.block_data.get_unchecked(actual_block_index)
        };

        let queue_index = (actual_block_index * BLOCK_SIZE) + actual_block.update_index;
        let queue_ref = unsafe {
            self.queue_data.get_unchecked_mut(queue_index)
        };

        let old_value = *queue_ref;
        *queue_ref = new_value;

        old_value
    }

    fn update_block_elements(&mut self, new_value: OrderedDouble, old_value: OrderedDouble) -> (usize, usize) {
        let actual_block_index = self.actual_block;
        let actual_block = unsafe {
            self.block_data.get_unchecked_mut(actual_block_index)
        };
        let block_slice = &mut actual_block.data[0..actual_block.length];
        let old_value_index = block_slice.iter().filter(|&&x| x < old_value).count();
        let new_value_index = if new_value > old_value {
            quantilewindow_utils::shift_in_forwards(block_slice, old_value_index, new_value)
        } else if new_value < old_value {
            quantilewindow_utils::shift_in_backwards(block_slice, old_value_index, new_value)
        } else {
            block_slice[old_value_index] = new_value;
            old_value_index
        };

        (old_value_index, new_value_index)
    }

    fn update_block<const IS_FLOOR_BLOCK: bool>(&mut self, new_value: OrderedDouble,
        insertion_indizes: (usize, usize)) -> QuantileWindowUpdateResult {
        let actual_block_index = self.actual_block;
        let actual_block = unsafe {
            self.block_data.get_unchecked(actual_block_index)
        };
        let deleted_index = insertion_indizes.0;
        let new_index = insertion_indizes.1;

        let mut update_pred_tree = false;
        let mut update_succ_tree = false;
        let actual_block_len = actual_block.length;
        let old_tracker = actual_block.tracker;
        let mut new_tracker = old_tracker;
        let mut ran_out_right = actual_block.ran_out_right;
        let mut update_tracker = true;

        if actual_block.ran_out_right {
            if new_value >= self.actual_floor_value {
                new_tracker = new_index;
                ran_out_right = false;

                self.actual_floor_rank -= 1;

                update_pred_tree = true;
                update_succ_tree = true;
                update_tracker = false;
            } else {
                if deleted_index == (actual_block_len - 1) ||
                    new_index == (actual_block_len - 1) {
                    update_pred_tree = true;
                    update_tracker = false;
                } else {
                    return QuantileWindowUpdateResult {
                        new_tracker,
                        ran_out_right,
                        update_pred_tree,
                        update_succ_tree
                    };
                }
            }
        }

        if !update_tracker {
            return QuantileWindowUpdateResult {
                new_tracker,
                ran_out_right,
                update_pred_tree,
                update_succ_tree
            };
        }

        if deleted_index < old_tracker {
            new_tracker -= 1;
            self.actual_floor_rank -= 1;
        }

        if old_tracker > 0 && deleted_index == old_tracker - 1 {
            update_pred_tree = true;
        }

        if deleted_index == old_tracker {
            if IS_FLOOR_BLOCK {
                self.update_global_to_successor();
            } else {
                update_succ_tree = true;
            }
        }

        if IS_FLOOR_BLOCK {
            if deleted_index == old_tracker + 1 {
                update_succ_tree = true;
            }
        }

        if new_index < new_tracker {
            new_tracker += 1;
            self.actual_floor_rank += 1;
        }

        if new_index == new_tracker {
            if new_value <= self.actual_floor_value {
                new_tracker += 1;
                self.actual_floor_rank += 1;
                update_pred_tree = true;
            } else {
                update_succ_tree = true;
            }
        }

        if IS_FLOOR_BLOCK {
            if new_index == new_tracker + 1 {
                update_succ_tree = true;
            }
        }

        if new_tracker == actual_block_len {
            ran_out_right = true;
        } else if new_tracker < actual_block_len {
            ran_out_right = false;
        }

        QuantileWindowUpdateResult {
            new_tracker,
            ran_out_right,
            update_pred_tree,
            update_succ_tree
        }
    }

    fn update_global_to_successor(&mut self) {
        // Adjustment of old floor block
        let old_floor_block = unsafe {
            self.block_data.get_unchecked_mut(self.actual_floor_block_index)
        };
        old_floor_block.set_actual_floor_block(false);

        // Adjustment of new floor block
        let new_floor_data = self.succ_tree.get_root();
        let new_floor_block = unsafe {
            self.block_data.get_unchecked_mut(new_floor_data.block_index)
        };
        new_floor_block.set_actual_floor_block(true);
        self.succ_tree.update_tree(new_floor_block,
            new_floor_data.block_index);

        self.actual_floor_block_index = new_floor_data.block_index;
        self.actual_floor_value = new_floor_data.value;
    }

    fn result_quantile(&mut self) -> f64 {
        let floor_value: f64 = self.actual_floor_value.to_f64();
        if !self.interpolation {
            return floor_value;
        } else {
            let successor_value = self.succ_tree.get_root().value;
            let successor_value = successor_value.to_f64();

            return quantilewindow_utils::calculate_interpolated_quantile(self.searched_rank,
                floor_value,
                successor_value);
        }
    }

    fn adjust_and_result_quantile(&mut self) -> f64 {
        self.global_right_shift();
        self.global_left_shift();
        self.result_quantile()
    }

    fn global_right_shift(&mut self) {
        while self.actual_floor_rank < self.global_floor_rank {
            let new_floor_data = self.succ_tree.get_root();

            // Adjustment of old Floor Block
            let old_floor_block_index = self.actual_floor_block_index;
            let old_floor_block =  unsafe {
                self.block_data.get_unchecked_mut(old_floor_block_index)
            };
            old_floor_block.tracker_forwards();
            old_floor_block.set_actual_floor_block(false);
            self.pred_tree.update_tree(old_floor_block,
                old_floor_block_index);
            self.succ_tree.update_tree(old_floor_block,
                old_floor_block_index);

            // Adjustment of new Floor Block
            let new_floor_block = unsafe {
                self.block_data.get_unchecked_mut(new_floor_data.block_index)
            };
            new_floor_block.set_actual_floor_block(true);
            self.succ_tree.update_tree(new_floor_block,
                new_floor_data.block_index);

            // Adjustment of window
            self.actual_floor_block_index = new_floor_data.block_index;
            self.actual_floor_value = new_floor_data.value;
            self.actual_floor_rank += 1;
        }
    }

    fn global_left_shift(&mut self) {
        while self.actual_floor_rank > self.global_floor_rank {
            let new_floor_data = self.pred_tree.get_root();

            // Old Floor block
            let old_floor_block_index = self.actual_floor_block_index;
            let old_floor_block = unsafe {
                self.block_data.get_unchecked_mut(old_floor_block_index)
            };
            old_floor_block.set_actual_floor_block(false);
            self.succ_tree.update_tree(old_floor_block,
                old_floor_block_index);

            // New floor block
            let new_floor_block = unsafe {
                self.block_data.get_unchecked_mut(new_floor_data.block_index)
            };
            new_floor_block.tracker_backwards();
            new_floor_block.set_actual_floor_block(true);
            self.pred_tree.update_tree(new_floor_block,
                new_floor_data.block_index);
            self.succ_tree.update_tree(new_floor_block,
                new_floor_data.block_index);

            // Window adjustment
            self.actual_floor_block_index = new_floor_data.block_index;
            self.actual_floor_value = new_floor_data.value;
            self.actual_floor_rank -= 1;
        }
    }
}

// Utils
mod quantilewindow_utils {
    use crate::window::{quantile_window::BLOCK_SIZE, utils::ordered_double::OrderedDouble};

    #[inline(always)]
    pub fn calculate_needed_blocks(window_size: usize) -> usize {
        window_size.div_ceil(BLOCK_SIZE)
    }

    #[inline(always)]
    pub fn calculate_interpolated_quantile(searched_rank: f64, floor_value: f64, successor_value: f64) -> f64 {
        floor_value + (successor_value - floor_value) * (searched_rank - searched_rank.floor())
    }

    #[inline(always)]
    pub fn shift_in_backwards(data: &mut [OrderedDouble], old_value_index: usize, new_value: OrderedDouble) -> usize {
        debug_assert!(!data.is_empty());
        debug_assert!(old_value_index < data.len());

        let mut index = old_value_index;
        unsafe {
            while (index > 0) && (new_value < *data.get_unchecked(index - 1)) {
                *data.get_unchecked_mut(index) = *data.get_unchecked(index - 1);
                index -= 1;
            }
            *data.get_unchecked_mut(index) = new_value;
        }

        index
    }

    #[inline(always)]
    pub fn shift_in_forwards(data: &mut [OrderedDouble], old_value_index: usize, new_value: OrderedDouble) -> usize {
        debug_assert!(!data.is_empty());
        debug_assert!(old_value_index < data.len());

        let mut index = old_value_index;
        unsafe {
            while (index < (data.len() - 1)) && (new_value > *data.get_unchecked(index + 1)) {
                *data.get_unchecked_mut(index) = *data.get_unchecked(index + 1);
                index += 1;
            }
            *data.get_unchecked_mut(index) = new_value;
        }

        index
    }
}
// Tree

trait QuantileWindowTreeFunctions<T>
where
    T: QuantileWindowTreeType {
    fn new(tree_metadata: (usize, usize)) -> QuantileWindowTree<T>;
    fn initialize_tree(&mut self, block_data: &[QuantileWindowBlock]);
    fn select_real_following_child(&self, position: usize) -> QuantileWindowTreeNode;
    fn select_following_child(&self, position: usize) -> usize;
    fn update_tree(&mut self, target_block: &QuantileWindowBlock, target_block_index: usize);
    fn get_root(&self) -> QuantileWindowTreeNode;
}

mod quantilewindow_tree_utils {
    use crate::window::quantile_window::K_ARY;

    #[inline(always)]
    pub fn tree_calculate_metadata(input_length: usize) -> (usize, usize) {
        let mut needed_length = K_ARY;
        while needed_length < input_length {
            needed_length *= K_ARY;
        }

        let result_length = ((needed_length - 1) / (K_ARY - 1)) + needed_length;
        let leafs_starting_index = result_length - needed_length;
        (leafs_starting_index, result_length)
    }

    #[inline(always)]
    pub fn tree_child_index(position: usize, child_num: usize) -> usize {
        (position * K_ARY) + child_num
    }

    #[inline(always)]
    pub fn tree_parent_index(position: usize) -> usize {
        (position - 1) / K_ARY
    }
}

#[derive(Clone, Copy)]
struct QuantileWindowTreeNode {
    value: OrderedDouble,
    block_index: usize,
}

struct QuantileWindowTree<T>
where
    T: QuantileWindowTreeType {
    data: Vec<QuantileWindowTreeNode>,
    trees_leaf_starting_index: usize,
    _marker: PhantomData<T>,
}

trait QuantileWindowTreeType {
    const DUMMY_VALUE: OrderedDouble;

    fn get_neighbor_info(target_block: &QuantileWindowBlock) -> (bool, OrderedDouble);
    fn is_better(value_one: OrderedDouble, value_two: OrderedDouble) -> bool;
}

struct QuantileWindowPredeccessorTree;
impl QuantileWindowTreeType for QuantileWindowPredeccessorTree {
    const DUMMY_VALUE: OrderedDouble = PRED_DUMMY_VALUE;

    #[inline(always)]
    fn get_neighbor_info(target_block: &QuantileWindowBlock) -> (bool, OrderedDouble) {
        let predeccessor_out_of_range = target_block.is_predeccessor_out_of_range();
        let predeccessor_value = target_block.get_predeccessor_value();

        (predeccessor_out_of_range, predeccessor_value)
    }

    #[inline(always)]
    fn is_better(value_one: OrderedDouble, value_two: OrderedDouble) -> bool {
        value_one > value_two
    }
}

struct QuantileWindowSuccessorTree;
impl QuantileWindowTreeType for QuantileWindowSuccessorTree {
    const DUMMY_VALUE: OrderedDouble = SUCC_DUMMY_VALUE;

    #[inline(always)]
    fn get_neighbor_info(target_block: &QuantileWindowBlock) -> (bool, OrderedDouble) {
        let successor_out_of_range = target_block.is_successor_out_of_range();
        let successor_value = target_block.get_successor_value();

        (successor_out_of_range, successor_value)
    }

    #[inline(always)]
    fn is_better(value_one: OrderedDouble, value_two: OrderedDouble) -> bool {
        value_one < value_two
    }
}

impl <T> QuantileWindowTree<T>
where
    T: QuantileWindowTreeType {

    #[inline(always)]
    fn tree_select_node(tree: &[QuantileWindowTreeNode], position: usize) -> usize {
        let first_child = quantilewindow_tree_utils::tree_child_index(position, 1);
        let mut result = first_child;

        for child in 1..K_ARY {
            let target_child_index = first_child + child;
            let child_node = unsafe {
                *tree.get_unchecked(target_child_index)
            };

            if child_node.block_index != TREE_INVALID_BLOCK_IDX {
                result = target_child_index;
            }
        }

        result
    }
}

impl<T> QuantileWindowTreeFunctions<T> for QuantileWindowTree<T>
where
    T: QuantileWindowTreeType {

    fn new(tree_metadata: (usize, usize)) -> QuantileWindowTree<T> {
        QuantileWindowTree {
            data: vec![QuantileWindowTreeNode {
                value: T::DUMMY_VALUE,
                block_index: TREE_INVALID_BLOCK_IDX,
            }; tree_metadata.1],
            trees_leaf_starting_index: tree_metadata.0,
            _marker: PhantomData,
        }
    }

    fn initialize_tree(&mut self, block_data: &[QuantileWindowBlock]) {
        for (index, block) in block_data.iter().enumerate() {
            let neighbor_info = T::get_neighbor_info(block);
            let tree_block_index = if neighbor_info.0 {
                TREE_INVALID_BLOCK_IDX
            } else {
                index
            };

            let target_index = self.trees_leaf_starting_index + index;
            let current_node = &mut self.data[target_index];
            current_node.value = neighbor_info.1;
            current_node.block_index = tree_block_index;
        }

        let mut current_index = self.trees_leaf_starting_index - 1;
        loop {
            let target_node = self.select_real_following_child(current_index);
            unsafe {
                *self.data.get_unchecked_mut(current_index) = target_node;
            }

            if current_index == 0 {
                break;
            }

            current_index -= 1;
        }
    }

    fn select_real_following_child(&self, position: usize) -> QuantileWindowTreeNode {
        let target_child_index = self.select_following_child(position);
        let target_node = unsafe {
            *self.data.get_unchecked(target_child_index)
        };

        if target_node.block_index != TREE_INVALID_BLOCK_IDX {
            return target_node;
        }

        let target_child_index = Self::tree_select_node(&self.data, position);
        unsafe {
            *self.data.get_unchecked(target_child_index)
        }
    }

    fn select_following_child(&self, position: usize) -> usize {
        let first_child = quantilewindow_tree_utils::tree_child_index(position, 1);

        let mut best = first_child;
        let mut best_value = unsafe {
            self.data.get_unchecked(best).value
        };

        for child in 1..K_ARY {
            let current_child_index = first_child + child;
            let current_node = unsafe {
                self.data.get_unchecked(current_child_index)
            };

            if T::is_better(current_node.value, best_value) {
                best = current_child_index;
                best_value = current_node.value;
            }
        }

        best
    }

    fn update_tree(&mut self, target_block: &QuantileWindowBlock, target_block_index: usize) {
        let target_node_index = self.trees_leaf_starting_index + target_block_index;
        let neighbor_info = T::get_neighbor_info(target_block);
        let tree_block_index = if neighbor_info.0 {
            TREE_INVALID_BLOCK_IDX
        } else {
            target_block_index
        };

        let current_node = unsafe {
            let current_node = self.data.get_unchecked_mut(target_node_index);
            current_node.value = neighbor_info.1;
            current_node.block_index = tree_block_index;
            *current_node
        };

        let mut current_index = target_node_index;
        loop {
            let parent_index = quantilewindow_tree_utils::tree_parent_index(current_index);
            let mut selected_node = unsafe {
                *self.data.get_unchecked(parent_index)
            };

            if selected_node.block_index == target_block_index {
                if T::is_better(current_node.value, selected_node.value) {
                    selected_node = current_node;
                } else {
                    let target_node = self.select_real_following_child(parent_index);
                    selected_node = target_node;
                }
            } else {
                if selected_node.block_index == TREE_INVALID_BLOCK_IDX ||
                    T::is_better(current_node.value, selected_node.value) {
                    selected_node = current_node;
                } else {
                    break;
                }
            }

            unsafe {
                *self.data.get_unchecked_mut(parent_index) = selected_node;
            }

            current_index = parent_index;
            if current_index == 0 {
                break;
            }
        }
    }

    fn get_root(&self) -> QuantileWindowTreeNode {
        self.data[0]
    }
}

// Main function
pub fn rolling_window(input_array: &[f64], window_size: usize, quantile: f64) -> Vec<f64> {
    let result_vec_len = (input_array.len() - window_size) + 1;
    let mut result_vec = Vec::with_capacity(result_vec_len);

    let input_slice = &input_array[0..window_size];
    let mut window = input_slice
        .iter()
        .fold(QuantileWindow::new(window_size, quantile),
            |mut window, value| {
                window.add(OrderedDouble::from_f64(*value));
                window
            });

    window.prepare();
    let first_result = window.result_quantile();
    result_vec.push(first_result);

    let input_slice = &input_array[window_size..];
    input_slice
        .iter()
        .for_each(|value| {
            let input_value = OrderedDouble::from_f64(*value);
            window.update_window(input_value);

            let result = window.adjust_and_result_quantile();
            result_vec.push(result);
        });

    result_vec
}

#[cfg(test)]
mod tests {
    use rand::{RngExt, SeedableRng, rngs::StdRng};
    use super::*;

    // Testing forwards and backwards shift in
    #[test]
    fn test_backwards_shift_complete_shift() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 1.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            quantilewindow_utils::shift_in_backwards(&mut test_input, 4, test_value)
        };

        let result_data = [1.0, 2.0, 3.0, 4.0, 5.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 0);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    fn test_backwards_shift_middle_shift_in() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 3.5;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            quantilewindow_utils::shift_in_backwards(&mut test_input, 4, test_value)
        };

        let result_data = [2.0, 3.0, 3.5, 4.0, 5.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 2);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    fn test_backwards_shift_no_shift_in() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 7.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            quantilewindow_utils::shift_in_backwards(&mut test_input, 4, test_value)
        };

        let result_data = [2.0, 3.0, 4.0, 5.0, 7.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 4);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    fn test_forwards_shift_complete_shift() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 7.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            quantilewindow_utils::shift_in_forwards(&mut test_input, 0, test_value)
        };

        let result_data = [3.0, 4.0, 5.0, 6.0, 7.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 4);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    fn test_forwards_shift_middle_shift_in() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 3.5;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            quantilewindow_utils::shift_in_forwards(&mut test_input, 0, test_value)
        };

        let result_data = [3.0, 3.5, 4.0, 5.0, 6.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 1);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    fn test_forwards_shift_no_shift_in() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 1.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            quantilewindow_utils::shift_in_forwards(&mut test_input, 0, test_value)
        };

        let result_data = [1.0, 3.0, 4.0, 5.0, 6.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 0);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    #[should_panic]
    fn test_backwards_shift_in_empty_input() {
        let mut input_data = [];

        let input_value = 1.0;
        let test_value = OrderedDouble::from_f64(input_value);

        quantilewindow_utils::shift_in_backwards(&mut input_data, 0, test_value);
    }

    #[test]
    #[should_panic]
    fn test_forwards_shift_in_empty_input() {
        let mut input_data = [];

        let input_value = 1.0;
        let test_value = OrderedDouble::from_f64(input_value);

        quantilewindow_utils::shift_in_forwards(&mut input_data, 0, test_value);
    }

    #[test]
    #[should_panic]
    fn test_backwards_shift_in_old_index_bigger_as_len() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);
        let input_value = 1.0;
        let test_value = OrderedDouble::from_f64(input_value);

        quantilewindow_utils::shift_in_backwards(&mut test_input, 5, test_value);
    }

    #[test]
    #[should_panic]
    fn test_forwards_shift_in_old_index_bigger_as_len() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 7.0;
        let test_value = OrderedDouble::from_f64(input_value);

        quantilewindow_utils::shift_in_forwards(&mut test_input, 5, test_value);
    }

    #[track_caller]
    fn turn_into_ordered_double_vec(input_array: &[f64]) -> Vec<OrderedDouble> {
        input_array
            .iter()
            .map(|x| OrderedDouble::from_f64(*x))
            .collect::<Vec<OrderedDouble>>()
    }

    const RAND_TESTING_SEED: u64 = 109;
    const QUANTILEWINDOW_TEST_SIZE: usize = 1000;
    const QUANTILEWINDOW_TEST_QUANTILE: f64 = 0.01;

    #[test]
    fn test_calculate_needed_blocks() {
        let expected_needed_blocks = QUANTILEWINDOW_TEST_SIZE.div_ceil(BLOCK_SIZE);
        let needed_blocks = quantilewindow_utils::
            calculate_needed_blocks(QUANTILEWINDOW_TEST_SIZE);

        assert_eq!(expected_needed_blocks, needed_blocks);
    }

    #[test]
    fn test_needed_trees_metadata_calc() {
        let needed_blocks = quantilewindow_utils::
            calculate_needed_blocks(QUANTILEWINDOW_TEST_SIZE);
        let needed_trees_metadata = quantilewindow_tree_utils::
            tree_calculate_metadata(needed_blocks);

        let (expected_leafs_starting_index, expected_tree_length) = needed_trees_metadata;
        assert!(expected_leafs_starting_index < expected_tree_length);

        let tree_base = expected_tree_length - expected_leafs_starting_index;
        assert!((tree_base % K_ARY) == 0);

        let tree_left_part_length = expected_leafs_starting_index;
        assert!(((tree_left_part_length - 1) % K_ARY) == 0);
    }

    #[test]
    fn test_quantilewindow_new() {
        let expected_actual_floor_value = OrderedDouble::from_f64(0.0);
        let expected_window_blocks = quantilewindow_utils::
            calculate_needed_blocks(QUANTILEWINDOW_TEST_SIZE);
        let expected_queue_length = QUANTILEWINDOW_TEST_SIZE;
        let expected_tree_metadata = quantilewindow_tree_utils::
            tree_calculate_metadata(expected_window_blocks);
        let (expected_leafs_starting_index, expected_tree_length) = expected_tree_metadata;

        let test_window = QuantileWindow::
            new(QUANTILEWINDOW_TEST_SIZE, QUANTILEWINDOW_TEST_QUANTILE);

        assert!(test_window.current_size == 0);
        assert!(test_window.quantile == QUANTILEWINDOW_TEST_QUANTILE);
        assert!(test_window.searched_rank == 0.0);
        assert!(test_window.global_floor_rank == 0);
        assert!(test_window.actual_floor_rank == 0);
        assert!(test_window.actual_floor_value == expected_actual_floor_value);
        assert!(test_window.actual_floor_block_index == 0);
        assert!(test_window.interpolation == false);
        assert!(test_window.actual_block == 0);

        assert!(!test_window.block_data.is_empty());
        assert!(test_window.block_data.len() == expected_window_blocks);
        test_blocks(&test_window.block_data);

        assert!(!test_window.queue_data.is_empty());
        assert!(test_window.queue_data.len() == expected_queue_length);

        assert!(!test_window.pred_tree.data.is_empty());
        assert!(test_window.pred_tree.data.len() == expected_tree_length);
        assert!(test_window.pred_tree.trees_leaf_starting_index == expected_leafs_starting_index);
        test_tree_nodes::<QuantileWindowPredeccessorTree>(&test_window.pred_tree);

        assert!(!test_window.succ_tree.data.is_empty());
        assert!(test_window.succ_tree.data.len() == expected_tree_length);
        assert!(test_window.succ_tree.trees_leaf_starting_index == expected_leafs_starting_index);
        test_tree_nodes::<QuantileWindowSuccessorTree>(&test_window.succ_tree);
    }

    #[track_caller]
    fn test_blocks(blocks_data: &[QuantileWindowBlock]) {
        for block in blocks_data {
            test_block_data_array(&block.data);
            assert!(block.length == 0);
            assert!(block.update_index == 0);
            assert!(block.tracker == 0);
            assert!(block.actual_floor_block == false);
            assert!(block.ran_out_right == false);
        }
    }

    #[track_caller]
    fn test_block_data_array(block_data: &[OrderedDouble]) {
        for value in block_data {
            assert!(*value == PLACE_HOLDER_VALUE);
        }
    }

    #[track_caller]
    fn test_tree_nodes<T: QuantileWindowTreeType>(tree: &QuantileWindowTree<T>) {
        for node in &tree.data {
            assert!(node.value == T::DUMMY_VALUE);
            assert!(node.block_index == TREE_INVALID_BLOCK_IDX);
        }
    }

    #[test]
    fn test_quantilewindow_add() {
        let test_input = (0..QUANTILEWINDOW_TEST_SIZE)
            .map(|x| x as f64)
            .collect::<Vec<f64>>();

        let mut test_window = QuantileWindow::
            new(QUANTILEWINDOW_TEST_SIZE, QUANTILEWINDOW_TEST_QUANTILE);

        let input_slice = &test_input[0..BLOCK_SIZE];
        for input in input_slice {
            let ordered_representation = OrderedDouble::from_f64(*input);
            test_window.add(ordered_representation);
        }

        assert!(test_window.actual_block == 0);
        let current_block = &test_window.block_data[test_window.actual_block];
        assert!(current_block.length == BLOCK_SIZE);

        let input = test_input[BLOCK_SIZE];
        let ordered_representation = OrderedDouble::from_f64(input);
        test_window.add(ordered_representation);
        assert!(test_window.actual_block == 1);

        let input_slice = &test_input[(BLOCK_SIZE + 1)..];
        for input in input_slice {
            let ordered_representation = OrderedDouble::from_f64(*input);
            test_window.add(ordered_representation);
        }

        let result_blocks_length = calculate_blocks_length(&test_window.block_data);
        assert!(result_blocks_length == QUANTILEWINDOW_TEST_SIZE);
        assert!(test_window.current_size == QUANTILEWINDOW_TEST_SIZE);
    }

    #[track_caller]
    fn calculate_blocks_length(block_data: &[QuantileWindowBlock]) -> usize {
        let mut result = 0;
        for block in block_data {
            result += block.length;
        }

        result
    }

    #[test]
    fn test_quantilewindow_prepare_initial_sort() {
        let mut rng = StdRng::seed_from_u64(RAND_TESTING_SEED);
        let test_input = (0..QUANTILEWINDOW_TEST_SIZE)
            .map(|_| rng.random_range(0.0..100.0))
            .collect::<Vec<f64>>();

        let mut test_window = QuantileWindow::
            new(QUANTILEWINDOW_TEST_SIZE, QUANTILEWINDOW_TEST_QUANTILE);

        for input in test_input {
            let ordered_representation = OrderedDouble::from_f64(input);
            test_window.add(ordered_representation);
        }

        test_window.prepare();
        test_blocks_on_sorted(&test_window.block_data);
    }

    #[track_caller]
    fn test_blocks_on_sorted(blocks_data: &[QuantileWindowBlock]) {
        for block in blocks_data {
            assert!(block.data.is_sorted());
        }
    }

    #[test]
    fn test_quantilewindow_prepare_floor_selection_initialization() {
        let mut rng = StdRng::seed_from_u64(RAND_TESTING_SEED);
        let mut test_input = (0..QUANTILEWINDOW_TEST_SIZE)
            .map(|_| rng.random_range(0.0..100.0))
            .collect::<Vec<f64>>();

        let mut test_window = QuantileWindow::
            new(QUANTILEWINDOW_TEST_SIZE, QUANTILEWINDOW_TEST_QUANTILE);

        let input_slice = &test_input[0..];
        for input in input_slice {
            let ordered_representation = OrderedDouble::from_f64(*input);
            test_window.add(ordered_representation);
        }

        test_window.prepare();

        let expected_searched_rank = QUANTILEWINDOW_TEST_QUANTILE * ((QUANTILEWINDOW_TEST_SIZE - 1) as f64);
        let expected_floor_rank = expected_searched_rank.floor() as usize;
        let (left_side, expected_floor_value, right_side) =
            test_input.select_nth_unstable_by(expected_floor_rank, |x, y| x.total_cmp(y));

        let expected_predeccessor_value =
            left_side.iter().max_by(|x, y| x.total_cmp(y));
        let result_predeccessor_value = test_window.pred_tree.get_root().value;
        match expected_predeccessor_value {
            Some(value) => {
                assert_eq!(result_predeccessor_value, OrderedDouble::from_f64(*value))
            },
            None => {
                assert!(result_predeccessor_value == PRED_DUMMY_VALUE);
                let predeccessor_block_index = test_window.pred_tree.get_root().block_index;
                assert!(predeccessor_block_index == TREE_INVALID_BLOCK_IDX);
            }
        }

        let expected_successor_value =
            right_side.iter().min_by(|x, y| x.total_cmp(y));
        let result_successor_value = test_window.succ_tree.get_root().value;
        match expected_successor_value {
            Some(value) => {
                assert_eq!(result_successor_value, OrderedDouble::from_f64(*value));
            },
            None => {
                assert!(result_successor_value == PRED_DUMMY_VALUE);
                let successor_block_index = test_window.succ_tree.get_root().block_index;
                assert!(successor_block_index == TREE_INVALID_BLOCK_IDX);
            }
        }

        assert!(test_window.searched_rank == expected_searched_rank);
        assert!(test_window.global_floor_rank == expected_floor_rank);
        assert!(test_window.actual_floor_rank == expected_floor_rank);

        let result_floor_value = test_window.actual_floor_value;
        let transformed_floor_value = OrderedDouble::to_f64(test_window.actual_floor_value);
        assert!(transformed_floor_value == *expected_floor_value);

        let floor_value_block = &test_window.block_data[test_window.actual_floor_block_index];
        let min_block_value = floor_value_block.data[0];
        let max_block_value = floor_value_block.data[floor_value_block.length - 1];
        assert!(min_block_value <= result_floor_value);
        assert!(max_block_value >= result_floor_value);
        assert!(floor_value_block.data[0..floor_value_block.length].contains(&result_floor_value));

        let expected_interpolation = !((expected_searched_rank % 1.0) == 0.0);
        assert!(test_window.interpolation == expected_interpolation);
    }
}
