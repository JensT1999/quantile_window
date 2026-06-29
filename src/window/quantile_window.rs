use std::{vec};

use crate::window::utils::{ordered_double::OrderedDouble,
    ordered_double_slice::OrderedDoubleSlice,
    sorting_networks,
    quantile_math};

const SORTING_NETWORK_SIZE: usize = 16;
const BLOCK_SIZE: usize = 64;
const SLICES_PER_BLOCK: usize = BLOCK_SIZE / SORTING_NETWORK_SIZE;
const K_ARY: usize = 8;

const PLACE_HOLDER_VALUE: OrderedDouble = OrderedDouble::MAX;
const PRED_DUMMY_VALUE: OrderedDouble = OrderedDouble::MIN;
const SUCC_DUMMY_VALUE: OrderedDouble = OrderedDouble::MAX;

struct QuantileWindow {
    current_size: usize,

    quantile: f64,
    searched_rank: f64,
    global_floor_rank: usize,
    actual_floor_rank: usize,
    actual_floor_value: OrderedDouble,
    actual_floor_big_block: usize,
    interpolation: bool,

    actual_block: usize,
    block_data: Vec<QuantileWindowBlock>,
    queue_data: Vec<OrderedDouble>,

    trees_leafs_starting_index: usize,
    pred_tree: Vec<QuantileWindowTreeNode>,
    succ_tree: Vec<QuantileWindowTreeNode>,
}

struct QuantileWindowBlock {
    data: [OrderedDouble; BLOCK_SIZE],
    length: usize,
    update_index: usize,
    tracker: usize,
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
        let mut current_slice = 0;
        while current_slice < SLICES_PER_BLOCK {
            slices_ptr[current_slice] = current_slice * SORTING_NETWORK_SIZE;
            current_slice += 1;
        }

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

                let slice_value = unsafe {
                    *temp_block_data.get_unchecked(*slice_ptr)
                };

                if slice_value <= smallest {
                    smallest = slice_value;
                    target_slice = index;
                    found = true;
                }
            }

            if !found {
                break;
            }

            unsafe {
                *self.data.get_unchecked_mut(k) = smallest;
            }
            slices_ptr[target_slice] += 1;
            k += 1;
        }
    }

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
    fn get_successor_value(&self, is_actual_floor_block: bool) -> OrderedDouble {
        let current_block_tracker = self.tracker;
        let succ_index = if is_actual_floor_block {
            current_block_tracker + 1
        } else if self.ran_out_right {
            self.length
        } else {
            current_block_tracker
        };

        if succ_index < self.length {
            unsafe {
                *self.data.get_unchecked(succ_index)
            }
        } else {
            SUCC_DUMMY_VALUE
        }
    }
}

#[derive(Clone, Copy)]
struct QuantileWindowTreeNode {
    value: OrderedDouble,
    block_index: usize,
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
    update_pred_heap: bool,
    update_succ_heap: bool,
}

impl QuantileWindow {

    fn new(window_size: usize, quantile: f64) -> QuantileWindow {
        let needed_blocks = calculate_needed_blocks(window_size);
        let needed_queue_size = window_size;
        let needed_trees_metadata = tree_calculate_metadata(needed_blocks);

        let mut result_window = QuantileWindow {
            current_size: 0,
            quantile,
            searched_rank: 0.0,
            global_floor_rank: 0,
            actual_floor_rank: 0,
            actual_floor_value: OrderedDouble::from_f64(0.0),
            actual_floor_big_block: 0,
            interpolation: false,
            actual_block: 0,
            block_data: Vec::with_capacity(needed_blocks),
            queue_data: vec![OrderedDouble::from_f64(0.0); needed_queue_size],
            trees_leafs_starting_index: needed_trees_metadata.0,
            pred_tree: vec![QuantileWindowTreeNode {
                value: PRED_DUMMY_VALUE,
                block_index: 0,
            }; needed_trees_metadata.1],
            succ_tree: vec![QuantileWindowTreeNode {
                value: SUCC_DUMMY_VALUE,
                block_index: 0,
            }; needed_trees_metadata.1],
        };

        let mut current_block = 0;
        while current_block < needed_blocks {
            result_window.block_data.push(QuantileWindowBlock {
                data: [PLACE_HOLDER_VALUE; BLOCK_SIZE],
                length: 0,
                update_index: 0,
                tracker: 0,
                ran_out_right: false
            });
            current_block += 1;
        }

        result_window
    }

    fn add(&mut self, value: OrderedDouble) {
        let current_block =
        if self.block_data[self.actual_block].length == BLOCK_SIZE {
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

    fn prepare(&mut self) {
        self.initial_sort();

        let searched_rank = self.quantile * ((self.current_size - 1) as f64);
        let floor_rank = searched_rank.floor() as usize;

        let selection_result = find_value_by_rank(&self.block_data,
            floor_rank);

        let global_floor_candidate = selection_result.selection_candidate;
        let mut global_floor_big_block_index = global_floor_candidate.block_index;
        initialize_block_tracker(&mut self.block_data, &global_floor_candidate);

        pred_tree_initial_build(&self.block_data,
            &mut self.pred_tree,
            self.trees_leafs_starting_index);
        succ_tree_initial_build(&self.block_data,
            &mut self.succ_tree,
            self.trees_leafs_starting_index,
            global_floor_big_block_index);

        // Muss definitiv noch in eigene methode
        let mut duplicates_to_skip = selection_result.duplicates_to_skip;
        while duplicates_to_skip > 0 {
            let new_floor_data = unsafe {
                *self.succ_tree.get_unchecked(0)
            };
            let new_floor_big_block_index = new_floor_data.block_index;

            let old_floor_big_block_index = global_floor_big_block_index;
            let old_floor_big_block = unsafe {
                self.block_data.get_unchecked_mut(old_floor_big_block_index)
            };

            old_floor_big_block.tracker += 1;
            if old_floor_big_block.tracker == old_floor_big_block.length {
                old_floor_big_block.ran_out_right = true;
            }

            pred_tree_update(old_floor_big_block,
                old_floor_big_block_index,
                &mut self.pred_tree,
                self.trees_leafs_starting_index);

            succ_tree_update(old_floor_big_block,
                old_floor_big_block_index,
                &mut self.succ_tree,
                self.trees_leafs_starting_index,
                new_floor_big_block_index);

            let new_floor_big_block = unsafe {
                self.block_data.get_unchecked(new_floor_big_block_index)
            };

            succ_tree_update(new_floor_big_block,
                new_floor_big_block_index,
                &mut self.succ_tree,
                self.trees_leafs_starting_index,
                new_floor_big_block_index);

            global_floor_big_block_index = new_floor_big_block_index;
            duplicates_to_skip -= 1;
        }

        self.searched_rank = searched_rank;
        self.global_floor_rank = floor_rank;
        self.actual_floor_rank = floor_rank;
        self.actual_floor_value = global_floor_candidate.block_value;
        self.actual_floor_big_block = global_floor_big_block_index;

        self.interpolation = !((searched_rank % 1.0) == 0.0);

        self.actual_block = 0;
    }

    fn initial_sort(&mut self) {
        for block in &mut self.block_data {
            block.sort();
        }
    }

    fn test_on_sorted(&self) -> bool {
        test_sorted_blocks(&self.block_data)
    }

    fn update_window(&mut self, new_value: OrderedDouble) {
        let actual_block_index = self.actual_block;
        let old_value = self.update_queue_get_old_value(new_value);
        let insertion_indizes = self.update_block_elements(new_value, old_value);

        let update_result = if actual_block_index == self.actual_floor_big_block {
            self.update_floor_block(new_value, insertion_indizes)
        } else {
            self.update_std_block(new_value, insertion_indizes)
        };

        let actual_block = unsafe {
            self.block_data.get_unchecked_mut(actual_block_index)
        };

        actual_block.tracker = update_result.new_tracker;
        actual_block.ran_out_right = update_result.ran_out_right;

        if update_result.update_pred_heap {
            pred_tree_update(actual_block,
                actual_block_index,
                &mut self.pred_tree,
                self.trees_leafs_starting_index);
        }

        if update_result.update_succ_heap {
            succ_tree_update(actual_block,
                actual_block_index,
                &mut self.succ_tree,
                self.trees_leafs_starting_index,
                self.actual_floor_big_block);
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
        let block_slice = unsafe {
            actual_block.data.get_unchecked_mut(0..actual_block.length)
        };

        let old_value_index = block_slice.iter().filter(|&&x| x < old_value).count();
        let new_value_index = if new_value > old_value {
            block_slice.shift_in_forwards(old_value_index, new_value)
        } else if new_value < old_value {
            block_slice.shift_in_backwards(old_value_index, new_value)
        } else {
            block_slice[old_value_index] = new_value;
            old_value_index
        };

        (old_value_index, new_value_index)
    }

    fn update_std_block(&mut self, new_value: OrderedDouble, insertion_indizes: (usize, usize)) ->
        QuantileWindowUpdateResult {
        let actual_block_index = self.actual_block;
        let actual_block = unsafe {
            self.block_data.get_unchecked(actual_block_index)
        };
        let deleted_index = insertion_indizes.0;
        let new_index = insertion_indizes.1;

        let mut update_pred_heap = false;
        let mut update_succ_heap = false;
        let actual_block_len = actual_block.length;
        let old_tracker = actual_block.tracker;
        let mut new_tracker = old_tracker;
        let mut ran_out_right = actual_block.ran_out_right;
        let mut update_tracker = true;

        if actual_block.ran_out_right {
            if new_value >= self.actual_floor_value {
                new_tracker = actual_block_len - 1;
                ran_out_right = false;

                self.actual_floor_rank -= 1;

                update_pred_heap = true;
                update_succ_heap = true;
                update_tracker = false;
            } else {
                if deleted_index == (actual_block.length - 1) ||
                    new_index == (actual_block.length - 1) {
                    update_pred_heap = true;
                    update_tracker = false;
                } else {
                    return QuantileWindowUpdateResult {
                        new_tracker,
                        ran_out_right,
                        update_pred_heap,
                        update_succ_heap
                    };
                }
            }
        }

        if !update_tracker {
            return QuantileWindowUpdateResult {
                new_tracker,
                ran_out_right,
                update_pred_heap,
                update_succ_heap
            };
        }

        if deleted_index < old_tracker {
            new_tracker -= 1;
            self.actual_floor_rank -= 1;
        }

        if old_tracker > 0 && deleted_index == old_tracker - 1 {
            update_pred_heap = true;
        }

        if deleted_index == old_tracker {
            update_succ_heap = true;
        }

        if new_index < new_tracker {
            new_tracker += 1;
            self.actual_floor_rank += 1;
        }

        if new_index == new_tracker {
            if new_value <= self.actual_floor_value {
                new_tracker += 1;
                self.actual_floor_rank += 1;
                update_pred_heap = true;
            } else {
                update_succ_heap = true;
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
            update_pred_heap,
            update_succ_heap
        }
    }

    fn update_floor_block(&mut self, new_value: OrderedDouble, insertion_indizes: (usize, usize)) ->
        QuantileWindowUpdateResult {
        let actual_block_index = self.actual_block;
        let actual_block = unsafe {
            self.block_data.get_unchecked(actual_block_index)
        };
        let deleted_index = insertion_indizes.0;
        let new_index = insertion_indizes.1;

        let mut update_pred_heap = false;
        let mut update_succ_heap = false;
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

                update_pred_heap = true;
                update_succ_heap = true;
                update_tracker = false;
            } else {
                if deleted_index == (actual_block.length - 1) ||
                    new_index == (actual_block.length - 1) {
                    update_pred_heap = true;
                    update_tracker = false;
                } else {
                    return QuantileWindowUpdateResult {
                        new_tracker,
                        ran_out_right,
                        update_pred_heap,
                        update_succ_heap
                    };
                }
            }
        }

        if !update_tracker {
            return QuantileWindowUpdateResult {
                new_tracker,
                ran_out_right,
                update_pred_heap,
                update_succ_heap
            };
        }

        if deleted_index < old_tracker {
            new_tracker -= 1;
            self.actual_floor_rank -= 1;
        }

        if old_tracker > 0 && deleted_index == old_tracker - 1 {
            update_pred_heap = true;
        }

        if deleted_index == old_tracker {
            self.update_global_to_successor();
        }

        if deleted_index == old_tracker + 1 {
            update_succ_heap = true;
        }

        if new_index < new_tracker {
            new_tracker += 1;
            self.actual_floor_rank += 1;
        }

        if new_index == new_tracker {
            if new_value <= self.actual_floor_value {
                new_tracker += 1;
                self.actual_floor_rank += 1;
                update_pred_heap = true;
            } else {
                update_succ_heap = true;
            }
        }

        if new_index == new_tracker + 1 {
            update_succ_heap = true;
        }

        if new_tracker == actual_block_len {
            ran_out_right = true;
        } else if new_tracker < actual_block_len {
            ran_out_right = false;
        }

        QuantileWindowUpdateResult {
            new_tracker,
            ran_out_right,
            update_pred_heap,
            update_succ_heap
        }
    }

    fn update_global_to_successor(&mut self) {
        let new_floor_data = unsafe {
            *self.succ_tree.get_unchecked(0)
        };

        let new_floor_big_block = unsafe {
            self.block_data.get_unchecked_mut(new_floor_data.block_index)
        };

        succ_tree_update(new_floor_big_block,
            new_floor_data.block_index,
            &mut self.succ_tree,
            self.trees_leafs_starting_index,
            new_floor_data.block_index);

        self.actual_floor_big_block = new_floor_data.block_index;
        self.actual_floor_value = new_floor_data.value;
    }

    fn result_quantile(&mut self) -> f64 {
        let floor_value: f64 = self.actual_floor_value.to_f64();
        if !self.interpolation {
            return floor_value;
        } else {
            let successor_value = unsafe {
                self.succ_tree.get_unchecked(0).value
            };
            let successor_value = successor_value.to_f64();

            return quantile_math::calculate_interpolated_quantile(self.searched_rank,
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
            let old_floor_big_block_index = self.actual_floor_big_block;
            let old_floor_big_block =  unsafe {
                self.block_data.get_unchecked_mut(old_floor_big_block_index)
            };
            if (old_floor_big_block.tracker + 1) == old_floor_big_block.length {
                old_floor_big_block.ran_out_right = true;
            } else {
                old_floor_big_block.tracker += 1;
            }

            let new_floor_data = unsafe {
                *self.succ_tree.get_unchecked(0)
            };

            pred_tree_update(old_floor_big_block,
                old_floor_big_block_index,
                &mut self.pred_tree,
                self.trees_leafs_starting_index);

            succ_tree_update(old_floor_big_block,
                old_floor_big_block_index,
                &mut self.succ_tree,
                self.trees_leafs_starting_index,
                new_floor_data.block_index);

            self.actual_floor_big_block = new_floor_data.block_index;
            self.actual_floor_value = new_floor_data.value;

            let new_floor_big_block = unsafe {
                self.block_data.get_unchecked(new_floor_data.block_index)
            };

            succ_tree_update(new_floor_big_block,
                new_floor_data.block_index,
                &mut self.succ_tree,
                self.trees_leafs_starting_index,
                new_floor_data.block_index);

            self.actual_floor_rank += 1;
        }
    }

    fn global_left_shift(&mut self) {
        while self.actual_floor_rank > self.global_floor_rank {
            let old_floor_big_block_index = self.actual_floor_big_block;
            let old_floor_big_block = unsafe {
                self.block_data.get_unchecked(old_floor_big_block_index)
            };

            let new_floor_data = unsafe {
                *self.pred_tree.get_unchecked(0)
            };

            succ_tree_update(old_floor_big_block,
                old_floor_big_block_index,
                &mut self.succ_tree,
                self.trees_leafs_starting_index,
                new_floor_data.block_index);

            self.actual_floor_big_block = new_floor_data.block_index;
            self.actual_floor_value = new_floor_data.value;

            let new_floor_big_block = unsafe {
                self.block_data.get_unchecked_mut(new_floor_data.block_index)
            };

            if new_floor_big_block.ran_out_right {
                new_floor_big_block.tracker = new_floor_big_block.length - 1;
                new_floor_big_block.ran_out_right = false;
            } else {
                new_floor_big_block.tracker -= 1;
            }

            pred_tree_update(new_floor_big_block,
                new_floor_data.block_index,
                &mut self.pred_tree,
                self.trees_leafs_starting_index);

            succ_tree_update(new_floor_big_block,
                new_floor_data.block_index,
                &mut self.succ_tree,
                self.trees_leafs_starting_index,
                new_floor_data.block_index);

            self.actual_floor_rank -= 1;
        }
    }
}

// Utils
fn calculate_needed_blocks(window_size: usize) -> usize {
    window_size.div_ceil(BLOCK_SIZE)
}

fn find_value_by_rank(block_data: &[QuantileWindowBlock], searched_rank: usize) -> QuantileWindowSelectionResult {
    let big_blocks_len = block_data.len();

    let mut selection_helpers: Vec<QuantileWindowSelectionHelper> = Vec::with_capacity(big_blocks_len);
    for (index, block) in block_data.iter().enumerate() {
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
        }; big_blocks_len];

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

fn initialize_block_tracker(block_data: &mut [QuantileWindowBlock],
    floor_value_canidate: &QuantileWindowSelectionCandidate) {
    for (index, block) in block_data.iter_mut().enumerate() {
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

// Trees

#[inline(always)]
fn tree_calculate_metadata(input_length: usize) -> (usize, usize) {
    let mut needed_length = K_ARY;
    while needed_length < input_length {
        needed_length *= K_ARY;
    }

    let result_length = ((needed_length - 1) / (K_ARY - 1)) + needed_length;
    let leafs_starting_index = result_length - needed_length;
    (leafs_starting_index, result_length)
}

fn pred_tree_initial_build(block_data: &[QuantileWindowBlock], pred_tree: &mut [QuantileWindowTreeNode],
    tree_leafs_starting_index: usize) {
    for (index, block) in block_data.iter().enumerate() {
        let pred_value = block.get_predeccessor_value();
        let target_index = tree_leafs_starting_index + index;
        unsafe {
            let current_node = pred_tree.get_unchecked_mut(target_index);
            current_node.value = pred_value;
            current_node.block_index = index;
        }
    }

    let mut current_index = tree_leafs_starting_index - 1;
    loop {
        let max_child_index = pred_tree_max_child(pred_tree, current_index);
        let max_node = unsafe {
            pred_tree.get_unchecked(max_child_index)
        };

        unsafe {
            *pred_tree.get_unchecked_mut(current_index) = *max_node;
        }

        if current_index == 0 {
            break;
        }

        current_index -= 1;
    }
}

fn pred_tree_max_child(pred_tree: &[QuantileWindowTreeNode], position: usize) -> usize {
    let first_child = tree_child_index(position, 1);

    let mut best = first_child;
    let mut current_node = unsafe {
        pred_tree.get_unchecked(best)
    };
    let mut best_value = current_node.value;

    for child in 1..K_ARY {
        let current_child_index = first_child + child;
        current_node = unsafe {
            pred_tree.get_unchecked(current_child_index)
        };

        if current_node.value > best_value {
            best = current_child_index;
            best_value = current_node.value;
        }
    }

    best
}

fn pred_tree_update(target_block: &QuantileWindowBlock, target_block_index: usize,
    pred_tree: &mut [QuantileWindowTreeNode], tree_leafs_starting_index: usize) {
    let pred_value = target_block.get_predeccessor_value();
    let mut current_index = tree_leafs_starting_index + target_block_index;
    let current_node = unsafe {
        let current_node = pred_tree.get_unchecked_mut(current_index);
        current_node.value = pred_value;
        *current_node
    };

    loop {
        let parent_index = tree_parent_index(current_index);
        let parent_node = unsafe {
            *pred_tree.get_unchecked(parent_index)
        };

        if current_node.block_index == parent_node.block_index {
            if current_node.value > parent_node.value {
                unsafe {
                    let parent_node = pred_tree.get_unchecked_mut(parent_index);
                    *parent_node = current_node;
                }
            } else {
                let max_child_index = pred_tree_max_child(pred_tree, parent_index);
                let max_node = unsafe {
                    *pred_tree.get_unchecked(max_child_index)
                };

                unsafe {
                    let parent_node = pred_tree.get_unchecked_mut(parent_index);
                    *parent_node = max_node;
                }
            }
        } else {
            if current_node.value > parent_node.value {
                unsafe {
                    let parent_node = pred_tree.get_unchecked_mut(parent_index);
                    *parent_node = current_node;
                }
            } else {
                break;
            }
        }

        current_index = parent_index;
        if current_index == 0 {
            break;
        }
    }
}

fn succ_tree_initial_build(block_data: &[QuantileWindowBlock], succ_tree: &mut [QuantileWindowTreeNode],
    tree_leafs_starting_index: usize, actual_floor_block: usize) {
    for (index, block) in block_data.iter().enumerate() {
        let is_actual_floor_block = index == actual_floor_block;
        let succ_value = block.get_successor_value(is_actual_floor_block);

        let target_index = tree_leafs_starting_index + index;
        unsafe {
            let current_node = succ_tree.get_unchecked_mut(target_index);
            current_node.value = succ_value;
            current_node.block_index = index;
        }
    }

    let mut current_index = tree_leafs_starting_index - 1;
    loop {
        let min_child_index = succ_tree_min_child(succ_tree, current_index);
        let min_node = unsafe {
            succ_tree.get_unchecked(min_child_index)
        };

        unsafe {
            *succ_tree.get_unchecked_mut(current_index) = *min_node;
        }

        if current_index == 0 {
            break;
        }

        current_index -= 1;
    }
}

fn succ_tree_min_child(succ_tree: &[QuantileWindowTreeNode], position: usize) -> usize {
    let first_child = tree_child_index(position, 1);

    let mut best = first_child;
    let mut current_node = unsafe {
        succ_tree.get_unchecked(best)
    };
    let mut best_value = current_node.value;

    for child in 1..K_ARY {
        let current_child_index = first_child + child;
        current_node = unsafe {
            succ_tree.get_unchecked(current_child_index)
        };

        if current_node.value < best_value {
            best = current_child_index;
            best_value = current_node.value;
        }
    }

    best
}

fn succ_tree_update(target_block: &QuantileWindowBlock, target_block_index: usize,
    succ_tree: &mut [QuantileWindowTreeNode], tree_leafs_starting_index: usize,
    actual_floor_block: usize) {
    let is_actual_floor_block = target_block_index == actual_floor_block;
    let succ_value = target_block.get_successor_value(is_actual_floor_block);

    let mut current_index = tree_leafs_starting_index + target_block_index;
    let current_node = unsafe {
        let current_node = succ_tree.get_unchecked_mut(current_index);
        current_node.value = succ_value;
        *current_node
    };

    loop {
        let parent_index = tree_parent_index(current_index);
        let parent_node = unsafe {
            succ_tree.get_unchecked_mut(parent_index)
        };

        if current_node.block_index == parent_node.block_index {
            if current_node.value < parent_node.value {
                unsafe {
                    let parent_node = succ_tree.get_unchecked_mut(parent_index);
                    *parent_node = current_node;
                }
            } else {
                let min_child_index = succ_tree_min_child(succ_tree, parent_index);
                let min_node = unsafe {
                    *succ_tree.get_unchecked(min_child_index)
                };

                unsafe {
                    let parent_node = succ_tree.get_unchecked_mut(parent_index);
                    *parent_node = min_node;
                }
            }
        } else {
            if current_node.value < parent_node.value {
                unsafe {
                    let parent_node = succ_tree.get_unchecked_mut(parent_index);
                    *parent_node = current_node;
                }
            } else {
                break;
            }
        }

        current_index = parent_index;
        if current_index == 0 {
            break;
        }
    }
}

#[inline(always)]
fn tree_child_index(position: usize, child_num: usize) -> usize {
    (position * K_ARY) + child_num
}

#[inline(always)]
fn tree_parent_index(position: usize) -> usize {
    (position - 1) / K_ARY
}

// Testing

fn test_sorted_blocks(block_data: &[QuantileWindowBlock]) -> bool {
    for block in block_data {
        let mut current_value = OrderedDouble::from_f64(-f64::NAN);
        for value in block.data {
            if value >= current_value {
                current_value = value;
            } else {
                return false;
            }
        }
    }

    true
}

// Main function
pub fn rolling_window(input_array: &[f64], window_size: usize, quantile: f64) -> Vec<f64> {
    let result_vec_len = (input_array.len() - window_size) + 1;
    let mut result_vec = Vec::with_capacity(result_vec_len);
    let mut window = QuantileWindow::new(window_size, quantile);

    let input_slice = &input_array[0..window_size];
    for value in input_slice {
        let input_value = OrderedDouble::from_f64(*value);
        window.add(input_value);
    }

    window.prepare();

    let first_result = window.result_quantile();
    result_vec.push(first_result);

    let input_slice = &input_array[window_size..];
    for input in input_slice {
        let input_value = OrderedDouble::from_f64(*input);
        window.update_window(input_value);

        let result = window.adjust_and_result_quantile();
        result_vec.push(result);
    }

    result_vec
}
