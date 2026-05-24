use std::{vec};

const SORTING_NETWORK_SIZE: usize = 16;
const BIG_BLOCK_SIZE: usize = 64;
const K_ARY_HEAP: usize = 8;

const PRED_DUMMY_VALUE: f64 = -f64::INFINITY;
const SUCC_DUMMY_VALUE: f64 = f64::INFINITY;

pub struct QuantileWindow {
    size: usize,
    current_size: usize,

    quantile: f64,
    searched_rank: f64,
    global_floor_rank: usize,
    actual_floor_rank: usize,
    actual_floor_value: f64,
    actual_floor_big_block: usize,
    interpolation: bool,

    actual_block: usize,
    block_data: Vec<QuantileWindowBigBlock>,
    queue_data: Vec<f64>,

    pred_heap: Vec<QuantileWindowHeapNode>,
    pred_heap_block_indizes: Vec<usize>,
    succ_heap: Vec<QuantileWindowHeapNode>,
    succ_heap_block_indizes: Vec<usize>,
}

pub struct QuantileWindowBigBlock {
    data: [f64; BIG_BLOCK_SIZE],
    length: usize,
    update_index: usize,
    tracker: usize,
    ran_out_right: bool,
}

#[derive(Clone, Copy)]
struct QuantileWindowHeapNode {
    value: f64,
    block_index: usize,
}

struct QuantileWindowSelectionHelper<'a> {
    block_index: usize,
    tracker: usize,
    tracker_low: usize,
    tracker_high: usize,
    data_slice: &'a [f64],
    tracker_value: f64,
    invalid: bool,
}

#[derive(Clone, Copy)]
struct QuantileWindowSelectionCandidate {
    block_index: usize,
    block_tracker: usize,
    block_value: f64,
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

    pub fn new(window_size: usize, quantile: f64) -> QuantileWindow {
        let needed_blocks = calculate_needed_big_blocks(window_size);
        let needed_queue_size = window_size;

        // Exp
        let needed_heap_blocks = (needed_blocks + K_ARY_HEAP - 1) / K_ARY_HEAP;
        let needed_heaps_size = K_ARY_HEAP * needed_heap_blocks + 1;

        let mut result_window = QuantileWindow {
            size: window_size,
            current_size: 0,
            quantile,
            searched_rank: 0.0,
            global_floor_rank: 0,
            actual_floor_rank: 0,
            actual_floor_value: 0.0,
            actual_floor_big_block: 0,
            interpolation: false,
            actual_block: 0,
            block_data: Vec::with_capacity(needed_blocks),
            queue_data: vec![0.0; needed_queue_size],
            pred_heap: vec![QuantileWindowHeapNode { block_index: 0, value: PRED_DUMMY_VALUE }; needed_heaps_size],
            pred_heap_block_indizes: vec![0; needed_heaps_size],
            succ_heap: vec![QuantileWindowHeapNode { block_index: 0, value: SUCC_DUMMY_VALUE }; needed_heaps_size],
            succ_heap_block_indizes: vec![0; needed_heaps_size],
        };

        let mut current_block = 0;
        while current_block < needed_blocks {
            result_window.block_data.push(QuantileWindowBigBlock {
                data: [f64::INFINITY; BIG_BLOCK_SIZE],
                length: 0,
                update_index: 0,
                tracker: 0,
                ran_out_right: false
            });
            current_block += 1;
        }

        result_window
    }

    pub fn add(&mut self, value: f64) {
        let current_block =
        if self.block_data[self.actual_block].length == BIG_BLOCK_SIZE {
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

    pub fn prepare(&mut self) {
        initial_sort(&mut self.block_data);
        merge_to_big_blocks(&mut self.block_data);

        let searched_rank = self.quantile * ((self.current_size - 1) as f64);
        let floor_rank = searched_rank.floor() as usize;

        let selection_result =
            find_global_floor_value_by_rank(&self.block_data ,floor_rank);

        let global_floor_candidate = selection_result.selection_candidate;
        let mut global_floor_big_block_index = global_floor_candidate.block_index;
        initialize_block_tracker(&mut self.block_data, &global_floor_candidate);

        pred_heap_initial_build(&mut self.block_data,
            &mut self.pred_heap,
            &mut self.pred_heap_block_indizes);
        succ_heap_initial_build(&mut self.block_data,
            &mut self.succ_heap,
            &mut self.succ_heap_block_indizes,
            global_floor_big_block_index);

        let mut duplicates_to_skip = selection_result.duplicates_to_skip;
        while duplicates_to_skip > 0 {
            let new_floor_data = unsafe {
                *self.succ_heap.get_unchecked(0)
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

            pred_heap_update(old_floor_big_block,
                old_floor_big_block_index,
                &mut self.pred_heap,
                &mut self.pred_heap_block_indizes);

            succ_heap_update(old_floor_big_block,
                old_floor_big_block_index,
                &mut self.succ_heap,
                &mut self.succ_heap_block_indizes,
                new_floor_big_block_index);

            let new_floor_big_block = unsafe {
                self.block_data.get_unchecked(new_floor_big_block_index)
            };

            succ_heap_update(new_floor_big_block,
                new_floor_big_block_index,
                &mut self.succ_heap,
                &mut self.succ_heap_block_indizes,
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

    pub fn test_on_sorted(&self) -> bool {
        test_sorted_blocks(&self.block_data)
    }

    pub fn update_window(&mut self, new_value: f64) {
        let actual_block_index = self.actual_block;
        let insertion_indizes = self.update_block_elements(new_value);

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
            pred_heap_update(actual_block,
                actual_block_index,
                &mut self.pred_heap,
                &mut self.pred_heap_block_indizes);
        }

        if update_result.update_succ_heap {
            succ_heap_update(actual_block,
                actual_block_index,
                &mut self.succ_heap,
                &mut self.succ_heap_block_indizes,
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

    fn update_block_elements(&mut self, new_value: f64) -> (usize, usize) {
        let actual_block_index = self.actual_block;
        let result_inidizes = unsafe {
            let actual_block = self.block_data.get_unchecked_mut(actual_block_index);
            let block_slice = actual_block.data.get_unchecked_mut(0..actual_block.length);
            let queue_ref = self.queue_data.get_unchecked_mut((actual_block_index * BIG_BLOCK_SIZE) +
                actual_block.update_index);
            let old_value = *queue_ref;
            *queue_ref = new_value;
            update_shift_in(block_slice, new_value, old_value)
        };

        result_inidizes
    }

    fn update_std_block(&mut self, new_value: f64, insertion_indizes: (usize, usize)) -> QuantileWindowUpdateResult {
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

        if let false = update_tracker {
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

        return QuantileWindowUpdateResult {
            new_tracker,
            ran_out_right,
            update_pred_heap,
            update_succ_heap
        };
    }

    fn update_floor_block(&mut self, new_value: f64, insertion_indizes: (usize, usize)) ->
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

        if let false = update_tracker {
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
            *self.succ_heap.get_unchecked(0)
        };

        let new_floor_big_block = unsafe {
            self.block_data.get_unchecked_mut(new_floor_data.block_index)
        };

        succ_heap_update(new_floor_big_block,
            new_floor_data.block_index,
            &mut self.succ_heap,
            &mut self.succ_heap_block_indizes,
            new_floor_data.block_index);

        self.actual_floor_big_block = new_floor_data.block_index;
        self.actual_floor_value = new_floor_data.value;
    }

    pub fn result_quantile(&mut self, result_vec: &mut Vec<f64>) {
        if !self.interpolation {
            result_vec.push(self.actual_floor_value);
        } else {
            let successor_value = unsafe {
                self.succ_heap.get_unchecked(0).value
            };

            let interpolated_result = calculate_interpolated_quantile(self.searched_rank,
                self.actual_floor_value,
                successor_value);

            result_vec.push(interpolated_result);
        }
    }

    pub fn adjust_and_result_quantile(&mut self, result_vec: &mut Vec<f64>) {
        self.global_right_shift();
        self.global_left_shift();
        self.result_quantile(result_vec);
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
                *self.succ_heap.get_unchecked(0)
            };

            pred_heap_update(old_floor_big_block,
                old_floor_big_block_index,
                &mut self.pred_heap,
                &mut self.pred_heap_block_indizes);

            succ_heap_update(old_floor_big_block,
                old_floor_big_block_index,
                &mut self.succ_heap,
                &mut self.succ_heap_block_indizes,
                new_floor_data.block_index);

            self.actual_floor_big_block = new_floor_data.block_index;
            self.actual_floor_value = new_floor_data.value;

            let new_floor_big_block = unsafe {
                self.block_data.get_unchecked(new_floor_data.block_index)
            };

            succ_heap_update(new_floor_big_block,
                new_floor_data.block_index,
                &mut self.succ_heap,
                &mut self.succ_heap_block_indizes,
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
                *self.pred_heap.get_unchecked(0)
            };

            succ_heap_update(old_floor_big_block,
                old_floor_big_block_index,
                &mut self.succ_heap,
                &mut self.succ_heap_block_indizes,
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

            pred_heap_update(new_floor_big_block,
                new_floor_data.block_index,
                &mut self.pred_heap,
                &mut self.pred_heap_block_indizes);

            succ_heap_update(new_floor_big_block,
                new_floor_data.block_index,
                &mut self.succ_heap,
                &mut self.succ_heap_block_indizes,
                new_floor_data.block_index);

            self.actual_floor_rank -= 1;
        }
    }
}

// Utils
fn calculate_needed_big_blocks(window_size: usize) -> usize {
    (window_size + (BIG_BLOCK_SIZE - 1)) / BIG_BLOCK_SIZE
}

fn calculate_interpolated_quantile(searched_rank: f64, floor_value: f64, successor_value: f64) -> f64 {
    floor_value + (successor_value - floor_value) * (searched_rank - searched_rank.floor())
}

// Initial phase
fn initial_sort(big_blocks: &mut [QuantileWindowBigBlock]) {
    let slices_per_block = BIG_BLOCK_SIZE / SORTING_NETWORK_SIZE;
    for big_block in big_blocks {
        let mut current_slice = 0;
        while current_slice < slices_per_block {
            let slice_start_index = current_slice * SORTING_NETWORK_SIZE;
            let slice_end_index = slice_start_index + SORTING_NETWORK_SIZE;
            let target_slice = &mut big_block.data[slice_start_index..slice_end_index];
            sorting_network_16(target_slice);
            current_slice += 1;
        }
    }
}

fn merge_to_big_blocks(big_blocks: &mut [QuantileWindowBigBlock]) {
    for big_block in big_blocks {
        k_way_merge_tiny_block(&mut big_block.data, BIG_BLOCK_SIZE);
    }
}

// Maybe auf Vec ändern, wegen Stack overflow
fn k_way_merge_tiny_block(data: &mut [f64; BIG_BLOCK_SIZE], big_block_size: usize) {
    let mut temp_data_vec = [0.0; BIG_BLOCK_SIZE];
    data.clone_into(&mut temp_data_vec);

    let tiny_blocks = big_block_size / SORTING_NETWORK_SIZE;
    let mut tiny_blocks_ptr: Vec<usize> = Vec::with_capacity(tiny_blocks);
    for index in 0..tiny_blocks {
        tiny_blocks_ptr.push(index * SORTING_NETWORK_SIZE);
    }

    let mut k = 0;
    loop {
        let mut smallest = f64::INFINITY;
        let mut target_block = 0;
        let mut found = false;

        for index in 0..tiny_blocks_ptr.len() {
            let temp_ptr_index = tiny_blocks_ptr[index];
            if temp_ptr_index == (index * SORTING_NETWORK_SIZE) + SORTING_NETWORK_SIZE {
                continue;
            }

            let temp_value = temp_data_vec[temp_ptr_index];
            if temp_value <= smallest {
                smallest = temp_value;
                target_block = index;
                found = true;
            }
        }

        if !found {
            break;
        }

        data[k] = smallest;
        tiny_blocks_ptr[target_block] += 1;
        k += 1;
    }
}

fn find_global_floor_value_by_rank(block_data: &[QuantileWindowBigBlock], searched_rank: usize) ->
    QuantileWindowSelectionResult {
    let big_blocks_len = block_data.len();

    let mut selection_helpers: Vec<QuantileWindowSelectionHelper> = Vec::with_capacity(big_blocks_len);
    for (index, block) in block_data.iter().enumerate() {
        selection_helpers.push(QuantileWindowSelectionHelper {
            block_index: index,
            tracker: 0,
            tracker_high: block.length,
            tracker_low: 0,
            data_slice: &block.data,
            tracker_value: 0.0,
            invalid: false,
        });
    }

    let mut pivot_candidates: Vec<QuantileWindowSelectionCandidate> = vec![
        QuantileWindowSelectionCandidate { block_index: 0,
            block_tracker: 0,
            block_value: 0.0 };
            big_blocks_len];

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
            |a, b| a.block_value.partial_cmp(&b.block_value).unwrap()).1;

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

fn initialize_block_tracker(block_data: &mut [QuantileWindowBigBlock],
    floor_value_canidate: &QuantileWindowSelectionCandidate) {
    for (index, block) in block_data.iter_mut().enumerate() {
        // Muss eigentlich weg, da wegen Duplikaten der tracker immer beim ersten auftretenden Element gesetzt
        // werden muss. Interessanterweise hat aber genau dieser if-block auswirkungen auf die IPC beim 0,5 quantil??
        // So bleibt es performancetechnisch wie vorher. Der Compiler scheint hier laut KI zu komplexen Code zu
        // generieren ohne den IF.
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

// Heaps

fn pred_heap_initial_build(block_data: &mut [QuantileWindowBigBlock], pred_heap: &mut [QuantileWindowHeapNode],
    pred_heap_block_indizes: &mut [usize]) {
    for (index, block) in  block_data.iter_mut().enumerate() {
        let current_block_tracker = block.tracker;
        let pred_value = if block.ran_out_right {
            unsafe {
                *block.data.get_unchecked(block.length - 1)
            }
        } else if current_block_tracker == 0 {
            PRED_DUMMY_VALUE
        } else {
            unsafe {
                *block.data.get_unchecked(current_block_tracker - 1)
            }
        };

        unsafe {
            let target_node = pred_heap.get_unchecked_mut(index);
            target_node.value = pred_value;
            target_node.block_index = index;
            *pred_heap_block_indizes.get_unchecked_mut(index) = index;
        }

        pred_heap_heapify_up(pred_heap, pred_heap_block_indizes, index);
    }
}

fn pred_heap_update(target_block: &QuantileWindowBigBlock, target_block_index: usize,
    pred_heap: &mut [QuantileWindowHeapNode], pred_heap_block_indizes: &mut [usize]) {
    let current_block_tracker = target_block.tracker;
    let pred_value = if target_block.ran_out_right {
        unsafe {
            *target_block.data.get_unchecked(target_block.length - 1)
        }
    } else if current_block_tracker == 0 {
        PRED_DUMMY_VALUE
    } else {
        unsafe {
            *target_block.data.get_unchecked(current_block_tracker - 1)
        }
    };

    let target_node_position = unsafe {
        *pred_heap_block_indizes.get_unchecked(target_block_index)
    };

    let target_node = unsafe {
        pred_heap.get_unchecked_mut(target_node_position)
    };

    let old_value = target_node.value;
    target_node.value = pred_value;

    if pred_value > old_value {
        pred_heap_heapify_up(pred_heap, pred_heap_block_indizes, target_node_position);
    } else {
        pred_heap_heapify_down(pred_heap, pred_heap_block_indizes, target_node_position);
    }
}

fn pred_heap_heapify_up(pred_heap: &mut [QuantileWindowHeapNode], pred_heap_block_indizes: &mut [usize],
    mut position: usize) {
    let temp_node_data = unsafe {
        *pred_heap.get_unchecked(position)
    };

    while position > 0 {
        let parent_node_position = heap_parent_index_calc(position, K_ARY_HEAP);
        let parent_node_data = unsafe {
            *pred_heap.get_unchecked(parent_node_position)
        };

        if temp_node_data.value <= parent_node_data.value {
            break;
        }

        unsafe {
            let current_node = pred_heap.get_unchecked_mut(position);
            *current_node = parent_node_data;
            *pred_heap_block_indizes.get_unchecked_mut(parent_node_data.block_index) = position;
        }

        position = parent_node_position;
    }

    unsafe {
        let current_node = pred_heap.get_unchecked_mut(position);
        *current_node = temp_node_data;
        *pred_heap_block_indizes.get_unchecked_mut(temp_node_data.block_index) = position;
    }
}

fn pred_heap_heapify_down(pred_heap: &mut [QuantileWindowHeapNode], pred_heap_block_indizes: &mut [usize],
    mut position: usize) {
    loop {
        let target_position = pred_heap_max_child(pred_heap, position);
        if target_position == position {
            break;
        }

        let position_node_data = unsafe {
            *pred_heap.get_unchecked(position)
        };

        let child_node_data = unsafe {
            *pred_heap.get_unchecked(target_position)
        };

        unsafe {
            *pred_heap.get_unchecked_mut(position) = child_node_data;
            *pred_heap.get_unchecked_mut(target_position) = position_node_data;

            *pred_heap_block_indizes.get_unchecked_mut(position_node_data.block_index) = target_position;
            *pred_heap_block_indizes.get_unchecked_mut(child_node_data.block_index) = position;
        }

        position = target_position;
    }
}

#[inline(always)]
fn pred_heap_max_child(pred_heap: &[QuantileWindowHeapNode], position: usize) -> usize {
    let heap_len = pred_heap.len();
    let first_child = heap_child_index_calc(position, K_ARY_HEAP, 1);
    if first_child >= heap_len { return  position; }
    // let last_child = heap_child_index_calc(position, K_ARY_HEAP, K_ARY_HEAP)
    //     .min(heap_len - 1);

    let mut best = position;
    let mut best_node_value = unsafe {
        pred_heap.get_unchecked(best).value
    };

    for child in 0..K_ARY_HEAP {
        let target_child = unsafe {
            pred_heap.get_unchecked(first_child + child)
        };

        if target_child.value > best_node_value {
            best = first_child + child;
            best_node_value = target_child.value;
        }
    }

    // for child in first_child..=last_child {
    //     let target_child = unsafe {
    //         pred_heap.get_unchecked(child)
    //     };

    //     if target_child.value > best_node_value {
    //         best = child;
    //         best_node_value = target_child.value;
    //     }
    // }

    best
}

fn succ_heap_initial_build(block_data: &mut [QuantileWindowBigBlock],
    succ_heap: &mut [QuantileWindowHeapNode], succ_heap_block_indizes: &mut [usize], floor_value_block_index: usize) {
    for (index, block) in  block_data.iter_mut().enumerate() {
        let current_block_tracker = block.tracker;
        let succ_index = if index == floor_value_block_index {
            current_block_tracker + 1
        } else if block.ran_out_right {
            block.length
        } else {
            current_block_tracker
        };

        let succ_value = if succ_index < block.length {
            unsafe {
                *block.data.get_unchecked(succ_index)
            }
        } else {
            SUCC_DUMMY_VALUE
        };

        unsafe {
            let target_node = succ_heap.get_unchecked_mut(index);
            target_node.value = succ_value;
            target_node.block_index = index;
            *succ_heap_block_indizes.get_unchecked_mut(index) = index;
        }

        succ_heap_heapify_up(succ_heap, succ_heap_block_indizes, index);
    }
}

fn succ_heap_update(target_block: &QuantileWindowBigBlock, target_block_index: usize,
    succ_heap: &mut [QuantileWindowHeapNode], succ_heap_block_indizes: &mut [usize], actual_floor_big_block: usize) {
    let current_block_tracker = target_block.tracker;
    let succ_index = if target_block_index == actual_floor_big_block {
        current_block_tracker + 1
    } else if target_block.ran_out_right {
        target_block.length
    } else {
        current_block_tracker
    };

    let succ_value = if succ_index < target_block.length {
        unsafe {
            *target_block.data.get_unchecked(succ_index)
        }
    } else {
        SUCC_DUMMY_VALUE
    };

    let target_node_position = unsafe {
        *succ_heap_block_indizes.get_unchecked(target_block_index)
    };

    let target_node = unsafe {
        succ_heap.get_unchecked_mut(target_node_position)
    };

    let old_value = target_node.value;
    target_node.value = succ_value;

    if succ_value < old_value {
        succ_heap_heapify_up(succ_heap, succ_heap_block_indizes, target_node_position);
    } else {
        succ_heap_heapify_down(succ_heap, succ_heap_block_indizes, target_node_position);
    }
}

fn succ_heap_heapify_up(succ_heap: &mut [QuantileWindowHeapNode], succ_heap_block_indizes: &mut [usize],
    mut position: usize) {
    let temp_node_data = unsafe {
        *succ_heap.get_unchecked(position)
    };

    while position > 0 {
        let parent_node_position = heap_parent_index_calc(position, K_ARY_HEAP);
        let parent_node_data = unsafe {
            *succ_heap.get_unchecked(parent_node_position)
        };

        if temp_node_data.value >= parent_node_data.value {
            break;
        }

        unsafe {
            let current_node = succ_heap.get_unchecked_mut(position);
            *current_node = parent_node_data;
            *succ_heap_block_indizes.get_unchecked_mut(parent_node_data.block_index) = position;
        }

        position = parent_node_position;
    }

    unsafe {
        let current_node = succ_heap.get_unchecked_mut(position);
        *current_node = temp_node_data;
        *succ_heap_block_indizes.get_unchecked_mut(temp_node_data.block_index) = position;
    }
}

fn succ_heap_heapify_down(succ_heap: &mut [QuantileWindowHeapNode], succ_heap_block_indizes: &mut [usize],
    mut position: usize) {
    loop {
        let target_position = succ_heap_min_child(succ_heap, position);
        if target_position == position {
            break;
        }

        let position_node_data = unsafe {
            *succ_heap.get_unchecked(position)
        };

        let child_node_data = unsafe {
            *succ_heap.get_unchecked(target_position)
        };

        unsafe {
            *succ_heap.get_unchecked_mut(position) = child_node_data;
            *succ_heap.get_unchecked_mut(target_position) = position_node_data;

            *succ_heap_block_indizes.get_unchecked_mut(position_node_data.block_index) = target_position;
            *succ_heap_block_indizes.get_unchecked_mut(child_node_data.block_index) = position;
        }

        position = target_position;
    }
}

#[inline(always)]
fn succ_heap_min_child(succ_heap: &[QuantileWindowHeapNode], position: usize) -> usize {
    let heap_len = succ_heap.len();
    let first_child = heap_child_index_calc(position, K_ARY_HEAP, 1);
    if first_child >= heap_len { return  position; }
    // let last_child = heap_child_index_calc(position, K_ARY_HEAP, K_ARY_HEAP)
    //     .min(heap_len - 1);

    let mut best = position;
    let mut best_node_value = unsafe {
        succ_heap.get_unchecked(best).value
    };

    for child in 0..K_ARY_HEAP {
        let target_child = unsafe {
            succ_heap.get_unchecked(first_child + child)
        };

        if target_child.value < best_node_value {
            best = first_child + child;
            best_node_value = target_child.value;
        }
    }

    // for child in first_child..=last_child {
    //     let target_child = unsafe {
    //         succ_heap.get_unchecked(child)
    //     };

    //     if target_child.value < best_node_value {
    //         best = child;
    //         best_node_value = target_child.value;
    //     }
    // }

    best
}

#[inline(always)]
fn heap_parent_index_calc(position: usize, k: usize) -> usize {
    (position - 1) / k
}

#[inline(always)]
fn heap_child_index_calc(position: usize, k: usize, num_child: usize) -> usize {
    (k * position) + num_child
}

// Update phase

fn update_shift_in(block_data_slice: &mut [f64], new_value: f64, old_value: f64) -> (usize, usize) {
    let index_old_value = block_data_slice.iter().filter(|&&x| x < old_value).count();
    let index_new_value = if new_value < old_value {
        shift_in_backwards(block_data_slice, index_old_value, new_value)
    } else if new_value > old_value {
        shift_in_forwards(block_data_slice, index_old_value, new_value)
    } else {
        block_data_slice[index_old_value] = new_value;
        index_old_value
    };

    (index_old_value, index_new_value)
}

#[inline(always)]
fn shift_in_backwards(data: &mut [f64], old_value_index: usize, new_value: f64) -> usize {
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
fn shift_in_forwards(data: &mut [f64], old_value_index: usize, new_value: f64) -> usize {
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

// Testing

fn test_sorted_blocks(block_data: &[QuantileWindowBigBlock]) -> bool {
    for block in block_data {
        let mut current_value = -f64::INFINITY;
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

// Sorting network

fn sorting_network_16(data: &mut [f64]) {
    sorting_network_cas(data, 0, 15);
    sorting_network_cas(data, 1, 14);
    sorting_network_cas(data, 2, 13);
    sorting_network_cas(data, 3, 12);
    sorting_network_cas(data, 4, 11);
    sorting_network_cas(data, 5, 10);
    sorting_network_cas(data, 6, 9);
    sorting_network_cas(data, 7, 8);

    sorting_network_cas(data, 0, 5);
    sorting_network_cas(data, 1, 7);
    sorting_network_cas(data, 2, 6);
    sorting_network_cas(data, 3, 4);
    sorting_network_cas(data, 8, 14);
    sorting_network_cas(data, 9, 13);
    sorting_network_cas(data, 10, 15);
    sorting_network_cas(data, 11, 12);

    sorting_network_cas(data, 0, 2);
    sorting_network_cas(data, 1, 3);
    sorting_network_cas(data, 4, 8);
    sorting_network_cas(data, 5, 9);
    sorting_network_cas(data, 6, 10);
    sorting_network_cas(data, 7, 11);
    sorting_network_cas(data, 12, 14);
    sorting_network_cas(data, 13, 15);

    sorting_network_cas(data, 0, 1);
    sorting_network_cas(data, 2, 7);
    sorting_network_cas(data, 3, 5);
    sorting_network_cas(data, 4, 6);
    sorting_network_cas(data, 8, 13);
    sorting_network_cas(data, 9, 11);
    sorting_network_cas(data, 10, 12);
    sorting_network_cas(data, 14, 15);

    sorting_network_cas(data, 1, 3);
    sorting_network_cas(data, 2, 4);
    sorting_network_cas(data, 5, 10);
    sorting_network_cas(data, 6, 9);
    sorting_network_cas(data, 7, 8);
    sorting_network_cas(data, 11, 13);
    sorting_network_cas(data, 12, 14);

    sorting_network_cas(data, 1, 2);
    sorting_network_cas(data, 3, 4);
    sorting_network_cas(data, 5, 7);
    sorting_network_cas(data, 8, 10);
    sorting_network_cas(data, 11, 12);
    sorting_network_cas(data, 13, 14);

    sorting_network_cas(data, 2, 3);
    sorting_network_cas(data, 4, 6);
    sorting_network_cas(data, 9, 11);
    sorting_network_cas(data, 12, 13);

    sorting_network_cas(data, 4, 5);
    sorting_network_cas(data, 6, 7);
    sorting_network_cas(data, 8, 9);
    sorting_network_cas(data, 10, 11);

    sorting_network_cas(data, 3, 4);
    sorting_network_cas(data, 5, 6);
    sorting_network_cas(data, 7,8);
    sorting_network_cas(data, 9, 10);
    sorting_network_cas(data, 11, 12);

    sorting_network_cas(data, 6, 7);
    sorting_network_cas(data, 8, 9);
}

#[inline(always)]
fn sorting_network_cas(data: &mut [f64], index1: usize, index2: usize) {
    let data_tup = unsafe {
        (*data.get_unchecked(index1), *data.get_unchecked(index2))
    };

    if data_tup.0 > data_tup.1 {
        unsafe {
            *data.get_unchecked_mut(index1) = data_tup.1;
            *data.get_unchecked_mut(index2) = data_tup.0;
        }
    }
}
