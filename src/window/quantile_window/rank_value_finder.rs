use crate::window::{
    quantile_window::QuantileWindow,
    utils::ordered_double::OrderedDouble
};

struct QuantileWindowSelectionHelper<'a> {
    block_index: usize,
    tracker: usize,
    tracker_low: usize,
    tracker_high: usize,
    data_slice: &'a [OrderedDouble],
    tracker_value: OrderedDouble,
    invalid: bool,
}

impl<'a> QuantileWindowSelectionHelper<'a> {

    #[inline(always)]
    fn new(
        block_index: usize,
        block_length: usize,
        block_data: &'a [OrderedDouble]
    ) -> Self {
        Self {
            block_index,
            tracker: 0,
            tracker_low: 0,
            tracker_high: block_length,
            data_slice: block_data,
            tracker_value: OrderedDouble::default(),
            invalid: false
        }
    }
}

#[derive(Clone, Copy)]
pub struct QuantileWindowSelectionCandidate {
    block_index: usize,
    block_tracker: usize,
    block_value: OrderedDouble,
}

impl Default for QuantileWindowSelectionCandidate {

    #[inline(always)]
    fn default() -> Self {
        Self {
            block_index: 0,
            block_tracker: 0,
            block_value: OrderedDouble::default()
        }
    }
}

impl QuantileWindowSelectionCandidate {

    #[inline(always)]
    pub(super) fn get_block_index(
        &self
    ) -> usize {
        self.block_index
    }

    #[inline(always)]
    pub(super) fn get_block_tracker(
        &self
    ) -> usize {
        self.block_tracker
    }

    #[inline(always)]
    pub(super) fn get_block_value(
        &self
    ) -> OrderedDouble {
        self.block_value
    }
}

pub struct QuantileWindowSelectionResult {
    selection_candidate: QuantileWindowSelectionCandidate,
    duplicates_to_skip: usize,
}

impl QuantileWindowSelectionResult {

    #[inline(always)]
    pub(super) fn get_selection_candidate(
        &self
    ) -> QuantileWindowSelectionCandidate {
        self.selection_candidate
    }

    #[inline(always)]
    pub(super) fn get_duplicates_to_skip(
        &self
    ) -> usize {
        self.duplicates_to_skip
    }
}

impl<const BLOCK_SIZE: usize> QuantileWindow<BLOCK_SIZE> {

    #[inline(always)]
    pub(super) fn find_value_by_rank(
        &self,
        searched_rank: usize,
    ) -> QuantileWindowSelectionResult {
        let blocks_len = self.block_data.len();
        let mut selection_helpers = Vec::with_capacity(blocks_len);
        for (index, block) in self.block_data.iter().enumerate() {
            selection_helpers.push(
                QuantileWindowSelectionHelper::new(index, block.length, &block.data)
            );
        }

        let mut pivot_candidates: Vec<QuantileWindowSelectionCandidate> = {
             vec![QuantileWindowSelectionCandidate::default(); blocks_len]
        };

        loop {
            let active_blocks = prepare_round(&mut selection_helpers, &mut pivot_candidates);

            let candidates_slice = &mut pivot_candidates[0..active_blocks];
            let pivot_element = candidates_slice.select_nth_unstable_by(
                active_blocks / 2,
                |a, b| a.block_value.cmp(&b.block_value)
            ).1;

            let (global_lower_bound, global_upper_bound) = {
                get_global_bounds(&selection_helpers, pivot_element)
            };

            if searched_rank >= global_lower_bound && searched_rank < global_upper_bound {
                let duplicates_to_skip = searched_rank - global_lower_bound;
                return QuantileWindowSelectionResult {
                    selection_candidate: *pivot_element,
                    duplicates_to_skip
                };
            } else if global_lower_bound < searched_rank {
                adjust_low_ranks(&mut selection_helpers, pivot_element);
            } else if global_lower_bound > searched_rank {
                adjust_high_ranks(&mut selection_helpers, pivot_element);
            }
        }
    }
}

#[inline(always)]
fn prepare_round(
    selection_helpers: &mut [QuantileWindowSelectionHelper],
    pivot_candidates: &mut [QuantileWindowSelectionCandidate]
) -> usize {
    let mut active_blocks = 0;
    for selection_helper in selection_helpers {
        if selection_helper.invalid {
            continue;
        }

        let tracker_index = calculate_tracker(
            selection_helper.tracker_low,
            selection_helper.tracker_high
        );
        selection_helper.tracker = tracker_index;

        let tracker_value = selection_helper.data_slice[tracker_index];
        selection_helper.tracker_value = tracker_value;

        pivot_candidates[active_blocks].block_index = selection_helper.block_index;
        pivot_candidates[active_blocks].block_tracker = tracker_index;
        pivot_candidates[active_blocks].block_value = tracker_value;
        active_blocks += 1;
    }

    active_blocks
}

#[inline(always)]
fn get_global_bounds(
    selection_helpers: &[QuantileWindowSelectionHelper],
    pivot_element: &QuantileWindowSelectionCandidate
) -> (usize, usize) {
    let mut global_lower_bound = 0;
    let mut global_upper_bound = 0;

    for selection_helper in selection_helpers {
        if selection_helper.invalid {
            continue;
        }

        let target_slice = {
            &selection_helper.data_slice[selection_helper.tracker_low..selection_helper.tracker_high]
        };

        let pivot_lb = {
            target_slice.partition_point(|&x| x < pivot_element.block_value)
        };
        global_lower_bound += pivot_lb;

        let pivot_ub = {
            target_slice.partition_point(|&x| x <= pivot_element.block_value)
        };
        global_upper_bound += pivot_ub;
    }

    let mut global_acc_low = 0;
    for selection_helper in selection_helpers {
        global_acc_low += selection_helper.tracker_low;
    }

    ((global_lower_bound + global_acc_low), (global_upper_bound + global_acc_low))
}

#[inline(always)]
fn adjust_low_ranks(
    selection_helpers: &mut [QuantileWindowSelectionHelper],
    pivot_element: &QuantileWindowSelectionCandidate
) {
    for selection_helper in selection_helpers {
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
}

#[inline(always)]
fn adjust_high_ranks(
    selection_helpers: &mut [QuantileWindowSelectionHelper],
    pivot_element: &QuantileWindowSelectionCandidate
) {
    for selection_helper in selection_helpers {
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

#[inline(always)]
fn calculate_tracker(
    low: usize,
    high: usize
) -> usize {
    (low + (high - 1)) / 2
}
