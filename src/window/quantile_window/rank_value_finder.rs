use crate::window::utils::ordered_double::OrderedDouble;

pub struct QuantileWindowSelectionHelper<'a> {
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
    pub fn new(
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

impl QuantileWindowSelectionCandidate {

    #[inline(always)]
    pub fn get_block_index(
        &self
    ) -> usize {
        self.block_index
    }

    #[inline(always)]
    pub fn get_block_tracker(
        &self
    ) -> usize {
        self.block_tracker
    }

    #[inline(always)]
    pub fn get_block_value(
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
    pub fn get_selection_cadidate(
        &self
    ) -> QuantileWindowSelectionCandidate {
        self.selection_candidate
    }

    #[inline(always)]
    pub fn get_duplicates_to_skip(
        &self
    ) -> usize {
        self.duplicates_to_skip
    }
}

#[inline(always)]
pub fn find_rank_value(
    searched_rank: usize,
    selection_helpers: Vec<QuantileWindowSelectionHelper>
) -> QuantileWindowSelectionResult {
    let mut selection_helpers = selection_helpers;
    let blocks_len = selection_helpers.len();

    let mut pivot_candidates: Vec<QuantileWindowSelectionCandidate> = vec![
        QuantileWindowSelectionCandidate {
            block_index: 0,
            block_tracker: 0,
            block_value: OrderedDouble::default()
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
        let pivot_element = candidates_slice
            .select_nth_unstable_by(active_blocks / 2,
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
