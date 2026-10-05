use std::ptr;
use crate::window::{
    utils::ordered_double::OrderedDouble
};

#[inline(always)]
pub fn calculate_needed_blocks<const BLOCK_SIZE: usize>(
    window_size: usize
) -> usize {
    window_size.div_ceil(BLOCK_SIZE)
}

#[inline(always)]
pub fn calculate_interpolated_quantile(
    searched_rank: f64,
    floor_value: f64,
    successor_value: f64
) -> f64 {
    if floor_value == successor_value {
        floor_value
    } else {
        floor_value + (successor_value - floor_value) * (searched_rank - searched_rank.floor())
    }
}

#[inline(always)]
pub fn shift_in_smaller(
    data: &mut [OrderedDouble],
    old_value_index: usize,
    new_value: OrderedDouble
) -> usize {
    debug_assert!(!data.is_empty());
    debug_assert!(old_value_index < data.len());
    debug_assert!(new_value < data[old_value_index]);
    debug_assert!(data.is_sorted());

    let new_value_index = data.iter().filter(|&&x| x <= new_value).count();
    debug_assert!(new_value_index <= old_value_index);

    // SAFETY: `new_value` is smaller than the value at `old_value_index` and the
    // block is sorted, so every element counted into `new_value_index` lies on the
    // left side of `old_value_index`. The copy therefore moves everything from
    // `new_value_index` one position to the right and the last written position is
    // `old_value_index`, which is in range.
    unsafe {
        let start_ptr = data.as_mut_ptr().add(new_value_index);
        ptr::copy(start_ptr,
            start_ptr.add(1),
            old_value_index - new_value_index);
        *data.get_unchecked_mut(new_value_index) = new_value;
    }

    new_value_index
}

#[inline(always)]
pub fn shift_in_larger(
    data: &mut [OrderedDouble],
    old_value_index: usize,
    new_value: OrderedDouble
) -> usize {
    debug_assert!(!data.is_empty());
    debug_assert!(old_value_index < data.len());
    debug_assert!(new_value > data[old_value_index]);
    debug_assert!(data.is_sorted());

    // This ensures that atleast the value at `old_value_index` lies left from
    // the `new_value` - so the calculation of `new_value_index` wont underflow.
    // Therefore it is neccessary that the `data` array is sorted, which is
    // asserted above.
    debug_assert!(data.iter().filter(|&&x| x < new_value).count() >= 1);

    // The `-1` is necessary, because `.count()` will return the positon of the
    // first element >= `new_value`. Moving that value would destroy the
    // sorting.
    let new_value_index = data.iter().filter(|&&x| x < new_value).count() - 1;
    debug_assert!(new_value_index >= old_value_index);
    debug_assert!(new_value_index < data.len());

    // SAFETY: `new_value` is larger than the value at `old_value_index`, so at
    // least that element is counted and the `- 1` cannot underflow. For the same
    // reason `new_value_index >= old_value_index`. The copy moves everything from
    // `new_value_index` one positon to the left, so the value at `old_value_index`
    // gets overridden.
    unsafe {
        let start_ptr = data.as_mut_ptr().add(old_value_index);
        ptr::copy(start_ptr.add(1),
            start_ptr,
            new_value_index - old_value_index);
        *data.get_unchecked_mut(new_value_index) = new_value;
    }

    new_value_index
}

#[inline(always)]
pub fn map_to_corresponding_value(
    value: f64
) -> OrderedDouble {
    if value.is_nan() {
        OrderedDouble::MAX
    } else {
        OrderedDouble::from_f64(value)
    }
}

#[inline(always)]
pub fn is_invalid_value(
    tested_value: OrderedDouble
) -> bool {
    tested_value == OrderedDouble::MAX
}

#[cfg(test)]
mod tests {
    use super::*;

    const QUANTILE_WINDOW_TEST_SIZE: usize = 1000;
    const TESTED_BLOCK_SIZE: usize = 64;

    #[test]
    fn test_calculate_needed_blocks() {
        let expected_needed_blocks = QUANTILE_WINDOW_TEST_SIZE.div_ceil(TESTED_BLOCK_SIZE);
        let needed_blocks = calculate_needed_blocks::<TESTED_BLOCK_SIZE>(
            QUANTILE_WINDOW_TEST_SIZE
        );

        assert_eq!(expected_needed_blocks, needed_blocks);
    }

    const TESTED_INTERPOLATION_PAIRS: [((f64, f64), f64); 5] = [
        ((4.0, 4.0), 4.0), ((f64::INFINITY, f64::INFINITY), f64::INFINITY),
        ((f64::NEG_INFINITY, f64::NEG_INFINITY), f64::NEG_INFINITY),
        ((f64::NEG_INFINITY, f64::INFINITY), f64::NAN),
        ((5.0, 7.0), 5.5)
    ];
    const TESTED_ARTIFICIAL_SEARCHED_RANK: f64 = 2.25;

    #[test]
    fn test_quantile_interpolation_pairs() {
        TESTED_INTERPOLATION_PAIRS
            .iter()
            .for_each(|((value_a, value_b), expected_result)| {
                let interpolation_result = calculate_interpolated_quantile(
                    TESTED_ARTIFICIAL_SEARCHED_RANK,
                    *value_a,
                    *value_b
                );

                if interpolation_result.is_nan() || expected_result.is_nan() {
                    assert!(
                        interpolation_result.is_nan() && expected_result.is_nan()
                    );
                } else {
                    assert_eq!(interpolation_result, *expected_result);
                }
            });
    }

    #[test]
    fn test_forwards_shift_in_landing_front() {
        let input_data = [5.0, 6.0, 7.0, 8.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 5.5;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            shift_in_larger(
                &mut test_input,
                0,
                test_value
            )
        };

        let result_data = [5.5, 6.0, 7.0, 8.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 0);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    fn test_backwards_shift_in_landing_front() {
        let input_data = [1.0, 2.0, 3.0, 9.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 0.5;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            shift_in_smaller(
                &mut test_input,
                3,
                test_value
            )
        };

        let result_data = [0.5, 1.0, 2.0, 3.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 0);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    fn test_forwards_shift_in_landing_end() {
        let input_data = [1.0, 2.0, 3.0, 4.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 9.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            shift_in_larger(
                &mut test_input,
                0,
                test_value
            )
        };

        let result_data = [2.0, 3.0, 4.0, 9.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 3);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    fn test_backwards_shift_in_landing_end() {
        let input_data = [1.0, 9.0, 9.0, 9.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 8.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            shift_in_smaller(
                &mut test_input,
                1,
                test_value
            )
        };

        let result_data = [1.0, 8.0, 9.0, 9.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 1);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    fn test_forwards_shift_in_landing_middle() {
        let input_data = [1.0, 2.0, 5.0, 6.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 3.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            shift_in_larger(
                &mut test_input,
                0,
                test_value
            )
        };

        let result_data = [2.0, 3.0, 5.0, 6.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 1);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    fn test_backwards_shift_in_landing_middle() {
        let input_data = [1.0, 4.0, 5.0, 9.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 2.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            shift_in_smaller(
                &mut test_input,
                3,
                test_value
            )
        };

        let result_data = [1.0, 2.0, 4.0, 5.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 1);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    fn test_forwards_shift_in_duplicates() {
        let input_data = [1.0, 3.0, 3.0, 3.0, 9.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 3.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            shift_in_larger(
                &mut test_input,
                0,
                test_value
            )
        };

        let result_data = [3.0, 3.0, 3.0, 3.0, 9.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 0);
        assert_eq!(&test_input, &ordered_result_data);
    }

    #[test]
    fn test_backwards_shift_in_duplicates() {
        let input_data = [1.0, 3.0, 3.0, 3.0, 9.0];
        let mut test_input = turn_into_ordered_double_vec(&input_data);

        let input_value = 3.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = {
            shift_in_smaller(
                &mut test_input,
                4,
                test_value
            )
        };

        let result_data = [1.0, 3.0, 3.0, 3.0, 3.0];
        let ordered_result_data = turn_into_ordered_double_vec(&result_data);

        assert_eq!(insert_index, 4);
        assert_eq!(&test_input, &ordered_result_data);
    }

    fn turn_into_ordered_double_vec(input_array: &[f64]) -> Vec<OrderedDouble> {
        input_array
            .iter()
            .map(|x| OrderedDouble::from_f64(*x))
            .collect::<Vec<OrderedDouble>>()
    }
}
