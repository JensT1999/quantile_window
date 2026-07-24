use crate::window::utils::ordered_double::OrderedDouble;

pub trait OrderedDoubleSlice {
    fn shift_in_backwards(&mut self, old_value_index: usize, new_value: OrderedDouble) -> usize;
    fn shift_in_forwards(&mut self, old_value_index: usize, new_value: OrderedDouble) -> usize;
}

impl OrderedDoubleSlice for [OrderedDouble] {

    fn shift_in_backwards(&mut self, old_value_index: usize, new_value: OrderedDouble) -> usize {


        let mut index = old_value_index;
        unsafe {
            while (index > 0) && (new_value < *self.get_unchecked(index - 1)) {
                *self.get_unchecked_mut(index) = *self.get_unchecked(index - 1);
                index -= 1;
            }
            *self.get_unchecked_mut(index) = new_value;
        }

        index
    }

    fn shift_in_forwards(&mut self, old_value_index: usize, new_value: OrderedDouble) -> usize {
        let mut index = old_value_index;
        unsafe {
            while (index < (self.len() - 1)) && (new_value > *self.get_unchecked(index + 1)) {
                *self.get_unchecked_mut(index) = *self.get_unchecked(index + 1);
                index += 1;
            }
            *self.get_unchecked_mut(index) = new_value;
        }

        index
    }
}

#[cfg(test)]
mod test {
    use crate::window::utils::{ordered_double::OrderedDouble, ordered_double_slice::OrderedDoubleSlice};

    #[test]
    fn test_backwards_shift_complete_shift() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = input_data
            .iter()
            .map(|x| OrderedDouble::from_f64(*x))
            .collect::<Vec<OrderedDouble>>();

        let input_value = 1.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = test_input.shift_in_backwards(4, test_value);
        assert!(insert_index == 0);
        assert!(test_input[0] == test_value);
        assert!(test_input[4] != OrderedDouble::from_f64(6.0));
    }

    #[test]
    fn test_backwards_shift_middle_shift_in() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = input_data
            .iter()
            .map(|x| OrderedDouble::from_f64(*x))
            .collect::<Vec<OrderedDouble>>();

        let input_value = 3.5;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = test_input.shift_in_backwards(4, test_value);
        assert!(insert_index == 2);
        assert!(test_input[2] == test_value);
        assert!(test_input[4] != OrderedDouble::from_f64(6.0));
    }

    #[test]
    fn test_backwards_shift_no_shift_in() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = input_data
            .iter()
            .map(|x| OrderedDouble::from_f64(*x))
            .collect::<Vec<OrderedDouble>>();

        let input_value = 7.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = test_input.shift_in_backwards(4, test_value);
        assert!(insert_index == 4);
        assert!(test_input[4] == test_value);
        assert!(test_input[4] != OrderedDouble::from_f64(6.0));
    }

    #[test]
    fn test_forwards_shift_complete_shift() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = input_data
            .iter()
            .map(|x| OrderedDouble::from_f64(*x))
            .collect::<Vec<OrderedDouble>>();

        let input_value = 7.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = test_input.shift_in_forwards(0, test_value);
        assert!(insert_index == 4);
        assert!(test_input[4] == test_value);
        assert!(test_input[0] != OrderedDouble::from_f64(2.0));
    }

    #[test]
    fn test_forwards_shift_middle_shift_in() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = input_data
            .iter()
            .map(|x| OrderedDouble::from_f64(*x))
            .collect::<Vec<OrderedDouble>>();

        let input_value = 3.5;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = test_input.shift_in_forwards(0, test_value);
        assert!(insert_index == 1);
        assert!(test_input[1] == test_value);
        assert!(test_input[0] != OrderedDouble::from_f64(2.0));
    }

    #[test]
    fn test_forwards_shift_no_shift_in() {
        let input_data = [2.0, 3.0, 4.0, 5.0, 6.0];
        let mut test_input = input_data
            .iter()
            .map(|x| OrderedDouble::from_f64(*x))
            .collect::<Vec<OrderedDouble>>();

        let input_value = 1.0;
        let test_value = OrderedDouble::from_f64(input_value);

        let insert_index = test_input.shift_in_forwards(0, test_value);
        assert!(insert_index == 0);
        assert!(test_input[0] == test_value);
        assert!(test_input[0] != OrderedDouble::from_f64(2.0));
    }
}
