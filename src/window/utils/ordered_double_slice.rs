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
