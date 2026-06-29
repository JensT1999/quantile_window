use std::cmp::Ordering;

use crate::window::utils::type_conversion;

#[derive(Clone, Copy)]
pub struct OrderedDouble {
    data: i64,
}

impl OrderedDouble {

    pub const MAX: OrderedDouble = OrderedDouble::from_f64(f64::NAN);
    pub const MIN: OrderedDouble = OrderedDouble::from_f64(-f64::NAN);

    pub const fn from_f64(value: f64) -> OrderedDouble {
        OrderedDouble {
            data: type_conversion::convert_f64_to_i64(value)
        }
    }

    pub const fn to_f64(self) -> f64 {
        type_conversion::convert_i64_to_f64(self.data)
    }
}

impl PartialEq for OrderedDouble {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl Eq for OrderedDouble {}

impl PartialOrd for OrderedDouble {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.data.cmp(&other.data))
    }
}

impl Ord for OrderedDouble {
    fn cmp(&self, other: &Self) -> Ordering {
        self.data.cmp(&other.data)
    }
}
