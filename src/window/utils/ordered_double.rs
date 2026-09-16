use std::fmt;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(transparent)]
pub struct OrderedDouble {
    data: i64,
}

impl fmt::Debug for OrderedDouble {

    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "OrderedDouble value: {}", self.to_f64())
    }
}

impl Default for OrderedDouble {

    #[inline(always)]
    fn default() -> Self {
        OrderedDouble::from_f64(0.0)
    }
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

mod type_conversion {

    // The conversions are taken from rusts `total_cmp` from `f64`.
    #[inline(always)]
    pub const fn convert_f64_to_i64(value: f64) -> i64 {
        let result_bits = value.to_bits() as i64;
        result_bits ^ (((result_bits >> 63) as u64) >> 1) as i64
    }

    #[inline(always)]
    pub const fn convert_i64_to_f64(value: i64) -> f64 {
        let result = value ^ (((value >> 63) as u64) >> 1) as i64;
        f64::from_bits(result as u64)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_ordered_double_eq() {
        let value_a = OrderedDouble::from_f64(5.0);
        let value_b = OrderedDouble::from_f64(5.0);

        assert!(value_a == value_b);
    }

    #[test]
    fn test_ordered_double_ord_a() {
        let value_a = OrderedDouble::from_f64(5.0);
        let value_b = OrderedDouble::from_f64(6.0);

        assert!(value_a < value_b);
        assert!(value_b > value_a);
    }

    #[test]
    fn test_ordered_double_ord_b() {
        let mut values = [3.0, 2.0, 1.0, 9.0, 7.0]
            .into_iter()
            .map(OrderedDouble::from_f64)
            .collect::<Vec<OrderedDouble>>();

        assert!(!values.is_sorted());
        values.sort();
        assert!(values.is_sorted());
    }

    #[test]
    fn test_ordered_double_nan() {
        let mut values = [3.0, 2.0, -f64::NAN, f64::NAN, 7.0]
            .into_iter()
            .map(OrderedDouble::from_f64)
            .collect::<Vec<OrderedDouble>>();

        assert!(!values.is_sorted());
        values.sort();
        assert!(values.is_sorted());

        assert!(values[0] == OrderedDouble::MIN);
        assert!(values[4] == OrderedDouble::MAX);
    }

    #[test]
    fn test_ordered_double_sorting_order() {
        let mut values = [3.0, f64::INFINITY, 0.0, -0.0, -3.0, -f64::INFINITY, f64::NAN, -f64::NAN]
            .into_iter()
            .map(OrderedDouble::from_f64)
            .collect::<Vec<OrderedDouble>>();

        assert!(!values.is_sorted());
        values.sort();
        assert!(values.is_sorted());

        assert!(values[0] == OrderedDouble::MIN);
        assert!(values[1] == OrderedDouble::from_f64(-f64::INFINITY));
        assert!(values[2] == OrderedDouble::from_f64(-3.0));
        assert!(values[3] == OrderedDouble::from_f64(-0.0));
        assert!(values[4] == OrderedDouble::from_f64(0.0));
        assert!(values[5] == OrderedDouble::from_f64(3.0));
        assert!(values[6] == OrderedDouble::from_f64(f64::INFINITY));
        assert!(values[7] == OrderedDouble::MAX);
    }
}
