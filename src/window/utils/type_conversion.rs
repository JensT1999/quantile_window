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
