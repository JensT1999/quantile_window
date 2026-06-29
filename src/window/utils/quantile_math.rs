#[inline(always)]
pub fn calculate_interpolated_quantile(searched_rank: f64, floor_value: f64, successor_value: f64) -> f64 {
    floor_value + (successor_value - floor_value) * (searched_rank - searched_rank.floor())
}
