use std::{env, error::Error, fmt::Display, time::{Duration, Instant}};
use rand::{RngExt, SeedableRng, rngs::StdRng};
use quantile_window::WindowError;

// Different benchmarks
const STD_BENCHMARK_KEY: &str = "std_bench";
const BLOCK_SIZES_BENCHMARK_KEY: &str = "block_sizes_bench";

// Basic harness configuration
const BENCHED_DATA_SEED: u64 = 42;
const STD_BENCH_ITERATIONS: usize = 10;
const STD_INDEX_OF_FIRST_VALID_RESULT: usize = 1;

// Basic length benchmark configuration
const LENGTH_BENCH_BENCHED_INPUT_LENGHTS: [usize; 3] = [
    5_000_000, 10_000_000, 20_000_000
];
const LENGTH_BENCH_BENCHED_QUANTILE: f64 = 0.5;
const LENGTH_BENCH_BENCHED_WINDOW_SIZE: usize = 1000;
const LENGTH_BENCH_STD_BENCHED_DISTRIBUTION: DataDistribution = DataDistribution::Continuous {
    data_bounds: DataBounds { lowest_possible_value: -1000.0, highest_possible_value: 1000.0 }
};

// Basic windowsize and distribution benchmark configuration
const WINDOW_SIZE_DIST_BENCH_BENCHED_LENGTH: usize = 20_000_000;
const WINDOW_SIZE_DIST_BENCH_BENCHED_QUANTILE: f64 = 0.5;
const WINDOW_SIZE_DIST_BENCH_BENCHED_WINDOW_SIZES: [usize; 6] = [
    100, 1_000, 10_000, 100_000, 1_000_000, 2_000_000
];
const WINDOW_SIZE_DIST_BENCH_BENCHED_DISTRIBUTIONS: [DataDistribution; 4] = [
    DataDistribution::Continuous {
        data_bounds: DataBounds { lowest_possible_value: -1000.0, highest_possible_value: 1000.0 }
    },

    DataDistribution::Trend {
        trend_ratio: 4.0,
        noise_scale: 200.0
    },

    DataDistribution::NaN {
        data_bounds: DataBounds { lowest_possible_value: -1000.0, highest_possible_value: 1000.0 },
        nan_ratio: 0.5
    },

    DataDistribution::SinusWave {
        frequency: 1.0,
        amplitude: 1000.0,
        noise_bounds: DataBounds { lowest_possible_value: -1000.0, highest_possible_value: 1000.0 }
    }
];

// Basic quantile benchmark configuration
const QUANTILE_BENCH_BENCHED_LENGTH: usize = 20_000_000;
const QUANTILE_BENCH_BENCHED_QUANTILES: [f64; 5] = [
    0.0, 0.25, 0.5, 0.75, 1.0
];
const QUANTILE_BENCH_BENCHED_DISTRIBUTION: DataDistribution = DataDistribution::Continuous {
    data_bounds: DataBounds { lowest_possible_value: -1000.0, highest_possible_value: 1000.0 }
};
const QUANTILE_BENCH_BENCHED_WINDOW_SIZE: usize = 1000;

// Basic block sizes benchmark configuration
const BLOCK_SIZES_BENCH_BENCHED_LENGTH: usize = 20_000_000;
const BLOCK_SIZES_BENCH_BENCHED_QUANTILE: f64 = 0.5;
const BLOCK_SIZES_BENCH_BENCHED_DISTRIBUTIONS: [DataDistribution; 2] = [
    DataDistribution::Continuous {
        data_bounds: DataBounds { lowest_possible_value: -1000.0, highest_possible_value: 1000.0 }
    },

    DataDistribution::Trend {
        trend_ratio: 50.0,
        noise_scale: 200.0
    }
];
const BLOCK_SIZES_BENCH_BENCHED_WINDOW_SIZES: [usize; 6] = [
    100, 1_000, 10_000, 100_000, 1_000_000, 2_000_000
];
const BLOCK_SIZES_BENCH_BENCHED_BLOCK_SIZES: [usize; 4] = [
    16, 32, 64, 128
];
const METRICS_OUTPUT_COL_FACTOR: usize = 15;

struct DataBounds {
    lowest_possible_value: f64,
    highest_possible_value: f64,
}

enum DataDistribution {
    Continuous {
        data_bounds: DataBounds
    },

    Trend {
        trend_ratio: f64,
        noise_scale: f64,
    },

    NaN {
        data_bounds: DataBounds,
        nan_ratio: f64,
    },

    SinusWave {
        frequency: f64,
        amplitude: f64,
        noise_bounds: DataBounds,
    }
}

impl Display for DataDistribution {

    fn fmt(
        &self, f: &mut std::fmt::Formatter<'_>
    ) -> std::fmt::Result {
        let output_string = match self {
            DataDistribution::Continuous { data_bounds} => {
                format!(
                    "continuous: bounds {}..{}",
                    data_bounds.lowest_possible_value,
                    data_bounds.highest_possible_value
                )
            },

            DataDistribution::Trend { trend_ratio, noise_scale } => {
                format!(
                    "trend: ratio {}, noise scale {}",
                    trend_ratio,
                    noise_scale
                )
            },

            DataDistribution::NaN { data_bounds, nan_ratio } => {
                format!(
                    "nan: bounds {}..{}, ratio {}",
                    data_bounds.lowest_possible_value,
                    data_bounds.highest_possible_value,
                    nan_ratio
                )
            },

            DataDistribution::SinusWave { frequency, amplitude, noise_bounds} => {
                format!(
                    "sinus_wave: frequency {}, amplitude {}, noise {}..{}",
                    frequency,
                    amplitude,
                    noise_bounds.lowest_possible_value,
                    noise_bounds.highest_possible_value
                )
            }
        };

        write!(f, "{}", output_string)
    }
}

impl DataDistribution {

    fn generate_distribution(
        &self,
        rng: &mut StdRng,
        length: usize,
        window_size: usize
    ) -> Vec<f64> {
        match self {
            DataDistribution::Continuous { data_bounds} => {
                (0..length)
                    .map(|_| rng
                        .random_range(
                            data_bounds.lowest_possible_value..data_bounds.highest_possible_value
                        )
                    )
                    .collect::<Vec<f64>>()
            },

            DataDistribution::Trend { trend_ratio, noise_scale } => {
                let trend_per_step = (trend_ratio * noise_scale) / (window_size as f64);
                (0..length)
                    .map(|index| {
                        let uniform = rng.random::<f64>().max(f64::MIN_POSITIVE);
                        let noise = -noise_scale * uniform.ln();
                        (index as f64) * trend_per_step + noise
                    })
                    .collect::<Vec<f64>>()
            },

            DataDistribution::NaN { data_bounds, nan_ratio } => {
                assert!(
                    *nan_ratio >= 0.01 &&
                    *nan_ratio <= 1.0,
                    "Please enter valid NaN range between 0.01 and 1.0."
                );

                (0..length)
                    .map(|_| {
                        if rng.random_bool(*nan_ratio) {
                            f64::NAN
                        } else {
                            rng.random_range(data_bounds.lowest_possible_value..data_bounds.highest_possible_value)
                        }
                    })
                    .collect::<Vec<f64>>()
            },

            DataDistribution::SinusWave { frequency, amplitude, noise_bounds } => {
                (0..length)
                    .map(|x| {
                        let wave = (x as f64 * frequency).sin() * amplitude;
                        let noise = rng.random_range(
                            noise_bounds.lowest_possible_value..noise_bounds.highest_possible_value
                        );

                        wave + noise
                    })
                    .collect::<Vec<f64>>()
            }
        }
    }

    fn short_description(
        &self
    ) -> String {
        match self {
            DataDistribution::Continuous { data_bounds: _ } => String::from("continuous"),
            DataDistribution::Trend { trend_ratio: _, noise_scale: _ } => String::from("trend"),
            DataDistribution::NaN { data_bounds: _, nan_ratio: _ } => String::from("nan"),
            DataDistribution::SinusWave { frequency: _, amplitude: _, noise_bounds: _ } => String::from("sinus_wave")
        }
    }
}

struct BenchmarkResult {
    input_length: usize,
    median: Duration,
    max: Duration,
    min: Duration,
}

impl BenchmarkResult {

    #[inline(always)]
    fn formatted_slowest_metric(
        &self
    ) -> String {
        utils::format_metrics_output(
            "slowest",
            self.get_slowest(),
            self.get_slowest_m_per_s()
        )
    }

    #[inline(always)]
    fn get_slowest(
        &self
    ) -> f64 {
        self.max.as_secs_f64()
    }

    #[inline(always)]
    fn get_slowest_m_per_s(
        &self
    ) -> f64 {
        utils::calc_m_per_s(
            self.get_slowest(),
            self.input_length
        )
    }

    #[inline(always)]
    fn formatted_median_metric(
        &self
    ) -> String {
        utils::format_metrics_output(
            "median",
            self.get_median(),
            self.get_median_m_per_s()
        )
    }

    #[inline(always)]
    fn get_median(
        &self
    ) -> f64 {
        self.median.as_secs_f64()
    }

    #[inline(always)]
    fn get_median_m_per_s(
        &self
    ) -> f64 {
        utils::calc_m_per_s(
            self.get_median(),
            self.input_length
        )
    }

    #[inline(always)]
    fn formatted_fastest_metric(
        &self
    ) -> String {
        utils::format_metrics_output(
            "fastest",
            self.get_fastest(),
            self.get_fastest_m_per_s()
        )
    }

    #[inline(always)]
    fn get_fastest(
        &self
    ) -> f64 {
        self.min.as_secs_f64()
    }

    #[inline(always)]
    fn get_fastest_m_per_s(
        &self
    ) -> f64 {
        utils::calc_m_per_s(
            self.get_fastest(),
            self.input_length
        )
    }

    #[inline(always)]
    fn formatted_spread_metric(
        &self
    ) -> String {
        format!("spread: {:.2}%", self.get_spread())
    }

    #[inline(always)]
    fn get_spread(
        &self
    ) -> f64 {
        utils::calc_spread_percent(
            self.max.as_secs_f64(),
            self.median.as_secs_f64(),
            self.min.as_secs_f64()
        )
    }
}

struct BenchmarkHarnessConfiguration {
    iterations: usize,
    first_valid_result: usize,
    length: usize,
    quantile: f64,
}

struct BenchmarkCaseConfiguration<'a> {
    distribution: &'a DataDistribution,
    window_size: usize,
}

trait BenchmarkedFunction {

    fn benchmark(
        test_data: &[f64],
        window_size: usize,
        quantile: f64
    ) -> Result<Vec<f64>, WindowError>;
}

struct BenchmarkStdDispatcher;
impl BenchmarkedFunction for BenchmarkStdDispatcher {

    fn benchmark(
        test_data: &[f64],
        window_size: usize,
        quantile: f64
    ) -> Result<Vec<f64>, WindowError> {
        quantile_window::rolling_quantile_window(
            test_data,
            window_size,
            quantile
        )
    }
}

struct Benchmark16BlockSize;
impl BenchmarkedFunction for Benchmark16BlockSize {

    fn benchmark(
        test_data: &[f64],
        window_size: usize,
        quantile: f64
    ) -> Result<Vec<f64>, WindowError> {
        quantile_window::rolling_quantile_window_generic::<16,1>(
            test_data,
            window_size,
            quantile
        )
    }
}

struct Benchmark32BlockSize;
impl BenchmarkedFunction for Benchmark32BlockSize {

    fn benchmark(
        test_data: &[f64],
        window_size: usize,
        quantile: f64
    ) -> Result<Vec<f64>, WindowError> {
        quantile_window::rolling_quantile_window_generic::<32,2>(
            test_data,
            window_size,
            quantile
        )
    }
}

struct Benchmark64BlockSize;
impl BenchmarkedFunction for Benchmark64BlockSize {

    fn benchmark(
        test_data: &[f64],
        window_size: usize,
        quantile: f64
    ) -> Result<Vec<f64>, WindowError> {
        quantile_window::rolling_quantile_window_generic::<64,4>(
            test_data,
            window_size,
            quantile
        )
    }
}

struct Benchmark128BlockSize;
impl BenchmarkedFunction for Benchmark128BlockSize {

    fn benchmark(
        test_data: &[f64],
        window_size: usize,
        quantile: f64
    ) -> Result<Vec<f64>, WindowError> {
        quantile_window::rolling_quantile_window_generic::<128,8>(
            test_data,
            window_size,
            quantile
        )
    }
}

fn run_std_benchmark(
    rng: &mut StdRng,
    harness_config: &BenchmarkHarnessConfiguration,
    case_config: &BenchmarkCaseConfiguration
) -> Result<BenchmarkResult, WindowError> {
    let test_data = case_config
        .distribution
        .generate_distribution(rng, harness_config.length, case_config.window_size);

    let mut benchmark_results = run_benchmark_iterations::<BenchmarkStdDispatcher>(
        harness_config.iterations,
        &test_data,
        case_config.window_size,
        harness_config.quantile
    )?;

    Ok(
        build_benchmark_result(
            &mut benchmark_results,
            harness_config.length,
            harness_config.first_valid_result
        )
    )
}

fn run_benchmark_iterations<B>(
    iterations: usize,
    test_data: &[f64],
    window_size: usize,
    quantile: f64
) -> Result<Vec<Duration>, WindowError> where
    B: BenchmarkedFunction {
    let mut result_vec = vec![];
    for _index in 0..iterations {
        let instant = Instant::now();
        let result = B::benchmark(test_data, window_size, quantile)?;
        result_vec.push(instant.elapsed());
        std::hint::black_box(result);
    }

    Ok(result_vec)
}

fn build_benchmark_result(
    input_data: &mut [Duration],
    length: usize,
    first_valid_result: usize
) -> BenchmarkResult {
    assert!(first_valid_result < input_data.len());

    let valid_durations = &mut input_data[first_valid_result..];
    valid_durations.sort();

    let valid_len = valid_durations.len();
    let median_duration = if valid_len.is_multiple_of(2) {
        let middle = valid_len / 2;
        let index_a = middle - 1;
        let index_b = middle;

        let dur_a = valid_durations[index_a];
        let dur_b = valid_durations[index_b];
        (dur_a + dur_b) / 2
    } else {
        let index_a = valid_len / 2;
        valid_durations[index_a]
    };

    let max_duration = valid_durations[valid_len - 1];
    let min_duration = valid_durations[0];

    BenchmarkResult {
        input_length: length,
        median: median_duration,
        max: max_duration,
        min: min_duration
    }
}

mod utils {

    #[inline(always)]
    pub fn calc_m_per_s(
        base_value: f64,
        input_length: usize
    ) -> f64 {
        let base_m_per_s = input_length as f64 / base_value;
        base_m_per_s / 1000000.0
    }

    #[inline(always)]
    pub fn calc_spread_percent(
        max_value: f64,
        median_value: f64,
        min_value: f64
    ) -> f64 {
        ((max_value - min_value) / median_value) * 100.0
    }

    #[inline(always)]
    pub fn format_metrics_output(
        name: &str,
        value: f64,
        m_per_s: f64
    ) -> String {
        format!("{}: {:.5}s - {:.1} M/s", name, value, m_per_s)
    }
}

fn main() -> Result<(), Box<dyn Error>>{
    let args = env::args().collect::<Vec<String>>();

    let std_benchmark = args.contains(&STD_BENCHMARK_KEY.to_string());
    let block_sizes_benchmark = args.contains(&BLOCK_SIZES_BENCHMARK_KEY.to_string());

    if std_benchmark || block_sizes_benchmark {
        start_benchmarks(
            std_benchmark,
            block_sizes_benchmark
        )?;
    } else {
        println!(
            "Select one of the following benchmarks as argument: Standard - {} or Block sizes - {}",
            STD_BENCHMARK_KEY,
            BLOCK_SIZES_BENCHMARK_KEY
        );
    }

    Ok(())
}

fn start_benchmarks(
    std_benchmark: bool,
    block_sizes_benchmark:bool
) -> Result<(), WindowError> {
    let mut rng = StdRng::seed_from_u64(BENCHED_DATA_SEED);

    if std_benchmark {
        // Input length benchmark
        start_length_benchmark(&mut rng)?;
        println!();

        // Window size + different distributions benchmark
        start_window_size_dist_benchmark(&mut rng)?;
        println!();

        // Quantiles benchmark
        start_quantile_benchmark(&mut rng)?;
        println!();
    }

    if block_sizes_benchmark {
        // Block sizes benchmark
        start_block_sizes_benchmark(&mut rng)?;
    }

    Ok(())
}

fn start_length_benchmark(
    rng: &mut StdRng
) -> Result<(), WindowError> {
    let mut benchmark_results = Vec::with_capacity(LENGTH_BENCH_BENCHED_INPUT_LENGHTS.len());
    for input_length in LENGTH_BENCH_BENCHED_INPUT_LENGHTS {
        let harness_config = BenchmarkHarnessConfiguration {
            iterations: STD_BENCH_ITERATIONS,
            first_valid_result: STD_INDEX_OF_FIRST_VALID_RESULT,
            length: input_length,
            quantile: LENGTH_BENCH_BENCHED_QUANTILE
        };

        let case_config = BenchmarkCaseConfiguration {
            distribution: &LENGTH_BENCH_STD_BENCHED_DISTRIBUTION,
            window_size: LENGTH_BENCH_BENCHED_WINDOW_SIZE
        };

        let benchmark_result = run_std_benchmark(
            rng,
            &harness_config,
            &case_config
        )?;
        benchmark_results.push(benchmark_result);
    }

    let benchmark_header = format!(
        "Benchmark configuration - window size: {} quantile: {} distribution: {}",
        LENGTH_BENCH_BENCHED_WINDOW_SIZE,
        LENGTH_BENCH_BENCHED_QUANTILE,
        LENGTH_BENCH_STD_BENCHED_DISTRIBUTION.short_description()
    );
    println!("{}", benchmark_header);
    build_length_benchmark_table(&benchmark_results)
        .iter()
        .for_each(|result| println!("{}", result));

    Ok(())
}

fn build_length_benchmark_table(
    benchmark_results: &[BenchmarkResult]
) -> Vec<String> {
    assert!(
        LENGTH_BENCH_BENCHED_INPUT_LENGHTS.len() == benchmark_results.len(),
        "It appears that arrays of different lengths were entered."
    );

    LENGTH_BENCH_BENCHED_INPUT_LENGHTS
        .iter()
        .zip(benchmark_results.iter())
        .map(|item| {
            let formatted_metrics = format!(
                "{} - {} - {} - {}",
                item.1.formatted_slowest_metric(),
                item.1.formatted_median_metric(),
                item.1.formatted_fastest_metric(),
                item.1.formatted_spread_metric()
            );

            format!(
                "length={:>10} - {}",
                item.0,
                formatted_metrics
            )
        })
        .collect::<Vec<String>>()
}

fn start_window_size_dist_benchmark(
    rng: &mut StdRng
) -> Result<(), WindowError> {
    let harness_config = BenchmarkHarnessConfiguration {
        iterations: STD_BENCH_ITERATIONS,
        first_valid_result: STD_INDEX_OF_FIRST_VALID_RESULT,
        length: WINDOW_SIZE_DIST_BENCH_BENCHED_LENGTH,
        quantile: WINDOW_SIZE_DIST_BENCH_BENCHED_QUANTILE
    };

    let benchmark_header = format!(
        "Benchmark configuration - input length: {} quantile: {}",
        WINDOW_SIZE_DIST_BENCH_BENCHED_LENGTH,
        WINDOW_SIZE_DIST_BENCH_BENCHED_QUANTILE
    );
    println!("{}", benchmark_header);
    print_window_size_dist_benchmark_distributions();

    for window_size in WINDOW_SIZE_DIST_BENCH_BENCHED_WINDOW_SIZES {
        let mut benchmark_results = Vec::with_capacity(
            WINDOW_SIZE_DIST_BENCH_BENCHED_DISTRIBUTIONS.len()
        );

        for data_distributon in WINDOW_SIZE_DIST_BENCH_BENCHED_DISTRIBUTIONS {
            let case_config = BenchmarkCaseConfiguration {
                distribution: &data_distributon,
                window_size
            };

            let benchmark_result = run_std_benchmark(
                rng,
                &harness_config,
                &case_config
            )?;

            benchmark_results.push(benchmark_result);
        }

        let benchmark_header = format!(
            "Benchmark results for window size {}:",
            window_size
        );
        println!("{}", benchmark_header);
        build_window_size_dist_benchmark_table(&benchmark_results)
            .iter()
            .for_each(|result| println!("{}", result));
        println!();
    }

    Ok(())
}

fn print_window_size_dist_benchmark_distributions() {
    println!("Benchmarking - data distributions configurations:");
    for data_distribution in WINDOW_SIZE_DIST_BENCH_BENCHED_DISTRIBUTIONS {
        println!("{}", data_distribution);
    }
    println!();
}

fn build_window_size_dist_benchmark_table(
    benchmark_results: &[BenchmarkResult]
) -> Vec<String> {
    assert!(
        WINDOW_SIZE_DIST_BENCH_BENCHED_DISTRIBUTIONS.len() == benchmark_results.len(),
        "It appears that arrays of different lengths were entered."
    );

    WINDOW_SIZE_DIST_BENCH_BENCHED_DISTRIBUTIONS
        .iter()
        .zip(benchmark_results.iter())
        .map(|item| {
            let formatted_metrics = format!(
                "{} - {}",
                item.1.formatted_median_metric(),
                item.1.formatted_spread_metric()
            );

            format!(
                "{:<10} - {}",
                item.0.short_description(),
                formatted_metrics
            )
        })
        .collect::<Vec<String>>()
}

fn start_quantile_benchmark(
    rng: &mut StdRng
) -> Result<(), WindowError> {
    let mut benchmark_results = Vec::with_capacity(QUANTILE_BENCH_BENCHED_QUANTILES.len());
    for quantile in QUANTILE_BENCH_BENCHED_QUANTILES {
        let harness_config = BenchmarkHarnessConfiguration {
            iterations: STD_BENCH_ITERATIONS,
            first_valid_result: STD_INDEX_OF_FIRST_VALID_RESULT,
            length: QUANTILE_BENCH_BENCHED_LENGTH,
            quantile
        };

        let case_config = BenchmarkCaseConfiguration {
            distribution: &QUANTILE_BENCH_BENCHED_DISTRIBUTION,
            window_size: QUANTILE_BENCH_BENCHED_WINDOW_SIZE
        };

        let benchmark_result = run_std_benchmark(
            rng,
            &harness_config,
            &case_config
        )?;
        benchmark_results.push(benchmark_result);
    }

    let benchmark_header = format!(
        "Benchmark configuration - window size: {} input length: {} distribution: {}",
        QUANTILE_BENCH_BENCHED_WINDOW_SIZE,
        QUANTILE_BENCH_BENCHED_LENGTH,
        QUANTILE_BENCH_BENCHED_DISTRIBUTION.short_description()
    );
    println!("{}", benchmark_header);
    build_quantile_benchmark_table(&benchmark_results)
        .iter()
        .for_each(|result| println!("{}", result));

    Ok(())
}

fn build_quantile_benchmark_table(
    benchmark_results: &[BenchmarkResult]
) -> Vec<String>{
    assert!(
        QUANTILE_BENCH_BENCHED_QUANTILES.len() == benchmark_results.len(),
        "It appears that arrays of different lengths were entered."
    );

    QUANTILE_BENCH_BENCHED_QUANTILES
        .iter()
        .zip(benchmark_results.iter())
        .map(|item| {
            let formatted_metrics = format!(
                "{} - {} - {} - {}",
                item.1.formatted_slowest_metric(),
                item.1.formatted_median_metric(),
                item.1.formatted_fastest_metric(),
                item.1.formatted_spread_metric()
            );

            format!(
                "{:<10} - {}",
                item.0,
                formatted_metrics
            )
        })
        .collect::<Vec<String>>()
}

/// Special benchmark to determine the thresholds for function [`quantile_window::rolling_quantile_window`]
fn start_block_sizes_benchmark(
    rng: &mut StdRng
) -> Result<(), WindowError> {
    for data_distribution in BLOCK_SIZES_BENCH_BENCHED_DISTRIBUTIONS {
        let mut benchmark_results_per_window = Vec::with_capacity(
            BLOCK_SIZES_BENCH_BENCHED_WINDOW_SIZES.len()
        );

        for window_size in BLOCK_SIZES_BENCH_BENCHED_WINDOW_SIZES {
            let test_data = data_distribution
                .generate_distribution(
                    rng,
                    BLOCK_SIZES_BENCH_BENCHED_LENGTH,
                    window_size
                );

            let mut benchmark_results = vec![];

            // 16 as block size
            benchmark_results.push(
                run_block_size_benchmark::<Benchmark16BlockSize>(
                    &test_data,
                    window_size
                )?
            );

            // 32 as block size
            benchmark_results.push(
                run_block_size_benchmark::<Benchmark32BlockSize>(
                    &test_data,
                    window_size
                )?
            );

            // 64 as block size
            benchmark_results.push(
                run_block_size_benchmark::<Benchmark64BlockSize>(
                    &test_data,
                    window_size
                )?
            );

            // 128 as block size
            benchmark_results.push(
                run_block_size_benchmark::<Benchmark128BlockSize>(
                    &test_data,
                    window_size
                )?
            );

            benchmark_results_per_window.push(benchmark_results);
        }

        let benchmark_header = format!(
            "Benchmark configuration - input length: {} distribution: {}",
            BLOCK_SIZES_BENCH_BENCHED_LENGTH,
            data_distribution
        );
        println!("{}", benchmark_header);

        let benchmark_table_header = build_block_size_benchmark_table_header();
        println!("{}", benchmark_table_header);

        let benchmark_table_divider_line = build_block_size_benchmark_table_divider_line();
        println!("{}", benchmark_table_divider_line);

        build_block_size_benchmark_table(&benchmark_results_per_window)
            .iter()
            .for_each(|result| println!("{}", result));

        println!();
    }

    Ok(())
}

fn run_block_size_benchmark<B>(
    test_data: &[f64],
    window_size: usize,
) -> Result<BenchmarkResult, WindowError> where
    B: BenchmarkedFunction {
    let mut benchmark_results = run_benchmark_iterations::<B>(
        STD_BENCH_ITERATIONS,
        test_data,
        window_size,
        BLOCK_SIZES_BENCH_BENCHED_QUANTILE
    )?;

    Ok(
        build_benchmark_result(
            &mut benchmark_results,
            test_data.len(),
            STD_INDEX_OF_FIRST_VALID_RESULT
        )
    )
}

fn build_block_size_benchmark_table_header() -> String {
    BLOCK_SIZES_BENCH_BENCHED_BLOCK_SIZES
        .iter()
        .fold(format!(
            "{:<METRICS_OUTPUT_COL_FACTOR$}|",
            "window"
        ),
            |mut acc, size| {
                acc.push_str(
                    &format!(
                        "{:^METRICS_OUTPUT_COL_FACTOR$}|",
                        size
                    )
                );

                acc
            })
}

fn build_block_size_benchmark_table_divider_line() -> String {
    (0..=BLOCK_SIZES_BENCH_BENCHED_BLOCK_SIZES.len())
        .fold(String::new(),
            |mut acc, _| {
                acc.push_str(&"-".repeat(METRICS_OUTPUT_COL_FACTOR));
                acc.push('+');

                acc
            })
}

fn build_block_size_benchmark_table(
    benchmark_results: &[Vec<BenchmarkResult>]
) -> Vec<String> {
    assert!(
        BLOCK_SIZES_BENCH_BENCHED_WINDOW_SIZES.len() == benchmark_results.len(),
        "It appears that arrays of different lengths were entered."
    );

    BLOCK_SIZES_BENCH_BENCHED_WINDOW_SIZES
        .iter()
        .zip(benchmark_results.iter())
        .fold(Vec::with_capacity(BLOCK_SIZES_BENCH_BENCHED_WINDOW_SIZES.len()),
            |mut acc, item| {
                let benchmark_specific_results = item.1;
                let result_str = benchmark_specific_results
                    .iter()
                    .fold(format!(
                        "{:<METRICS_OUTPUT_COL_FACTOR$}|",
                        item.0
                    ),
                        |mut acc, res| {
                            let cell_data = format!(
                                "{:.1} ({:.1}%)",
                                res.get_median_m_per_s(),
                                res.get_spread()
                            );

                            acc.push_str(
                                &format!(
                                    "{:^METRICS_OUTPUT_COL_FACTOR$}|",
                                    cell_data
                                )
                            );

                            acc
                        });

                acc.push(result_str);
                acc
            })
}
