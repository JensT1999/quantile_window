mod window;

use rand::prelude::*;
use std::{clone, time::Instant};
use std::f64::consts::PI;

// 50000 length und 193 windowSize
const TEST_VEC_LEN: usize = 100000000;
const WINDOW_SIZE: usize = 1000000;
const QUANTILE: f64 = 0.5;

fn main() {
    test_window();
}

fn test_window() {
    // let mut rng = StdRng::seed_from_u64(42);
    let mut rng = rand::rng();
    let mut test_vec: Vec<f64> = Vec::with_capacity(TEST_VEC_LEN);

    // for index in 0..TEST_VEC_LEN{
    //     let rand = rng.random_range(-1000.0..1000.0);
    //     test_vec.push(rand);
    // }

    // Erzeugt ein heftiges Auf und Ab innerhalb des Fensters
    // for i in 0..TEST_VEC_LEN {
    //     if i % 2 == 0 {
    //         test_vec.push(rng.random_range(900.0..1000.0)); // Ganz oben
    //     } else {
    //         test_vec.push(rng.random_range(-1000.0..-900.0)); // Ganz unten
    //     }
    // }

    // Generiert ein verrauschtes Wellenmuster
    // for i in 0..TEST_VEC_LEN {
    //     // Eine langsame Sinuswelle, die die Werteebene verschiebt
    //     let wave = (i as f64 * 0.05).sin() * 500.0;
    //     // Ein starkes, unvorhersehbares Rauschen obendrauf
    //     let noise = rng.random_range(-500.0..500.0);

    //     test_vec.push(wave + noise);
    // }

    // Der Median-Zerstörer
    for i in 0..TEST_VEC_LEN {
        // Ein linearer Trend, der den Median zwingt, permanent zu steigen
        let trend = i as f64 * 0.001;
        // Ein asymmetrisches Rauschen (Exponentialverteilung simuliert)
        // Das zieht die Daten extrem in eine Richtung (Rechtsschreibe-Effekt)
        let asymmetric_noise = (rng.random::<f64>()).ln() * -200.0;

        test_vec.push(trend + asymmetric_noise);
    }

    // test_vec.sort_by(|a, b| a.partial_cmp(&b).unwrap());

    let inst = Instant::now();
    // let mut cloned_vec = test_vec.clone();
    // let test_quantiles = gen_test_quantiles(&mut cloned_vec,
    //     WINDOW_SIZE, QUANTILE);
    let time = inst.elapsed().as_millis();

    println!("{} ms", time);

    let inst = Instant::now();
    let r = window::rolling_quantile_window(&test_vec, WINDOW_SIZE, QUANTILE).unwrap();
    let time = inst.elapsed().as_millis();

    // println!("{:?}", r);
    println!("{} ms", time);

    // assert_eq!(r, *quantile.1);

    // assert_eq!(&test_quantiles, &r);
}

fn gen_test_quantiles(input_vec: &mut [f64], window_size: usize, quantile: f64) -> Vec<f64> {
    let num_windows = input_vec.len() - window_size + 1;
    let mut result_vec = Vec::with_capacity(num_windows);
    let searched_rank = quantile * (window_size - 1) as f64;
    for windows in 0..num_windows {
        if windows == 155 {
            println!("hi");
        }

        let window_slice = &mut input_vec[windows..windows + window_size];
        let mut cloned_window = Vec::with_capacity(window_slice.len());
        window_slice.clone_into(&mut cloned_window);
        cloned_window.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let floor_rank = searched_rank.floor() as usize;

        if searched_rank % 1.0 == 0.0 {
            result_vec.push(cloned_window[floor_rank]);
        } else {
            let floor_value = cloned_window[floor_rank];
            let ceil_rank = floor_rank + 1;
            let ceil_value = cloned_window[ceil_rank];
            let interpolated_quantile = floor_value + (ceil_value - floor_value) *
                (searched_rank - searched_rank.floor());
            result_vec.push(interpolated_quantile);
        }
    }

    result_vec
}
