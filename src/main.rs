mod window;

use rand::prelude::*;
use std::{clone, time::Instant};
use std::f64::consts::PI;

// 50000 length und 193 windowSize
const TEST_VEC_LEN: usize = 1000000000;
const WINDOW_SIZE: usize = 100000000;
const QUANTILE: f64 = 0.5;

fn main() {
    test_window();
}

fn test_window() {
    // let mut rng = StdRng::seed_from_u64(42);
    let mut rng = rand::rng();
    let mut test_vec: Vec<f64> = Vec::with_capacity(TEST_VEC_LEN);
    // let test_vec = generiere_iot_daten(TEST_VEC_LEN);

    for index in 0..TEST_VEC_LEN{
        let rand = rng.random_range(-1000.0..1000.0);
        test_vec.push(rand);
    }

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
    // for i in 0..TEST_VEC_LEN {
    //     // Ein linearer Trend, der den Median zwingt, permanent zu steigen
    //     let trend = i as f64 * 0.001;
    //     // Ein asymmetrisches Rauschen (Exponentialverteilung simuliert)
    //     // Das zieht die Daten extrem in eine Richtung (Rechtsschreibe-Effekt)
    //     let asymmetric_noise = (rng.random::<f64>()).ln() * -200.0;

    //     test_vec.push(trend + asymmetric_noise);
    // }

    test_vec.sort_by(|a, b| a.partial_cmp(&b).unwrap());

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
        if windows == 784 {
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

fn generiere_realistische_daten(anzahl: usize, mittelwert: f64, std_abweichung: f64) -> Vec<f64> {
    // In rand 0.10.1 holen Sie den Generator einfach über rand::rng()
    let mut rng = rand::rng();

    // 800 MB RAM vorab fest reservieren
    let mut daten = Vec::with_capacity(anzahl);

    for _ in 0..anzahl {
        // rng.random() ersetzt das alte rng.gen() für Zahlen zwischen 0.0 und 1.0
        let mut u1: f64 = rng.random();
        let u2: f64 = rng.random();

        // Verhindert den Logarithmus von 0
        while u1 <= 0.0 {
            u1 = rng.random();
        }

        // Box-Muller-Transformation für die realistische Gauß-Glockenkurve
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        // Auf Ihren gewünschten Mittelwert und die Abweichung skalieren
        let realistischer_wert = z0 * std_abweichung + mittelwert;
        daten.push(realistischer_wert);
    }

    daten
}

fn generiere_log_normal_daten(anzahl: usize, mu: f64, sigma: f64) -> Vec<f64> {
    let mut rng = rand::rng();

    // 800 MB RAM vorab fest reservieren
    let mut daten = Vec::with_capacity(anzahl);

    for _ in 0..anzahl {
        let mut u1: f64 = rng.random();
        let u2: f64 = rng.random();

        while u1 <= 0.0 {
            u1 = rng.random();
        }

        // 1. Standard-Normalverteilung via Box-Muller
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();

        // 2. Skalierung im Log-Raum
        let normal_wert = z0 * sigma + mu;

        // 3. Transformation in den Log-Normal-Raum via Exponentialfunktion
        // Dadurch werden alle Werte strikt positiv und es entstehen die realistischen "Ausreißer" nach oben
        let log_normal_wert = normal_wert.exp();

        daten.push(log_normal_wert);
    }

    daten
}
// --- SZENARIO 1: FINANZDATEN (Log-Normaler Random Walk) ---
fn generiere_finanz_daten(anzahl: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    let mut daten = Vec::with_capacity(anzahl);
    let mut aktueller_preis = 100.0; // Startpreis einer Aktie

    for _ in 0..anzahl {
        // Box-Muller für kleinen täglichen Schock
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let z0 = if u1 > 0.0 { (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos() } else { 0.0 };

        // Der Preis multipliziert sich mit dem Schock (Prozentuale Änderung)
        let prozent_aenderung = (z0 * 0.002).exp();
        aktueller_preis *= prozent_aenderung;

        daten.push(aktueller_preis);
    }
    daten
}

// --- SZENARIO 2: IoT / SERVER-LAST (Sinus-Saisonalität + Rauschen) ---
fn generiere_iot_daten(anzahl: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    let mut daten = Vec::with_capacity(anzahl);

    // Wir simulieren z.B. Minuten-Schritte über Tage hinweg
    for i in 0..anzahl {
        // Tag/Nacht-Rhythmus (Sinuswelle, die sich alle 1440 Elemente wiederholt)
        let tageszeit_effekt = (i as f64 * 2.0 * PI / 1440.0).sin() * 20.0;
        let basis_last = 50.0;

        // Normales Grundrauschen
        let rauschen: f64 = rng.random::<f64>() * 5.0;

        // Gelegentliche extreme Lastspitzen (1% Chance auf einen "Ausschlag")
        let spitze = if rng.random_bool(0.01) {
            rng.random::<f64>() * 100.0
        } else {
            0.0
        };

        daten.push(basis_last + tageszeit_effekt + rauschen + spitze);
    }
    daten
}

// --- SZENARIO 3: NETZWERK-LATENZ (Stabil mit extremen Ausreißern) ---
fn generiere_netzwerk_daten(anzahl: usize) -> Vec<f64> {
    let mut rng = rand::rng();
    let mut daten = Vec::with_capacity(anzahl);

    for _ in 0..anzahl {
        // 98% der Pakete haben eine perfekte Latenz zwischen 10 und 15ms
        if rng.random_bool(0.98) {
            let basis_ping = 10.0 + rng.random::<f64>() * 5.0;
            daten.push(basis_ping);
        } else {
            // 2% der Pakete hängen im Timeout oder Routing-Loop (bis zu 1500ms)
            let timeout_ping = 100.0 + (rng.random::<f64>()).ln() * -300.0;
            daten.push(timeout_ping);
        }
    }
    daten
}
