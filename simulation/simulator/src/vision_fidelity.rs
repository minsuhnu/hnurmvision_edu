//! Camera-facing fidelity.
//!
//! The preview window is judged by a human eye; a vision stack is judged against footage
//! from a real camera. The two differ in ways that matter to a detector: armour LEDs are
//! blown out rather than tinted, they bleed a halo into their neighbours, and the sensor
//! adds noise. Everything here is knob-gated from `[vision]` in `config.toml` and is a
//! no-op at its default value.

use bevy::prelude::*;

use crate::components::Infantry;
use crate::config::SimulationConfig;
use crate::robomaster::visibility::StatefulAppearanceCreator;

#[derive(Component)]
pub struct LampEmissiveBoosted;

/// Scale the emissive term of every vehicle's materials once, so armour lamps render at
/// the brightness a real camera sees instead of the glb's own faint glow.
///
/// The lamps as shipped peak at grey ~142, under the 150 the team's classical pipeline
/// thresholds at. That stage is not what decides whether an armour is detected — the
/// YOLOv5 network is — but it is the corner-refinement pass that runs on every detection,
/// and corners it cannot find are left at the network's own (looser) estimate. At 142 the
/// refinement never fires; at ~195 it does.
///
/// Vehicles are `WorldAssetRoot` scenes: their meshes and materials only exist after the
/// glb finishes loading, and that can happen at any point after spawn. So walk the tree
/// each frame until it actually touches a material, then mark the root — each vehicle is
/// scaled exactly once, no matter how late its assets arrive.
pub fn boost_vehicle_emissive(
    config: Res<SimulationConfig>,
    mut commands: Commands,
    mut creator: StatefulAppearanceCreator,
    vehicles: Query<Entity, (With<Infantry>, Without<LampEmissiveBoosted>)>,
) {
    let scale = config.vision.armor_emissive_scale;
    if !scale.is_finite() || scale <= 0.0 || (scale - 1.0).abs() < f32::EPSILON {
        return;
    }
    let mut missed = 0;
    for vehicle in &vehicles {
        let changed = creator.scale_emissive_tree(vehicle, scale);
        if changed > 0 {
            commands.entity(vehicle).insert(LampEmissiveBoosted);
        } else {
            missed += 1;
        }
    }
    if missed == 0 {
        info!("VISION emissive bootstrapped, scale={}", scale);
    }
}

pub fn has_unboosted_vehicles(
    config: Res<SimulationConfig>,
    vehicles: Query<(), (With<Infantry>, Without<LampEmissiveBoosted>)>,
) -> bool {
    let scale = config.vision.armor_emissive_scale;
    if !scale.is_finite() || scale <= 0.0 || (scale - 1.0).abs() < f32::EPSILON {
        return false;
    }
    !vehicles.is_empty()
}

/// Per-pixel Gaussian sensor noise, in gray levels, applied in place.
///
/// A real camera's readout is never this clean, and a stack validated only on noise-free
/// frames tends to rely on a background that is exactly constant. Sampling Box-Muller per
/// channel would cost more than the rest of the capture path, so the curve is tabulated
/// once per frame and indexed with a xorshift step.
pub fn add_sensor_noise(data: &mut [u8], sigma: f32, seed: u32) {
    if !sigma.is_finite() || sigma <= 0.0 || data.is_empty() {
        return;
    }

    const TABLE: usize = 4096;
    let mut table = [0i32; TABLE];
    let mut state = seed | 1;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        state
    };
    for slot in table.iter_mut() {
        // Two uniforms -> one normal sample; the result is kept in 1/256 gray levels.
        let u1 = (next() >> 8) as f32 / 16_777_216.0 + f32::EPSILON;
        let u2 = (next() >> 8) as f32 / 16_777_216.0;
        let normal = (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos();
        *slot = (normal * sigma * 256.0).clamp(-4096.0, 4096.0) as i32;
    }

    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        let seeds = std::array::from_fn(|_| next() | 1);
        // Runtime feature check above; the kernel checks every 8-byte slice boundary.
        unsafe { add_noise_avx2(data, &table, seeds) };
        return;
    }
    for value in data.iter_mut() {
        // The high bits of a xorshift step are the well-mixed ones; TABLE is a power of two.
        let delta = table[((next() >> 16) as usize) & (TABLE - 1)] >> 8;
        *value = (*value as i32 + delta).clamp(0, 255) as u8;
    }
}

/// Eight independent xorshift streams retain the Gaussian lookup distribution while
/// avoiding one long dependency chain over every channel of a megapixel frame.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn add_noise_avx2(data: &mut [u8], table: &[i32; 4096], mut seeds: [u32; 8]) {
    use std::arch::x86_64::*;
    unsafe {
        let mut state = _mm256_loadu_si256(seeds.as_ptr().cast());
        let mask = _mm256_set1_epi32(4095);
        let zero = _mm256_setzero_si256();
        let maximum = _mm256_set1_epi32(255);
        let mut offset = 0;
        while offset + 8 <= data.len() {
            state = _mm256_xor_si256(state, _mm256_slli_epi32(state, 13));
            state = _mm256_xor_si256(state, _mm256_srli_epi32(state, 17));
            state = _mm256_xor_si256(state, _mm256_slli_epi32(state, 5));
            let indices = _mm256_and_si256(_mm256_srli_epi32(state, 16), mask);
            let deltas = _mm256_srai_epi32(_mm256_i32gather_epi32(table.as_ptr(), indices, 4), 8);
            let bytes = _mm_loadl_epi64(data.as_ptr().add(offset).cast());
            let pixels = _mm256_cvtepu8_epi32(bytes);
            let noisy = _mm256_min_epi32(maximum, _mm256_max_epi32(zero, _mm256_add_epi32(pixels, deltas)));
            let words = _mm_packus_epi32(_mm256_castsi256_si128(noisy), _mm256_extracti128_si256(noisy, 1));
            let packed = _mm_packus_epi16(words, _mm_setzero_si128());
            _mm_storel_epi64(data.as_mut_ptr().add(offset).cast(), packed);
            offset += 8;
        }
        _mm256_storeu_si256(seeds.as_mut_ptr().cast(), state);
        let mut tail = seeds[0];
        for value in &mut data[offset..] {
            tail ^= tail << 13;
            tail ^= tail >> 17;
            tail ^= tail << 5;
            let delta = table[((tail >> 16) as usize) & 4095] >> 8;
            *value = (*value as i32 + delta).clamp(0, 255) as u8;
        }
    }
}

#[cfg(test)]
mod noise_tests {
    use super::add_sensor_noise;

    #[test]
    fn noise_keeps_strength_and_seed_behavior() {
        let mut first = vec![128; 1_000_003];
        let mut second = first.clone();
        add_sensor_noise(&mut first, 2.0, 12345);
        add_sensor_noise(&mut second, 2.0, 12345);
        assert_eq!(first, second);
        let mean = first.iter().map(|&x| x as f64 - 128.0).sum::<f64>() / first.len() as f64;
        let variance = first.iter().map(|&x| (x as f64 - 128.0 - mean).powi(2)).sum::<f64>() / first.len() as f64;
        assert!((-0.9..0.0).contains(&mean)); // Original integer lookup rounds negative deltas down.
        assert!((1.7..2.3).contains(&variance.sqrt()));
        let mut different = vec![128; first.len()];
        add_sensor_noise(&mut different, 2.0, 54321);
        assert_ne!(first, different);
    }

    #[test]
    fn noise_handles_short_tails_and_disabled_mode() {
        for n in 0..32 {
            for value in [0, 128, 255] {
                let mut pixels = vec![value; n];
                add_sensor_noise(&mut pixels, 2.0, 77);
                assert_eq!(pixels.len(), n);
                let unchanged = pixels.clone();
                add_sensor_noise(&mut pixels, 0.0, 77);
                assert_eq!(pixels, unchanged);
            }
        }
    }
}
