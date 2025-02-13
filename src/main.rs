use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use eframe::egui;
use egui_plot::{Plot, PlotPoints};
use num_complex::Complex;
use once_cell::sync::Lazy;
use rustfft::FftPlanner;
use std::{
    sync::{Arc, Mutex},
    thread,
};

static BUFFER: Lazy<Arc<Mutex<Vec<f32>>>> = Lazy::new(|| Arc::new(Mutex::new(Vec::new())));

struct AudioVisualizer {
    fft_planner: FftPlanner<f32>,
    sample_rate: f64,
    downsample_factor: usize, // 添加降采样因子字段
}

impl AudioVisualizer {
    fn new(sample_rate: f64, downsample_factor: usize) -> Self { // 添加参数
        Self {
            fft_planner: FftPlanner::new(),
            sample_rate,
            downsample_factor,
        }
    }
}

impl eframe::App for AudioVisualizer {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.heading("sout audio.");

            let buffer: Vec<f32>;
            {
                let locked_buffer = BUFFER.lock().unwrap();
                buffer = locked_buffer.clone();
            }

            if !buffer.is_empty() {
                let original_buffer_len = buffer.len();

                // FFT
                let fft = self
                    .fft_planner
                    .plan_fft(buffer.len(), rustfft::FftDirection::Forward);
                let mut buffer_complex = buffer
                    .iter()
                    .map(|&sample| Complex::new(sample, 0.0))
                    .collect::<Vec<Complex<f32>>>();
                fft.process(&mut buffer_complex);

                let half_len = buffer_complex.len() / 2;
                let buffer = buffer_complex[0..half_len]
                    .iter()
                    .map(|sample| {
                        let norm = sample.norm_sqr().sqrt() / (buffer.len() as f32).sqrt();
                        20.0 * norm.log10().max(-120.0)
                    })
                    .collect::<Vec<f32>>();

                // 使用正确的实际采样率计算频率轴
                let actual_sample_rate = self.sample_rate / self.downsample_factor as f64;
                let frequency_resolution = actual_sample_rate / original_buffer_len as f64;
                
                let x_values: Vec<f64> = (0..half_len)
                    .map(|i| i as f64 * frequency_resolution)
                    .collect();
                let y_values: Vec<f64> = buffer.iter().map(|v| *v as f64).collect();
                let points: Vec<[f64; 2]> = x_values
                    .into_iter()
                    .zip(y_values.into_iter())
                    .map(|(x, y)| [x, y * 100.0])
                    .collect();

                Plot::new("spectrum_plot")
                    .data_aspect(1.0)
                    .width(ui.available_width())
                    .x_axis_formatter(|value, _| format!("{:.1} Hz", value.value))
                    .x_axis_position(egui_plot::VPlacement::Bottom)
                    .show(ui, |plot_ui| {
                        plot_ui.line(egui_plot::Line::new(PlotPoints::new(points)));
                    });
            }

            thread::sleep(std::time::Duration::from_millis(10));

            ctx.request_repaint();
        });
    }
}

fn main() {
    let host = cpal::default_host();
    let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);
    let device = host.default_output_device().unwrap();

    let mut supported_configs_range = device.supported_output_configs().unwrap();
    let supported_config = supported_configs_range
        .next()
        .unwrap()
        .with_max_sample_rate();

    let sample_format = supported_config.sample_format();
    let config: cpal::StreamConfig = supported_config.into();
    let sample_rate = config.sample_rate.0 as f64;
    let downsample_factor = 2; // 降低采样步长改为2

    thread::spawn(move || {
        let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buffer = BUFFER.lock().unwrap();

            // 降采样因子改为2
            let downsampled_data: Vec<f32> = data.iter().step_by(downsample_factor).copied().collect();
            buffer.extend(downsampled_data);

            let max_length: usize = (sample_rate * 0.1 / downsample_factor as f64) as usize;

            if buffer.len() > max_length {
                let excess = buffer.len() - max_length;
                buffer.drain(0..excess);
            }
        };

        let stream = device.build_input_stream(&config, input_data_fn, err_fn, None).unwrap();
        stream.play().unwrap();

        loop {
            thread::sleep(std::time::Duration::from_secs(1));
        }
    });

    let app = AudioVisualizer::new(sample_rate, downsample_factor);
    let native_options = eframe::NativeOptions::default();
    let _ = eframe::run_native(
        "Rust Waver (DEV)",
        native_options,
        Box::new(|_cc| Ok(Box::new(app))),
    );
}