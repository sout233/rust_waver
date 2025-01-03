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
}

impl AudioVisualizer {
    fn new(sample_rate: f64) -> Self {
        Self {
            fft_planner: FftPlanner::new(),
            sample_rate,
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

                // 此处使用original_buffer_len，而不是half_len，因为half_len是fft的长度，而本大神需要显示原始的采样率
                let frequency_resolution = self.sample_rate / (original_buffer_len as f64);
                // x坐标点们，从0开始，到half_len-1结束，间隔frequency_resolution
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
                    .x_axis_formatter(|value, _| format!("{:?}", value.value)) // Format x-axis labels
                    .x_axis_position(egui_plot::VPlacement::Bottom)
                    .x_grid_spacer(egui_plot::log_grid_spacer(10))
                    .show(ui, |plot_ui| {
                        plot_ui.line(egui_plot::Line::new(PlotPoints::new(points)));
                    });
            }

            thread::sleep(std::time::Duration::from_millis(16)); // 60 FPS

            ctx.request_repaint();
        });
    }
}

fn main() {
    let host = cpal::default_host();

    let err_fn = |err| eprintln!("an error occurred on the output audio stream: {}", err);

    let device = host.default_output_device().unwrap();
    let _ = host.devices().unwrap().map(|d| {
        println!("dev name: {}", d.name().unwrap());
    });

    let mut supported_configs_range = device.supported_output_configs().unwrap();
    let supported_config = supported_configs_range
        .next()
        .unwrap()
        .with_max_sample_rate();

    let sample_format = supported_config.sample_format();
    let config: cpal::StreamConfig = supported_config.into();
    let sample_rate = config.sample_rate.0 as f64;
    println!("sample_rate: {}", sample_rate);

    thread::spawn(move || {
        let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buffer = BUFFER.lock().unwrap();

            // Downsample 但是 factor 10
            let downsampled_data: Vec<f32> = data.iter().step_by(10).copied().collect();
            buffer.extend(downsampled_data);

            let max_length: usize = (sample_rate*0.1) as usize ;

            if buffer.len() > max_length {
                let excess = buffer.len() - max_length;
                buffer.drain(0..excess);
            }
            // BUFFER.lock().unwrap().clear();
            // BUFFER.lock().unwrap().extend(buffer.iter().copied());
        };

        let stream = match sample_format {
            _ => device.build_input_stream(&config, input_data_fn, err_fn, None),
        }
        .unwrap();

        stream.play().unwrap();

        loop {
            std::thread::sleep(std::time::Duration::from_millis(1000));
        }
    });

    let app = AudioVisualizer::new(sample_rate);
    let native_options = eframe::NativeOptions::default();
    let _ = eframe::run_native(
        "Rust Waver (DEV)",
        native_options,
        Box::new(|_cc| Ok(Box::new(app))),
    );
}
