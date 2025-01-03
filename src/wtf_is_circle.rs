use std::cell::RefCell;
use std::sync::{Arc, Mutex, RwLock};
use std::thread;
use std::{f32::consts::PI, time::Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use iced::mouse;
use iced::widget::canvas::{self, stroke, Cache, Canvas, Geometry, Path, Stroke};
use iced::{Element, Fill, Point, Rectangle, Renderer, Subscription, Theme};
use num_complex::Complex;
use once_cell::sync::Lazy;

static BUFFER: Lazy<Arc<Mutex<Vec<f32>>>> = Lazy::new(|| Arc::new(Mutex::new(Vec::new())));
static FFT_BUFFER: Lazy<Arc<RwLock<Vec<f32>>>> = Lazy::new(|| Arc::new(RwLock::new(Vec::new())));

pub fn main() -> iced::Result {
    thread::spawn(move || {
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
        let config = supported_config.into();

        let mut fft_planer: rustfft::FftPlanner<f32> = rustfft::FftPlanner::new();

        let input_data_fn = move |data: &[f32], _: &cpal::InputCallbackInfo| {
            let mut buffer = BUFFER.lock().unwrap();
            let mut fft_buffer = FFT_BUFFER.write().unwrap();

            // Downsample 但是 factor 10
            let downsampled_data: Vec<f32> = data.iter().step_by(10).copied().collect();
            buffer.extend(downsampled_data);

            let max_length: usize = 48000 / 10;

            if buffer.len() > max_length {
                let excess = buffer.len() - max_length;
                buffer.drain(0..excess);
            }

            let fft = fft_planer.plan_fft(buffer.len(), rustfft::FftDirection::Forward);
            let mut buffer_complex = buffer
                .iter()
                .map(|&sample| Complex::new(sample, 0.0))
                .collect::<Vec<Complex<f32>>>();
            fft.process(&mut buffer_complex);

            let amplitudes: Vec<f32> = buffer_complex
                .iter()
                .map(|&x| (x.re.powi(2) + x.im.powi(2)).sqrt())
                .collect();

            fft_buffer.extend(amplitudes);
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

    iced::application("Boxin", SpectrumAnalyzer::update, SpectrumAnalyzer::view)
        .subscription(SpectrumAnalyzer::subscription)
        .theme(|_| Theme::Dark)
        .antialiasing(true)
        .run()
}

struct SpectrumAnalyzer {
    start: Instant,
    cache: Cache,
    fft_planner: RefCell<rustfft::FftPlanner<f32>>,
}

#[derive(Debug, Clone, Copy)]
enum Message {
    Tick,
}

impl SpectrumAnalyzer {
    fn update(&mut self, _: Message) {
        self.cache.clear();
    }

    fn view(&self) -> Element<Message> {
        Canvas::new(self).width(Fill).height(Fill).into()
    }

    fn subscription(&self) -> Subscription<Message> {
        iced::time::every(std::time::Duration::from_millis(16)).map(|_| Message::Tick)
    }
}

impl Default for SpectrumAnalyzer {
    fn default() -> Self {
        SpectrumAnalyzer {
            start: Instant::now(),
            cache: Cache::default(),
            fft_planner: RefCell::new(rustfft::FftPlanner::new()),
        }
    }
}

impl<Message> canvas::Program<Message> for SpectrumAnalyzer {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &Renderer,
        theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<Geometry> {
        let buffer: Vec<f32>;
        {
            let locked_buffer = BUFFER.lock().unwrap();
            buffer = locked_buffer.clone();
        }
        let buffer_len = buffer.len();

        if !buffer.is_empty() {
            let fft = self
                .fft_planner
                .borrow_mut()
                .plan_fft(buffer.len(), rustfft::FftDirection::Forward);
            let mut buffer_complex = buffer
                .iter()
                .map(|&sample| Complex::new(sample, 0.0))
                .collect::<Vec<Complex<f32>>>();
            fft.process(&mut buffer_complex);

            let half_len = buffer_complex.len() / 2;
            let original_buffer_len = buffer.len();
            let buffer = buffer_complex[0..half_len]
                .iter()
                .map(|sample| {
                    let norm = sample.norm_sqr().sqrt() / (buffer.len() as f32).sqrt();
                    20.0 * norm.log10().max(-120.0)
                })
                .collect::<Vec<f32>>();

            // 此处使用original_buffer_len，而不是half_len，因为half_len是fft的长度，而本大神需要显示原始的采样率
            let frequency_resolution = 48000.0 / (original_buffer_len as f64);
            // x坐标点们，从0开始，到half_len-1结束，间隔frequency_resolution
            let x_values: Vec<f64> = (0..half_len)
                .map(|i| i as f64 * frequency_resolution)
                .collect();
            let y_values: Vec<f64> = buffer.iter().map(|v| *v as f64).collect();

            let geometry = self.cache.draw(renderer, bounds.size(), |frame| {
                let palette = theme.palette();

                let width = frame.width();
                let height = frame.height();
                let mid_height = height / 2.0;

                let elapsed = (self.start.elapsed().as_millis() % 10_000) as f32;

                let points: Vec<Point> =x_values
                   .iter()
                   .zip(y_values.iter())
                   .map(|(x, y)| {
                        let x = *x as f32;
                        let y = *y as f32;
                        let angle = 2.0 * PI * (x / 48000.0 + 1.0 / 1000.0);
                        let radius = y.min(mid_height) + 1.0;
                        let x = width / 2.0 + radius * angle.cos();
                        let y = mid_height - radius * angle.sin();
                        Point::new(x, y)
                    })
                   .collect();

                let path = Path::new(|builder| {
                    if let Some(first_point) = points.first() {
                        builder.move_to(*first_point);
                    }
                    for point in points.iter().skip(1) {
                        builder.line_to(*point);
                    }
                });

                frame.stroke(
                    &path,
                    Stroke {
                        style: stroke::Style::Solid(palette.text),
                        width: 2.0, // 设置线条宽度
                        ..Stroke::default()
                    },
                );
            });

            vec![geometry]
        } else {
            vec![]
        }
    }
}
