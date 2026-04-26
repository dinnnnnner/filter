use anyhow::Result;
use eframe::egui;
use egui_plot::{Line, Plot, PlotPoints};
use std::collections::VecDeque;
use std::f32::consts::PI;
use std::fs;

// =========================
// 1) 自己实现一个二阶 Biquad
// =========================
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f32,
    b1: f32,
    b2: f32,
    a1: f32,
    a2: f32,

    x1: f32,
    x2: f32,
    y1: f32,
    y2: f32,
}

impl Biquad {
    fn low_pass(sample_rate: f32, cutoff: f32, q: f32) -> Self {
        let omega = 2.0 * PI * cutoff / sample_rate;
        let sin = omega.sin();
        let cos = omega.cos();
        let alpha = sin / (2.0 * q);

        let b0 = (1.0 - cos) / 2.0;
        let b1 = 1.0 - cos;
        let b2 = (1.0 - cos) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos;
        let a2 = 1.0 - alpha;

        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    fn process(&mut self, x0: f32) -> f32 {
        let y0 = self.b0 * x0
            + self.b1 * self.x1
            + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;

        self.x2 = self.x1;
        self.x1 = x0;

        self.y2 = self.y1;
        self.y1 = y0;

        y0
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

// =====================================
// 2) 高阶 Butterworth：多个 Biquad 级联
// =====================================
#[derive(Debug, Clone)]
struct ButterworthCascade {
    sections: Vec<Biquad>,
}

impl ButterworthCascade {
    fn new(order: usize, sample_rate_hz: f32, cutoff_hz: f32) -> Result<Self> {
        anyhow::ensure!(order >= 2, "order must be at least 2");
        anyhow::ensure!(order % 2 == 0, "order must be even");
        anyhow::ensure!(sample_rate_hz > 0.0, "sample_rate must be positive");
        anyhow::ensure!(cutoff_hz > 0.0, "cutoff must be positive");
        anyhow::ensure!(
            cutoff_hz < sample_rate_hz * 0.5,
            "cutoff must be below Nyquist"
        );

        let section_count = order / 2;
        let mut sections = Vec::with_capacity(section_count);

        for index in 1..=section_count {
            let q = butterworth_section_q(order, index);
            sections.push(Biquad::low_pass(sample_rate_hz, cutoff_hz, q));
        }

        Ok(Self { sections })
    }

    fn process(&mut self, mut x: f32) -> f32 {
        for sec in &mut self.sections {
            x = sec.process(x);
        }
        x
    }

    fn reset(&mut self) {
        for sec in &mut self.sections {
            sec.reset();
        }
    }
}

fn butterworth_section_q(order: usize, section_index: usize) -> f32 {
    let theta = ((2 * section_index - 1) as f32 * PI) / (2.0 * order as f32);
    1.0 / (2.0 * theta.cos())
}

// =========================
// 3) 可视化 App
// =========================
struct FilterApp {
    sample_rate: f32,
    cutoff_hz: f32,
    order: usize,

    time: f32,
    dt: f32,

    raw_data: VecDeque<[f64; 2]>,
    filtered_data: VecDeque<[f64; 2]>,

    max_points: usize,

    filter: ButterworthCascade,

    low_freq_hz: f32,
    high_freq_hz: f32,
}

impl FilterApp {
    fn new() -> Self {
        let sample_rate = 200.0;
        let cutoff_hz = 5.0;
        let order = 4;
        let filter = ButterworthCascade::new(order, sample_rate, cutoff_hz).unwrap();

        Self {
            sample_rate,
            cutoff_hz,
            order,

            time: 0.0,
            dt: 1.0 / sample_rate,

            raw_data: VecDeque::new(),
            filtered_data: VecDeque::new(),

            max_points: 1000,

            filter,

            low_freq_hz: 1.0,
            high_freq_hz: 20.0,
        }
    }

    fn rebuild_filter(&mut self) {
        if let Ok(new_filter) = ButterworthCascade::new(self.order, self.sample_rate, self.cutoff_hz)
        {
            self.filter = new_filter;
            self.filtered_data.clear();
        }
    }

    fn clear_all(&mut self) {
        self.time = 0.0;
        self.raw_data.clear();
        self.filtered_data.clear();
        self.filter.reset();
    }

    fn generate_sample(&self, t: f32) -> f32 {
        // 低频信号 + 高频干扰
        let low = (2.0 * PI * self.low_freq_hz * t).sin();
        let high = 0.35 * (2.0 * PI * self.high_freq_hz * t).sin();

        low + high
    }

    fn step_simulation(&mut self) {
        // 每帧推进多个点，保证画面更流畅。
        for _ in 0..5 {
            let x = self.generate_sample(self.time);
            let y = self.filter.process(x);

            self.raw_data.push_back([self.time as f64, x as f64]);
            self.filtered_data.push_back([self.time as f64, y as f64]);

            while self.raw_data.len() > self.max_points {
                self.raw_data.pop_front();
            }
            while self.filtered_data.len() > self.max_points {
                self.filtered_data.pop_front();
            }

            self.time += self.dt;
        }
    }
}

impl eframe::App for FilterApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.step_simulation();

        egui::TopBottomPanel::top("top_controls").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("采样率:");
                let mut sr = self.sample_rate;
                if ui.add(egui::DragValue::new(&mut sr).speed(1.0)).changed() && sr > 1.0 {
                    self.sample_rate = sr;
                    self.dt = 1.0 / self.sample_rate;
                    self.rebuild_filter();
                }

                ui.separator();

                ui.label("截止频率:");
                let mut cutoff = self.cutoff_hz;
                if ui.add(egui::DragValue::new(&mut cutoff).speed(0.1)).changed()
                    && cutoff > 0.0
                    && cutoff < self.sample_rate * 0.5
                {
                    self.cutoff_hz = cutoff;
                    self.rebuild_filter();
                }

                ui.separator();

                ui.label("阶数:");
                let mut order = self.order as i32;
                if ui.add(egui::DragValue::new(&mut order).speed(1)).changed()
                    && order >= 2
                    && order % 2 == 0
                {
                    self.order = order as usize;
                    self.rebuild_filter();
                }

                ui.separator();

                if ui.button("重置").clicked() {
                    self.clear_all();
                    self.rebuild_filter();
                }
            });

            ui.horizontal(|ui| {
                ui.label("低频信号:");
                ui.add(egui::DragValue::new(&mut self.low_freq_hz).speed(0.1));

                ui.separator();

                ui.label("高频干扰:");
                ui.add(egui::DragValue::new(&mut self.high_freq_hz).speed(0.5));

                ui.separator();

                ui.label(format!(
                    "当前: 采样率 = {:.1} Hz, 截止频率 = {:.1} Hz, 阶数 = {}",
                    self.sample_rate, self.cutoff_hz, self.order
                ));
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            ui.columns(2, |columns| {
                columns[0].heading("滤波前");
                let raw_points: PlotPoints = self.raw_data.iter().copied().collect();
                let raw_line = Line::new(raw_points).name("原始信号");

                Plot::new("raw_plot")
                    .height(400.0)
                    .include_y(-2.0)
                    .include_y(2.0)
                    .show(&mut columns[0], |plot_ui| {
                        plot_ui.line(raw_line);
                    });

                columns[1].heading("滤波后");
                let filtered_points: PlotPoints = self.filtered_data.iter().copied().collect();
                let filtered_line = Line::new(filtered_points).name("滤波信号");

                Plot::new("filtered_plot")
                    .height(400.0)
                    .include_y(-2.0)
                    .include_y(2.0)
                    .show(&mut columns[1], |plot_ui| {
                        plot_ui.line(filtered_line);
                    });
            });
        });

        ctx.request_repaint();
    }
}

fn install_chinese_font(ctx: &egui::Context) {
    let font_paths = [
        "C:\\Windows\\Fonts\\msyh.ttc",
        "C:\\Windows\\Fonts\\simhei.ttf",
        "C:\\Windows\\Fonts\\simsun.ttc",
    ];

    let Some(font_data) = font_paths.iter().find_map(|path| fs::read(path).ok()) else {
        return;
    };

    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        "chinese_font".to_owned(),
        egui::FontData::from_owned(font_data),
    );

    for family in [
        egui::FontFamily::Proportional,
        egui::FontFamily::Monospace,
    ] {
        fonts
            .families
            .entry(family)
            .or_default()
            .insert(0, "chinese_font".to_owned());
    }

    ctx.set_fonts(fonts);
}

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions::default();
    eframe::run_native(
        "滤波器可视化测试",
        options,
        Box::new(|cc| {
            install_chinese_font(&cc.egui_ctx);
            Ok(Box::new(FilterApp::new()))
        }),
    )
}
