mod app;
mod bg_image;
mod perlin_noise;

use app::TrainsApp;

fn main() {
    let mut native_options = eframe::NativeOptions::default();
    native_options.viewport = native_options
        .viewport
        .with_inner_size((800 as f32, 600 as f32));

    eframe::run_native(
        "trains-rs",
        native_options,
        Box::new(|_cc| Ok(Box::new(TrainsApp::new()))),
    )
    .unwrap();
}
