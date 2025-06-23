mod app;
mod bg_image;
mod marching_squares;
mod path_utils;
mod perlin_noise;
mod train;
mod train_tracks;
mod transform;
mod vec2;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use app::TrainsApp;

#[cfg(not(target_arch = "wasm32"))]
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

// when compiling to web using trunk.
#[cfg(target_arch = "wasm32")]
fn main() {
    use wasm_bindgen::JsCast;
    // Make sure panics are logged using `console.error`.
    console_error_panic_hook::set_once();

    // Redirect tracing to console.log and friends:
    tracing_wasm::set_as_global_default();

    #[derive(Clone)]
    #[wasm_bindgen]
    struct WebHandle {
        runner: eframe::WebRunner,
    }

    impl WebHandle {
        pub fn new() -> Self {
            Self {
                runner: eframe::WebRunner::new(),
            }
        }

        pub async fn start(
            &self,
            canvas: web_sys::HtmlCanvasElement,
        ) -> Result<(), wasm_bindgen::JsValue> {
            self.runner
                .start(
                    canvas,
                    eframe::WebOptions::default(),
                    Box::new(|_| Ok(Box::new(TrainsApp::new()))),
                )
                .await
        }
    }

    wasm_bindgen_futures::spawn_local(async {
        let canvas = eframe::web_sys::window()
            .expect("no global `window` exists")
            .document()
            .expect("should have a document")
            .get_element_by_id("the_canvas_id")
            .expect("should have #the_canvas_id on the page")
            .dyn_into::<eframe::web_sys::HtmlCanvasElement>()
            .expect("#the_canvas_id should be a <canvas> element");
        WebHandle::new()
            .start(canvas)
            .await
            .expect("failed to start eframe");
    });
}
