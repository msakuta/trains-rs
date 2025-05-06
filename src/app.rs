mod heightmap;

use eframe::egui::{Frame, Ui};
use heightmap::init_heightmap;

use crate::bg_image::BgImage;

pub(crate) const AREA_WIDTH: usize = 500;
pub(crate) const AREA_HEIGHT: usize = 500;
// const AREA_SHAPE: Shape = (AREA_WIDTH as isize, AREA_HEIGHT as isize);

pub(crate) struct TrainsApp {
    heightmap: Vec<f32>,
    bg: BgImage,
}

impl TrainsApp {
    pub fn new() -> Self {
        Self {
            heightmap: init_heightmap(),
            bg: BgImage::new(),
        }
    }

    fn render(&mut self, ui: &mut Ui) {
        let (response, painter) =
            ui.allocate_painter(ui.available_size(), eframe::egui::Sense::hover());

        let _ = self.bg.paint(
            &response,
            &painter,
            (),
            |_| -> Result<_, ()> {
                let min_p = self
                    .heightmap
                    .iter()
                    .fold(None, |acc, cur| {
                        if let Some(acc) = acc {
                            if acc < *cur { Some(acc) } else { Some(*cur) }
                        } else {
                            Some(*cur)
                        }
                    })
                    .ok_or(())?;
                let max_p = self
                    .heightmap
                    .iter()
                    .fold(None, |acc, cur| {
                        if let Some(acc) = acc {
                            if acc < *cur { Some(*cur) } else { Some(acc) }
                        } else {
                            Some(*cur)
                        }
                    })
                    .ok_or(())?;
                let bitmap: Vec<_> = self
                    .heightmap
                    .iter()
                    .map(|p| ((p - min_p) / (max_p - min_p) * 127. + 127.) as u8)
                    .collect();
                let img = eframe::egui::ColorImage::from_gray([AREA_WIDTH, AREA_HEIGHT], &bitmap);
                Ok(img)
            },
            [0., 0.],
            1.,
        );

        self.render_contours(&painter, &painter.clip_rect(), &|p| p);
    }
}

impl eframe::App for TrainsApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            Frame::canvas(ui.style()).show(ui, |ui| {
                self.render(ui);
            });
        });
    }
}
