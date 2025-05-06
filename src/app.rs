mod heightmap;

use self::heightmap::init_heightmap;
use eframe::egui::{Frame, Ui, pos2};

use crate::{
    bg_image::BgImage,
    transform::{Transform, half_rect},
};

pub(crate) const AREA_WIDTH: usize = 500;
pub(crate) const AREA_HEIGHT: usize = 500;
// const AREA_SHAPE: Shape = (AREA_WIDTH as isize, AREA_HEIGHT as isize);

pub(crate) struct TrainsApp {
    transform: Transform,
    heightmap: Vec<f32>,
    bg: BgImage,
    show_contours: bool,
    show_grid: bool,
}

impl TrainsApp {
    pub fn new() -> Self {
        Self {
            transform: Transform::new(1.),
            heightmap: init_heightmap(),
            bg: BgImage::new(),
            show_contours: true,
            show_grid: false,
        }
    }

    fn render(&mut self, ui: &mut Ui) {
        let (response, painter) =
            ui.allocate_painter(ui.available_size(), eframe::egui::Sense::hover());

        if ui.ui_contains_pointer() {
            ui.input(|i| self.transform.handle_mouse(i, half_rect(&response.rect)));
        }

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
            self.transform.offset_scr(),
            self.transform.scale(),
        );

        self.render_contours(&painter, &|p| {
            let p = self.transform.transform_point(p);
            pos2(p.x, AREA_HEIGHT as f32 * self.transform.scale() - p.y)
                + painter.clip_rect().left_top().to_vec2()
        });
    }

    fn ui_panel(&mut self, ui: &mut Ui) {
        ui.checkbox(&mut self.show_contours, "Show contour lines");
        ui.checkbox(&mut self.show_grid, "Show grid");
    }
}

impl eframe::App for TrainsApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        eframe::egui::SidePanel::right("side_panel")
            .min_width(200.)
            .show(ctx, |ui| self.ui_panel(ui));

        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            Frame::canvas(ui.style()).show(ui, |ui| {
                self.render(ui);
            });
        });
    }
}
