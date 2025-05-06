use eframe::egui::{Frame, Ui};

use crate::{
    bg_image::BgImage,
    perlin_noise::{Xor128, gen_terms, perlin_noise_pixel},
};

pub(crate) const AREA_WIDTH: usize = 500;
pub(crate) const AREA_HEIGHT: usize = 500;
const NOISE_SCALE: f64 = 0.1;

pub(crate) struct TrainsApp {
    heightmap: Vec<f32>,
    bg: BgImage,
}

impl TrainsApp {
    pub fn new() -> Self {
        let mut rng = Xor128::new(8357);
        let terms = gen_terms(&mut rng, 3);
        Self {
            heightmap: (0..AREA_WIDTH * AREA_HEIGHT)
                .map(|i| {
                    let x = (i % AREA_WIDTH) as f64 * NOISE_SCALE;
                    let y = (i / AREA_WIDTH) as f64 * NOISE_SCALE;
                    perlin_noise_pixel(x, y, 3, &terms) as f32
                })
                .collect(),
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
                println!("heightmap: {:?}", &self.heightmap[..100]);
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
                dbg!(min_p, max_p);
                let bitmap: Vec<_> = self
                    .heightmap
                    .iter()
                    .map(|p| ((p - min_p) / (max_p - min_p) * 127.) as u8)
                    .collect();
                println!("bitmap: {:?}", &bitmap[..100]);
                let img = eframe::egui::ColorImage::from_gray([AREA_WIDTH, AREA_HEIGHT], &bitmap);
                Ok(img)
            },
            [0., 0.],
            1.,
        );
        //     to_screen.transform_rect(Rect::from_min_size(
        //         egui::pos2(
        //             (x as f32 + xofs) * CELL_SIZE_F,
        //             (y as f32 + yofs) * CELL_SIZE_F,
        //         ),
        //         Vec2::splat(CELL_SIZE_F * 0.5),
        //     )),
        //     tex_rect,
        //     Color32::WHITE,
        // );
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
