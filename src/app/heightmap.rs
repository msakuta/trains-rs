use eframe::egui::{Color32, Painter, Pos2, pos2};

use crate::{
    marching_squares::{Shape, border_pixel, cell_border_interpolated, pick_bits, pick_values},
    perlin_noise::{Xor128, gen_terms, perlin_noise_pixel},
};

use super::{AREA_HEIGHT, AREA_WIDTH, TrainsApp};

const NOISE_SCALE: f64 = 0.03;

const DOWNSAMPLE: usize = 10;
const DOWNSAMPLED_SHAPE: Shape = (
    (AREA_WIDTH / DOWNSAMPLE) as isize,
    (AREA_HEIGHT / DOWNSAMPLE) as isize,
);

impl TrainsApp {
    pub fn render_contours(&self, painter: &Painter, to_pos2: &impl Fn(Pos2) -> Pos2) {
        let downsampled: Vec<_> = (0..AREA_WIDTH * AREA_HEIGHT / DOWNSAMPLE / DOWNSAMPLE)
            .map(|i| {
                let x = (i % (AREA_WIDTH / DOWNSAMPLE)) * DOWNSAMPLE;
                let y = (i / (AREA_WIDTH / DOWNSAMPLE)) * DOWNSAMPLE;
                self.heightmap[x + y * AREA_WIDTH]
            })
            .collect();

        render_grid(painter, to_pos2);

        let resol = DOWNSAMPLE as f32; //self.resolution;
        for i in 0..4 {
            let level = i as f32 - 2.;
            // let offset = vec2(offset_x, offset_y);
            for cy in 0..DOWNSAMPLED_SHAPE.1 - 1 {
                let offset_y = (cy as f32 + 0.5) * resol;
                for cx in 0..DOWNSAMPLED_SHAPE.0 - 1 {
                    let offset_x = (cx as f32 + 0.5) * resol;
                    let bits = pick_bits(&downsampled, DOWNSAMPLED_SHAPE, (cx, cy), level);
                    if !border_pixel(bits) {
                        continue;
                    }
                    let values = pick_values(&downsampled, DOWNSAMPLED_SHAPE, (cx, cy), level);
                    if let Some((lines, len)) = cell_border_interpolated(bits, values) {
                        for line in lines.chunks(4).take(len / 4) {
                            let points = [
                                to_pos2(pos2(
                                    line[0] * resol * 0.5 + offset_x,
                                    line[1] * resol * 0.5 + offset_y,
                                )),
                                to_pos2(pos2(
                                    line[2] * resol * 0.5 + offset_x,
                                    line[3] * resol * 0.5 + offset_y,
                                )),
                            ];
                            painter
                                .line_segment(points, (1., Color32::from_rgb(127 + i * 32, 0, 0)));
                        }
                    }
                }
            }
        }
    }
}

fn render_grid(painter: &Painter, to_pos2: &impl Fn(Pos2) -> Pos2) {
    let right = AREA_WIDTH as f32;
    for iy in 0..AREA_HEIGHT / DOWNSAMPLE {
        let y = (iy * DOWNSAMPLE) as f32;
        painter.line_segment(
            [to_pos2(pos2(0., y)), to_pos2(pos2(right, y))],
            (1., Color32::GRAY),
        );
    }

    let bottom = AREA_HEIGHT as f32;
    for ix in 0..AREA_WIDTH / DOWNSAMPLE {
        let x = (ix * DOWNSAMPLE) as f32;
        painter.line_segment(
            [to_pos2(pos2(x, 0.)), to_pos2(pos2(x, bottom))],
            (1., Color32::GRAY),
        );
    }
}

pub(super) fn init_heightmap() -> Vec<f32> {
    let mut rng = Xor128::new(8357);
    let terms = gen_terms(&mut rng, 3);
    (0..AREA_WIDTH * AREA_HEIGHT)
        .map(|i| {
            let x = (i % AREA_WIDTH) as f64 * NOISE_SCALE;
            let y = (i / AREA_WIDTH) as f64 * NOISE_SCALE;
            perlin_noise_pixel(x, y, 3, &terms) as f32 * 10.
        })
        .collect()
}
