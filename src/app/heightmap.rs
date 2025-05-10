use eframe::egui::{Color32, ColorImage, Painter, Pos2, pos2};

use crate::{
    marching_squares::{
        Idx, Shape, border_pixel, cell_border_interpolated, pick_bits, pick_values,
    },
    perlin_noise::{Xor128, gen_terms, perlin_noise_pixel},
};

use super::{AREA_HEIGHT, AREA_WIDTH, TrainsApp};

const NOISE_BITS: u32 = 4;
const NOISE_SCALE: f64 = 0.03;

const DOWNSAMPLE: usize = 10;
const DOWNSAMPLED_SHAPE: Shape = (
    (AREA_WIDTH / DOWNSAMPLE) as isize,
    (AREA_HEIGHT / DOWNSAMPLE) as isize,
);

pub(crate) struct HeightMap {
    pub(super) map: Vec<f32>,
    pub(super) shape: Shape,
}

impl HeightMap {
    fn new(map: Vec<f32>, shape: Shape) -> Self {
        Self { map, shape }
    }

    pub fn get_image(&self) -> Result<ColorImage, ()> {
        let min_p = self
            .map
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
            .map
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
            .map
            .iter()
            .map(|p| ((p - min_p) / (max_p - min_p) * 127. + 127.) as u8)
            .collect();
        let img = eframe::egui::ColorImage::from_gray([AREA_WIDTH, AREA_HEIGHT], &bitmap);
        Ok(img)
    }

    pub(crate) fn gradient(&self, pos: &crate::vec2::Vec2<f64>) -> crate::vec2::Vec2<f64> {
        let [x, y] = [pos.x as isize, pos.y as isize];
        if x < 0 || self.shape.0 <= x || y < 0 || self.shape.1 < y {
            return crate::vec2::Vec2::zero();
        }

        let dx = self.map[self.shape.idx(x + 1, y)] - self.map[self.shape.idx(x, y)];
        let dy = self.map[self.shape.idx(x, y + 1)] - self.map[self.shape.idx(x, y)];

        crate::vec2::Vec2::new(dx as f64, dy as f64)

        // let [fx, fy] = [pos.x % 1., pos.y % 1.];

        // let lerp = |a, b, f| (1. - f) * a + f * b;
        // lerp(lerp(heightmap[shape.idx(x, y)], heightmap[shape.idx(x + 1, y)], fx),
        // lerp(heightmap[shape.idx(x, y + 1)], heightmap[shape.idx(x + 1, y + 1)], fx), fy)
    }
}

impl std::ops::Index<(isize, isize)> for HeightMap {
    type Output = f32;
    fn index(&self, index: (isize, isize)) -> &Self::Output {
        &self.map[self.shape.idx(index.0, index.1)]
    }
}

impl std::ops::IndexMut<(isize, isize)> for HeightMap {
    fn index_mut(&mut self, index: (isize, isize)) -> &mut Self::Output {
        &mut self.map[self.shape.idx(index.0, index.1)]
    }
}

impl TrainsApp {
    pub fn render_contours(&self, painter: &Painter, to_pos2: &impl Fn(Pos2) -> Pos2) {
        let downsampled: Vec<_> = (0..AREA_WIDTH * AREA_HEIGHT / DOWNSAMPLE / DOWNSAMPLE)
            .map(|i| {
                let x = (i % (AREA_WIDTH / DOWNSAMPLE)) * DOWNSAMPLE;
                let y = (i / (AREA_WIDTH / DOWNSAMPLE)) * DOWNSAMPLE;
                self.heightmap[(x as isize, y as isize)]
            })
            .collect();

        if self.show_grid {
            render_grid(painter, to_pos2);
        }

        if !self.show_contours {
            return;
        }

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

pub(super) fn init_heightmap() -> HeightMap {
    let mut rng = Xor128::new(8357);
    let terms = gen_terms(&mut rng, NOISE_BITS);
    HeightMap::new(
        (0..AREA_WIDTH * AREA_HEIGHT)
            .map(|i| {
                let x = (i % AREA_WIDTH) as f64 * NOISE_SCALE;
                let y = (i / AREA_WIDTH) as f64 * NOISE_SCALE;
                perlin_noise_pixel(x, y, NOISE_BITS, &terms) as f32 * 10.
            })
            .collect(),
        (AREA_WIDTH as isize, AREA_HEIGHT as isize),
    )
}
