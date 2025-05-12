use std::collections::HashMap;

use eframe::egui::{self, Color32, ColorImage, Painter, Pos2, Ui, pos2};

use crate::{
    marching_squares::{
        Idx, Shape, border_pixel, cell_border_interpolated, pick_bits, pick_values,
    },
    perlin_noise::{Xorshift64Star, gen_seeds, perlin_noise_pixel, white_noise},
};

use super::{AREA_HEIGHT, AREA_WIDTH, TrainsApp};

const MAX_PERSISTENCE_OCTAVES: u32 = 10;
const DEFAULT_PERSISTENCE_OCTAVES: u32 = 3;

const MAX_PERSISTENCE_SCALE: f64 = 2.;
/// Meaning the default value for the minimum of persistence value.
const DEFAULT_MIN_PERSISTENCE: f64 = 0.1;
const DEFAULT_PERSISTENCE_SCALE: f64 = 1.;
const MAX_MIN_PERSISTENCE: f64 = 1.;

const MAX_PERSISTENCE_NOISE_SCALE: f64 = 0.5;
const MIN_PERSISTENCE_NOISE_SCALE: f64 = 0.01;
const DEFAULT_PERSISTENCE_NOISE_SCALE: f64 = 0.01;

const MAX_NOISE_OCTAVES: u32 = 10;
const DEFAULT_NOISE_OCTAVES: u32 = 4;

const MAX_NOISE_SCALE: f64 = 0.5;
const MIN_NOISE_SCALE: f64 = 0.01;
const DEFAULT_NOISE_SCALE: f64 = 0.05;

const DEFAULT_HEIGHT_SCALE: f64 = 10.;
const MAX_HEIGHT_SCALE: f64 = 50.;

const DOWNSAMPLE: usize = 10;
const DOWNSAMPLED_SHAPE: Shape = (
    (AREA_WIDTH / DOWNSAMPLE) as isize,
    (AREA_HEIGHT / DOWNSAMPLE) as isize,
);

#[derive(PartialEq, Eq)]
pub(crate) enum NoiseType {
    White,
    Perlin,
}

pub(crate) struct HeightMapParams {
    pub noise_type: NoiseType,
    pub persistence_octaves: u32,
    pub persistence_noise_scale: f64,
    pub persistence_scale: f64,
    pub min_persistence: f64,
    pub noise_octaves: u32,
    pub noise_scale: f64,
    pub height_scale: f64,
}

impl HeightMapParams {
    pub(super) fn new() -> Self {
        Self {
            noise_type: NoiseType::Perlin,
            persistence_octaves: DEFAULT_PERSISTENCE_OCTAVES,
            persistence_noise_scale: DEFAULT_PERSISTENCE_NOISE_SCALE,
            persistence_scale: DEFAULT_PERSISTENCE_SCALE,
            min_persistence: DEFAULT_MIN_PERSISTENCE,
            noise_octaves: DEFAULT_NOISE_OCTAVES,
            noise_scale: DEFAULT_NOISE_SCALE,
            height_scale: DEFAULT_HEIGHT_SCALE,
        }
    }

    pub(super) fn params_ui(&mut self, ui: &mut Ui) {
        ui.group(|ui| {
            ui.radio_value(&mut self.noise_type, NoiseType::Perlin, "Perlin");
            ui.radio_value(&mut self.noise_type, NoiseType::White, "White");
        });
        ui.horizontal(|ui| {
            ui.label("Persistence noise octaves:");
            ui.add(egui::Slider::new(
                &mut self.persistence_octaves,
                1..=MAX_PERSISTENCE_OCTAVES,
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Persistence noise scale:");
            ui.add(egui::Slider::new(
                &mut self.persistence_noise_scale,
                MIN_PERSISTENCE_NOISE_SCALE..=MAX_PERSISTENCE_NOISE_SCALE,
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Persistence scale:");
            ui.add(egui::Slider::new(
                &mut self.persistence_scale,
                (0.)..=MAX_PERSISTENCE_SCALE,
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Min persistence:");
            ui.add(egui::Slider::new(
                &mut self.min_persistence,
                (0.)..=MAX_MIN_PERSISTENCE,
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Noise octaves:");
            ui.add(egui::Slider::new(
                &mut self.noise_octaves,
                1..=MAX_NOISE_OCTAVES,
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Noise scale:");
            ui.add(egui::Slider::new(
                &mut self.noise_scale,
                MIN_NOISE_SCALE..=MAX_NOISE_SCALE,
            ));
        });
        ui.horizontal(|ui| {
            ui.label("Height scale:");
            ui.add(egui::Slider::new(
                &mut self.height_scale,
                (0.)..=MAX_HEIGHT_SCALE,
            ));
        });
    }
}

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

pub type ContoursCache = HashMap<i32, Vec<[Pos2; 2]>>;

impl TrainsApp {
    pub fn render_contours(&self, painter: &Painter, to_pos2: &impl Fn(Pos2) -> Pos2) {
        if self.show_grid {
            render_grid(painter, to_pos2);
        }

        if !self.show_contours {
            return;
        }

        self.heightmap.process_contours(|level, points| {
            let points = [to_pos2(points[0]), to_pos2(points[1])];

            let line_width = if level % 4 == 0 { 1.5 } else { 1. };
            let r = 127 + (level * 32).wrapping_rem_euclid(128);

            painter.line_segment(points, (line_width, Color32::from_rgb(r as u8, 0, 0)));
        });
    }

    pub fn render_contours_with_cache(&self, painter: &Painter, to_pos2: &impl Fn(Pos2) -> Pos2) {
        if self.show_grid {
            render_grid(painter, to_pos2);
        }

        if !self.show_contours {
            return;
        }

        if let Some(cache) = &self.contours_cache {
            HeightMap::render_with_cache(painter, cache, to_pos2);
        }
    }
}

impl HeightMap {
    pub fn cache_contours(&self) -> ContoursCache {
        let mut ret = ContoursCache::new();
        self.process_contours(|level, points| {
            ret.entry(level).or_default().push(*points);
        });

        ret
    }

    pub fn render_with_cache(
        painter: &Painter,
        cache: &ContoursCache,
        to_pos2: &impl Fn(Pos2) -> Pos2,
    ) {
        for (level, contours) in cache {
            for points in contours {
                let points = [to_pos2(points[0]), to_pos2(points[1])];

                let line_width = if level % 4 == 0 { 1.5 } else { 1. };
                let r = 127 + (level * 32).wrapping_rem_euclid(128);

                painter.line_segment(points, (line_width, Color32::from_rgb(r as u8, 0, 0)));
            }
        }
    }

    fn process_contours(&self, mut f: impl FnMut(i32, &[Pos2; 2])) {
        let downsampled: Vec<_> = (0..AREA_WIDTH * AREA_HEIGHT / DOWNSAMPLE / DOWNSAMPLE)
            .map(|i| {
                let x = (i % (AREA_WIDTH / DOWNSAMPLE)) * DOWNSAMPLE;
                let y = (i / (AREA_WIDTH / DOWNSAMPLE)) * DOWNSAMPLE;
                self[(x as isize, y as isize)]
            })
            .collect();

        let resol = DOWNSAMPLE as f32; //self.resolution;

        let minmax_contour = downsampled
            .iter()
            .fold(None, |acc: Option<(f32, f32)>, cur| {
                if let Some(acc) = acc {
                    Some((acc.0.min(*cur), acc.1.max(*cur)))
                } else {
                    Some((*cur, *cur))
                }
            });

        let Some(minmax_contour) = minmax_contour else {
            return;
        };

        let min_i = minmax_contour.0.ceil() as i32;
        let max_i = minmax_contour.1.ceil() as i32;

        let num_levels = max_i - min_i;
        let incr = num_levels / 10 + 1;

        for i in min_i / incr..max_i / incr {
            let level = (i * incr) as f32;
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
                                pos2(
                                    line[0] * resol * 0.5 + offset_x,
                                    line[1] * resol * 0.5 + offset_y,
                                ),
                                pos2(
                                    line[2] * resol * 0.5 + offset_x,
                                    line[3] * resol * 0.5 + offset_y,
                                ),
                            ];
                            f(i, &points);
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

pub(super) fn init_heightmap(params: &HeightMapParams) -> HeightMap {
    let mut rng = Xorshift64Star::new(8357);

    let persistence_seeds = gen_seeds(&mut rng, params.persistence_octaves);
    let seeds = gen_seeds(&mut rng, params.noise_octaves);
    HeightMap::new(
        (0..AREA_WIDTH * AREA_HEIGHT)
            .map(|i| {
                let ix = (i % AREA_WIDTH) as f64;
                let iy = (i / AREA_WIDTH) as f64;
                let p_pos =
                    crate::vec2::Vec2::new(ix as f64, iy as f64) * params.persistence_noise_scale;
                let persistence_sample = perlin_noise_pixel(
                    p_pos.x,
                    p_pos.y,
                    params.persistence_octaves,
                    &persistence_seeds,
                    0.5,
                )
                .abs()
                    * params.persistence_scale
                    + params.min_persistence;
                let pos = crate::vec2::Vec2::new(ix as f64, iy as f64) * params.noise_scale;
                let val = match params.noise_type {
                    NoiseType::Perlin => perlin_noise_pixel(
                        pos.x,
                        pos.y,
                        params.noise_octaves,
                        &seeds,
                        persistence_sample,
                    ),
                    NoiseType::White => white_noise(pos.x, pos.y, seeds[0]),
                };
                val as f32 * params.height_scale as f32
            })
            .collect(),
        (AREA_WIDTH as isize, AREA_HEIGHT as isize),
    )
}
