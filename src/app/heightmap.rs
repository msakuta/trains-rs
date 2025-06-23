mod noise_expr;
mod parser;

use std::collections::HashMap;

use eframe::egui::{self, Color32, ColorImage, Painter, Pos2, Ui, pos2};
use serde::{Deserialize, Serialize};

use crate::{
    marching_squares::{
        Idx, Shape, border_pixel, cell_border_interpolated, pick_bits, pick_values,
    },
    perlin_noise::Xorshift64Star,
    vec2::Vec2,
};

use super::{AREA_HEIGHT, AREA_WIDTH, TrainsApp};

use self::{
    noise_expr::{Value, precompute, run},
    parser::parse,
};

const DEFAULT_PERSISTENCE_OCTAVES: u32 = 3;

const DEFAULT_NOISE_OCTAVES: u32 = 4;

const DEFAULT_WATER_LEVEL: f32 = 0.05;

const BRIDGE_HEIGHT: f64 = 0.1;

pub(super) const DOWNSAMPLE: usize = 16;

#[derive(PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum NoiseType {
    White,
    Perlin,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct HeightMapParams {
    pub noise_type: NoiseType,
    pub width: usize,
    pub height: usize,
    pub seed: u64,
    seed_buf: String,
    pub water_level: f32,
    pub noise_expr: String,
}

impl HeightMapParams {
    pub(super) fn new() -> Self {
        let seed = 8357;
        let seed_buf = seed.to_string();
        Self {
            noise_type: NoiseType::Perlin,
            width: AREA_WIDTH,
            height: AREA_HEIGHT,
            seed,
            seed_buf,
            water_level: DEFAULT_WATER_LEVEL,
            noise_expr: format!(
                "scaled_x = x * 0.05;
octaves = {};
abs_rounding = 0.1;
height_scale = 10;
pers = perlin_noise(scaled_x, {}, 0.5);

softmax(
  softabs(
    perlin_noise(scaled_x, octaves, pers),
    abs_rounding
  ),
  {} - softclamp(
    softabs(
      perlin_noise(scaled_x * 0.5, octaves, pers),
      abs_rounding
    ),
    abs_rounding
  )
) * height_scale",
                DEFAULT_NOISE_OCTAVES, DEFAULT_PERSISTENCE_OCTAVES, BRIDGE_HEIGHT
            ),
        }
    }

    pub(super) fn params_ui(&mut self, ui: &mut Ui) {
        ui.group(|ui| {
            ui.radio_value(&mut self.noise_type, NoiseType::Perlin, "Perlin");
            ui.radio_value(&mut self.noise_type, NoiseType::White, "White");
        });
        ui.horizontal(|ui| {
            ui.label("Width:");
            ui.add(egui::Slider::new(&mut self.width, 1..=1024));
        });
        ui.horizontal(|ui| {
            ui.label("Height:");
            ui.add(egui::Slider::new(&mut self.height, 1..=1024));
        });
        ui.horizontal(|ui| {
            ui.label("Seed:");
            if ui.text_edit_singleline(&mut self.seed_buf).changed() {
                if let Ok(val) = self.seed_buf.parse() {
                    self.seed = val;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Water level:");
            ui.add(egui::Slider::new(&mut self.water_level, (0.)..=1.));
        });
        ui.label("Noise expression:");
        ui.add(
            egui::TextEdit::multiline(&mut self.noise_expr)
                .font(egui::TextStyle::Monospace)
                .code_editor()
                .desired_rows(10)
                .lock_focus(true)
                .desired_width(f32::INFINITY),
        );
    }
}

pub(crate) struct HeightMap {
    pub(super) map: Vec<f32>,
    pub(super) shape: Shape,
    pub(super) water_level: f32,
}

impl HeightMap {
    pub(super) fn new(params: &HeightMapParams) -> Result<Self, String> {
        let mut rng = Xorshift64Star::new(params.seed);

        let mut ast = parse(&params.noise_expr)?;
        precompute(&mut ast, &mut rng)?;

        let map: Vec<_> = (0..params.width * params.height)
            .map(|i| {
                let ix = (i % params.width) as f64;
                let iy = (i / params.width) as f64;
                let pos = crate::vec2::Vec2::new(ix as f64, iy as f64);
                let Value::Scalar(eval_res) = run(&ast, &pos)? else {
                    return Err("Eval result was not a scalar".to_string());
                };
                let val = eval_res;
                Ok(val as f32)
            })
            .collect::<Result<_, _>>()?;

        let min_p = map
            .iter()
            .fold(None, |acc, cur| {
                if let Some(acc) = acc {
                    if acc < *cur { Some(acc) } else { Some(*cur) }
                } else {
                    Some(*cur)
                }
            })
            .ok_or_else(|| "Min value not found".to_string())?;
        let max_p = map
            .iter()
            .fold(None, |acc, cur| {
                if let Some(acc) = acc {
                    if acc < *cur { Some(*cur) } else { Some(acc) }
                } else {
                    Some(*cur)
                }
            })
            .ok_or_else(|| "Max value not found".to_string())?;
        let water_level = (max_p - min_p) * params.water_level + min_p;

        Ok(Self {
            map,
            shape: (params.width as isize, params.height as isize),
            water_level,
        })
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
            .enumerate()
            .map(|(i, p)| {
                let is_water = *p < self.water_level;
                if is_water {
                    let water_depth = (self.water_level - p) / (self.water_level - min_p);
                    let inten = 1. / (1. + 0.5 * water_depth);
                    [0, (95. * inten) as u8, (191. * inten) as u8]
                } else {
                    let above_water = (p - self.water_level) / (max_p - self.water_level);
                    let white = above_water.powi(4) as f64;
                    let x = (i % self.shape.0 as usize) as f64;
                    let y = (i / self.shape.1 as usize) as f64;
                    let grad = self.gradient(&Vec2::new(x, y));
                    let dot = (grad.x - grad.y) * 10.;
                    let diffuse = (dot + 1.) / 2.5;
                    let greenness = (1. - white) / (1. + 1000. * grad.length2());
                    let redness = 0.5 - greenness * 0.5;
                    let red = ((diffuse * (1. + redness - 0.5 * greenness as f64) + 0.2)
                        .clamp(0., 1.)
                        * 255.) as u8;
                    let blue = ((diffuse * (1. - 0.5 * greenness as f64) + 0.2).clamp(0., 1.)
                        * 255.) as u8;
                    let green = ((diffuse + 0.2).clamp(0., 1.) * 255.) as u8;
                    [red, green, blue]
                }
                .into_iter()
            })
            .flatten()
            .collect();
        // let _ = image::save_buffer(
        //     "noise.png",
        //     &bitmap,
        //     self.shape.0 as u32,
        //     self.shape.1 as u32,
        //     image::ColorType::L8,
        // );
        let img = eframe::egui::ColorImage::from_rgb(
            [self.shape.0 as usize, self.shape.1 as usize],
            &bitmap,
        );
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

    pub(crate) fn is_water(&self, pos: &crate::vec2::Vec2<f64>) -> bool {
        let [x, y] = [pos.x as isize, pos.y as isize];
        if x < 0 || self.shape.0 <= x || y < 0 || self.shape.1 < y {
            return false;
        }
        self.map[self.shape.idx(x, y)] < self.water_level
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

pub struct ContoursCache {
    grid_step: usize,
    contours: HashMap<i32, Vec<[Pos2; 2]>>,
}

impl ContoursCache {
    fn new(grid_step: usize) -> Self {
        Self {
            grid_step,
            contours: HashMap::new(),
        }
    }

    pub(super) fn grid_step(&self) -> usize {
        self.grid_step
    }
}

impl TrainsApp {
    pub fn render_contours(
        &self,
        painter: &Painter,
        to_pos2: &impl Fn(Pos2) -> Pos2,
        contour_grid_step: usize,
    ) {
        if self.show_grid {
            render_grid(painter, &self.heightmap.shape, to_pos2, contour_grid_step);
        }

        if !self.show_contours {
            return;
        }

        self.heightmap
            .process_contours(contour_grid_step, |level, points| {
                let points = [to_pos2(points[0]), to_pos2(points[1])];

                let line_width = if level % 4 == 0 { 1.5 } else { 1. };
                let r = 127 + (level * 32).wrapping_rem_euclid(128);

                painter.line_segment(points, (line_width, Color32::from_rgb(r as u8, 0, 0)));
            });
    }

    pub fn render_contours_with_cache(
        &self,
        painter: &Painter,
        to_pos2: &impl Fn(Pos2) -> Pos2,
        contour_grid_step: usize,
    ) {
        if self.show_grid {
            render_grid(painter, &self.heightmap.shape, to_pos2, contour_grid_step);
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
    pub fn cache_contours(&self, contour_grid_step: usize) -> ContoursCache {
        let mut ret = ContoursCache::new(contour_grid_step);
        self.process_contours(contour_grid_step, |level, points| {
            ret.contours.entry(level).or_default().push(*points);
        });

        ret
    }

    pub fn render_with_cache(
        painter: &Painter,
        cache: &ContoursCache,
        to_pos2: &impl Fn(Pos2) -> Pos2,
    ) {
        for (level, contours) in &cache.contours {
            for points in contours {
                let points = [to_pos2(points[0]), to_pos2(points[1])];

                let line_width = if level % 4 == 0 { 1.5 } else { 1. };
                let r = 127 + (level * 32).wrapping_rem_euclid(128);

                painter.line_segment(points, (line_width, Color32::from_rgb(r as u8, 0, 0)));
            }
        }
    }

    fn process_contours(&self, contour_grid_step: usize, mut f: impl FnMut(i32, &[Pos2; 2])) {
        let downsampled: Vec<_> = (0..self.shape.0 as usize * self.shape.1 as usize
            / contour_grid_step
            / contour_grid_step)
            .map(|i| {
                let x = (i % (self.shape.0 as usize / contour_grid_step)) * contour_grid_step;
                let y = (i / (self.shape.0 as usize / contour_grid_step)) * contour_grid_step;
                self[(x as isize, y as isize)]
            })
            .collect();

        let resol = contour_grid_step as f32;

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

        let downsampled_shape = (
            self.shape.0 / contour_grid_step as isize,
            self.shape.1 / contour_grid_step as isize,
        );

        for i in min_i / incr..max_i / incr {
            let level = (i * incr) as f32;
            // let offset = vec2(offset_x, offset_y);
            for cy in 0..downsampled_shape.1 - 1 {
                let offset_y = (cy as f32 + 0.5) * resol;
                for cx in 0..downsampled_shape.0 - 1 {
                    let offset_x = (cx as f32 + 0.5) * resol;
                    let bits = pick_bits(&downsampled, downsampled_shape, (cx, cy), level);
                    if !border_pixel(bits) {
                        continue;
                    }
                    let values = pick_values(&downsampled, downsampled_shape, (cx, cy), level);
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

fn render_grid(
    painter: &Painter,
    shape: &Shape,
    to_pos2: &impl Fn(Pos2) -> Pos2,
    contour_grid_step: usize,
) {
    let right = shape.0 as f32;
    for iy in 0..shape.1 as usize / contour_grid_step {
        let y = (iy * contour_grid_step) as f32;
        painter.line_segment(
            [to_pos2(pos2(0., y)), to_pos2(pos2(right, y))],
            (1., Color32::GRAY),
        );
    }

    let bottom = shape.1 as f32;
    for ix in 0..shape.0 as usize / contour_grid_step {
        let x = (ix * contour_grid_step) as f32;
        painter.line_segment(
            [to_pos2(pos2(x, 0.)), to_pos2(pos2(x, bottom))],
            (1., Color32::GRAY),
        );
    }
}

/// A function that converges to max(a, b) when the ratio of a and b are great,
/// but acts as a smooth interpolation between them when they are similar.
/// No, it doesn't help to avoid sharp edges, but it helps to blend 2 heightmaps without clear gaps.
fn softmax(a: f64, b: f64) -> f64 {
    let denom = a + b;
    (a * a + b * b) / denom
}

/// A function that acts like abs in a great input, but wraps smoothly around 0 using quadratic function.
/// The offset is subtracted to make C1 continuity.
fn softabs(a: f64, rounding: f64) -> f64 {
    if a.abs() < rounding {
        (a * a / rounding) / 2.
    } else {
        a.abs() - rounding / 2.
    }
}

fn softclamp(x: f64, max: f64) -> f64 {
    (x / max).tanh() * max
}
