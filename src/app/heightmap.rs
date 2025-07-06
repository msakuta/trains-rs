mod noise_expr;
mod parser;

use std::collections::{HashMap, VecDeque};

use eframe::egui::{self, Color32, ColorImage, Painter, Pos2, Ui, pos2};
use serde::{Deserialize, Serialize};

use crate::{
    bg_image::BgImage,
    marching_squares::{
        Idx, Shape, border_pixel, cell_border_interpolated, pick_bits, pick_values,
    },
    perlin_noise::Xorshift64Star,
    structure::BELT_MAX_SLOPE,
    transform::PaintTransform,
    vec2::Vec2,
};

use super::{AREA_HEIGHT, AREA_WIDTH, ClickMode, TrainsApp};

use self::{
    noise_expr::{Value, precompute, run},
    parser::parse,
};

const DEFAULT_PERSISTENCE_OCTAVES: u32 = 3;

const DEFAULT_NOISE_OCTAVES: u32 = 4;

const DEFAULT_WATER_LEVEL: f32 = 0.4;

const BRIDGE_HEIGHT: f64 = 0.1;

pub(super) const CONTOURS_GRID_STEPE: usize = 8;

pub(super) const HEIGHTMAP_LEVELS: usize = 4;
pub(super) const HEIGHTMAP_LEVEL_SCALE: f64 = 2.;
const HEIGHTMAP_LEVEL_SCALE_U: usize = HEIGHTMAP_LEVEL_SCALE as usize;
pub(super) const TILE_SIZE: usize = 128;
const TILE_SIZE_I: isize = TILE_SIZE as isize;
const TILE_SHAPE: Shape = (TILE_SIZE_I, TILE_SIZE_I);

const TILE_WITH_MARGIN_SIZE: usize = TILE_SIZE + 1;
const TILE_WITH_MARGIN_SIZE_I: isize = TILE_WITH_MARGIN_SIZE as isize;
const TILE_WITH_MARGIN_SHAPE: Shape = (TILE_WITH_MARGIN_SIZE_I, TILE_WITH_MARGIN_SIZE_I);

#[derive(Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum NoiseType {
    White,
    Perlin,
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct HeightMapParams {
    pub noise_type: NoiseType,
    pub limited_size: bool,
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
            limited_size: false,
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

odist2 = length2(x) * 0.0075 * 0.0075;
center_plateau = 0.2 / (1 + odist2);

land_height = softmax(
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
);

softmax(center_plateau, land_height) * height_scale",
                DEFAULT_NOISE_OCTAVES, DEFAULT_PERSISTENCE_OCTAVES, BRIDGE_HEIGHT
            ),
        }
    }

    pub(super) fn params_ui(&mut self, ui: &mut Ui) {
        ui.group(|ui| {
            ui.radio_value(&mut self.noise_type, NoiseType::Perlin, "Perlin");
            ui.radio_value(&mut self.noise_type, NoiseType::White, "White");
        });
        ui.checkbox(&mut self.limited_size, "Limited world size");
        ui.group(|ui| {
            if !self.limited_size {
                ui.disable();
            }
            ui.horizontal(|ui| {
                ui.label("Width:");
                ui.add(egui::Slider::new(&mut self.width, 128..=4096).logarithmic(true));
            });
            ui.horizontal(|ui| {
                ui.label("Height:");
                ui.add(egui::Slider::new(&mut self.height, 128..=4096).logarithmic(true));
            });
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct HeightMapKey {
    pub level: usize,
    pub pos: [i32; 2],
}

#[derive(Clone, Debug)]
pub(crate) struct HeightMapTile {
    pub map: Vec<f32>,
    pub contours: ContoursCache,
}

pub(crate) struct HeightMap {
    pub(super) tiles: HashMap<HeightMapKey, HeightMapTile>,
    pub(super) shape: Shape,
    pub(super) water_level: f32,
    params: HeightMapParams,
    gen_queue: VecDeque<HeightMapKey>,
}

impl HeightMap {
    pub(super) fn new(params: &HeightMapParams) -> Result<Self, String> {
        Ok(Self {
            tiles: HashMap::new(),
            shape: (params.width as isize, params.height as isize),
            water_level: params.water_level,
            params: params.clone(),
            gen_queue: VecDeque::new(),
        })
    }

    /// Get or initialize a map at the given level.
    ///
    /// Heightmaps are lazily evaluated, so we may not have the data.
    pub fn get_map(&mut self, key: &HeightMapKey) -> Result<Option<&HeightMapTile>, String> {
        // We could not use entry API due to a lifetime issue.
        if !self.tiles.contains_key(key) {
            if !self.gen_queue.iter().any(|queued| queued == key) {
                self.gen_queue.push_back(*key);
            }
        }
        Ok(self.tiles.get(&key))
    }

    fn new_map(params: &HeightMapParams, key: &HeightMapKey) -> Result<Vec<f32>, String> {
        let mut rng = Xorshift64Star::new(params.seed);

        let mut ast = parse(&params.noise_expr)?;
        precompute(&mut ast, &mut rng)?;

        let map: Vec<_> = (0..TILE_WITH_MARGIN_SIZE.pow(2))
            .map(|i| {
                let ix = ((i % TILE_WITH_MARGIN_SIZE) as i32
                    + key.pos[0] * TILE_WITH_MARGIN_SIZE as i32) as f64
                    * HEIGHTMAP_LEVEL_SCALE.powi(key.level as i32);
                let iy = ((i / TILE_WITH_MARGIN_SIZE) as i32
                    + key.pos[1] * TILE_WITH_MARGIN_SIZE as i32) as f64
                    * HEIGHTMAP_LEVEL_SCALE.powi(key.level as i32);
                let pos = crate::vec2::Vec2::new(ix as f64, iy as f64);
                let Value::Scalar(eval_res) = run(&ast, &pos)? else {
                    return Err("Eval result was not a scalar".to_string());
                };
                let val = eval_res;
                Ok(val as f32)
            })
            .collect::<Result<_, _>>()?;

        // Heightmap cannot be normalized by values anymore since the noise can be sampled infinitely
        // let min_p = map
        //     .iter()
        //     .fold(None, |acc, cur| {
        //         if let Some(acc) = acc {
        //             if acc < *cur { Some(acc) } else { Some(*cur) }
        //         } else {
        //             Some(*cur)
        //         }
        //     })
        //     .ok_or_else(|| "Min value not found".to_string())?;
        // let max_p = map
        //     .iter()
        //     .fold(None, |acc, cur| {
        //         if let Some(acc) = acc {
        //             if acc < *cur { Some(*cur) } else { Some(acc) }
        //         } else {
        //             Some(*cur)
        //         }
        //     })
        //     .ok_or_else(|| "Max value not found".to_string())?;

        Ok(map)
    }

    /// Generate ore veins on a specified tile. Since the map is procedurally generated and has infinite size, we need
    /// to delay the ore generation until the tile is generated.
    pub(super) fn gen_ore_veins(&self, tile_pos: [i32; 2], tile: &HeightMapTile) -> Vec<Vec2> {
        let mut rng = Xorshift64Star::new(
            (tile_pos[0] as u64)
                .wrapping_add((tile_pos[1] as u64).wrapping_mul(209123))
                .wrapping_mul(40925612),
        );
        let num_veins = rng.nexti() % 5;
        (0..num_veins)
            .filter_map(|_| {
                let x = rng.next() * TILE_SIZE as f64;
                let y = rng.next() * TILE_SIZE as f64;
                let pos = Vec2::new(x, y);
                let ix = x as isize;
                let iy = y as isize;
                let is_water = tile.map[TILE_SHAPE.idx(ix, iy)] < self.water_level;

                if is_water {
                    None
                } else {
                    Some(Vec2::new(
                        pos.x + tile_pos[0] as f64 * TILE_SIZE as f64,
                        pos.y + tile_pos[1] as f64 * TILE_SIZE as f64,
                    ))
                }
            })
            .collect()
    }

    pub fn get_image(
        &mut self,
        key: &HeightMapKey,
        slope_threshold: Option<f64>,
    ) -> Result<ColorImage, String> {
        let water_level = self.water_level;

        let divisor = HEIGHTMAP_LEVEL_SCALE_U.pow(key.level as u32);

        // If the global map size is bounded, tiles on the edge can have size smaller than TILE_SIZE.
        let local_shape = if self.params.limited_size {
            (
                (self.params.width / divisor)
                    .saturating_sub(key.pos[0].max(0) as usize * TILE_SIZE)
                    .min(TILE_SIZE),
                (self.params.height / divisor)
                    .saturating_sub(key.pos[1].max(0) as usize * TILE_SIZE)
                    .min(TILE_SIZE),
            )
        } else {
            (TILE_SIZE, TILE_SIZE)
        };

        let map = &self.get_map(key)?.ok_or_else(|| "Tile not ready")?.map;
        let min_p = map
            .iter()
            .fold(None, |acc, cur| {
                if let Some(acc) = acc {
                    if acc < *cur { Some(acc) } else { Some(*cur) }
                } else {
                    Some(*cur)
                }
            })
            .ok_or_else(|| "Map has no pixel".to_string())?;
        let max_p = map
            .iter()
            .fold(None, |acc, cur| {
                if let Some(acc) = acc {
                    if acc < *cur { Some(*cur) } else { Some(acc) }
                } else {
                    Some(*cur)
                }
            })
            .ok_or_else(|| "Map has no pixel".to_string())?;
        let bitmap: Vec<_> = map
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let is_water = *p < water_level;
                let x = (i % TILE_WITH_MARGIN_SIZE) as f64;
                let y = (i / TILE_WITH_MARGIN_SIZE) as f64;
                if local_shape.0 as f64 <= x || local_shape.1 as f64 <= y {
                    // Outside of world
                    [255, 255, 255]
                } else if is_water {
                    let water_depth = (water_level - p) / (water_level - min_p);
                    let inten = 1. / (1. + 0.5 * water_depth);
                    [0, (95. * inten) as u8, (191. * inten) as u8]
                } else {
                    let above_water = (p - water_level) / (max_p - water_level);
                    let white = above_water.powi(4) as f64;
                    let grad = Self::local_gradient(map, TILE_WITH_MARGIN_SHAPE, &Vec2::new(x, y))
                        / HEIGHTMAP_LEVEL_SCALE.powi(key.level as i32);
                    let slope = grad.length();
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
                    if slope_threshold.is_some_and(|threshold| threshold < slope) {
                        [red / 2 + 127, green / 2, blue / 2]
                    } else {
                        [red, green, blue]
                    }
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
            [TILE_WITH_MARGIN_SIZE, TILE_WITH_MARGIN_SIZE],
            &bitmap,
        );
        Ok(img)
    }

    pub(crate) fn gradient(&self, pos: &crate::vec2::Vec2<f64>) -> crate::vec2::Vec2<f64> {
        self.tiles
            .get(&Self::key_from_pos(pos))
            .map_or(Vec2::zero(), |tile| {
                Self::local_gradient(&tile.map, TILE_SHAPE, pos)
            })
    }

    fn local_gradient(map: &[f32], shape: Shape, pos: &crate::vec2::Vec2) -> crate::vec2::Vec2 {
        let [x, y] = [pos.x as isize, pos.y as isize];
        if x < 0 || shape.0 <= x + 1 || y < 0 || shape.1 <= y + 1 {
            return crate::vec2::Vec2::zero();
        }

        let dx = map[shape.idx(x + 1, y)] - map[shape.idx(x, y)];
        let dy = map[shape.idx(x, y + 1)] - map[shape.idx(x, y)];

        crate::vec2::Vec2::new(dx as f64, dy as f64)

        // let [fx, fy] = [pos.x % 1., pos.y % 1.];

        // let lerp = |a, b, f| (1. - f) * a + f * b;
        // lerp(lerp(heightmap[shape.idx(x, y)], heightmap[shape.idx(x + 1, y)], fx),
        // lerp(heightmap[shape.idx(x, y + 1)], heightmap[shape.idx(x + 1, y + 1)], fx), fy)
    }

    pub(crate) fn is_water(&self, pos: &crate::vec2::Vec2<f64>) -> bool {
        let [x, y] = [pos.x as isize, pos.y as isize];
        if self.params.limited_size && (x < 0 || self.shape.0 <= x || y < 0 || self.shape.1 < y) {
            return true;
        }
        let key = Self::key_from_pos(pos);
        let ix = x.rem_euclid(TILE_SIZE_I);
        let iy = y.rem_euclid(TILE_SIZE_I);
        self.tiles.get(&key).map_or(true, |tile| {
            tile.map[TILE_SHAPE.idx(ix, iy)] < self.water_level
        })
    }

    pub(super) fn key_from_pos(pos: &crate::vec2::Vec2) -> HeightMapKey {
        HeightMapKey {
            level: 0,
            pos: [
                (pos.x as isize).div_euclid(TILE_SIZE_I) as i32,
                (pos.y as isize).div_euclid(TILE_SIZE_I) as i32,
            ],
        }
    }

    pub(super) fn update(&mut self, contours_grid_step: usize) -> Result<Vec<[i32; 2]>, String> {
        let mut ret = vec![];
        // We would like to offload the tile generation to another thread, but until we know if it works in wasm,
        // we use the main thread to do it, but progressively.
        if let Some(key) = self.gen_queue.pop_front() {
            let tile = HeightMapTile::new(Self::new_map(&self.params, &key)?, contours_grid_step);
            if key.level == 0 {
                ret.push(key.pos);
                // ret.extend_from_slice(&self.gen_structures(key.pos, &tile));
            }
            self.tiles.insert(key, tile);
        }
        Ok(ret)
    }
}

impl std::ops::Index<(isize, isize)> for HeightMap {
    type Output = f32;
    fn index(&self, index: (isize, isize)) -> &Self::Output {
        self.tiles
            .get(&HeightMap::key_from_pos(&crate::vec2::Vec2::new(
                index.0 as f64,
                index.1 as f64,
            )))
            .map_or(&0., |tile| &tile.map[TILE_SHAPE.idx(index.0, index.1)])
    }
}

// impl std::ops::IndexMut<(isize, isize)> for HeightMap {
//     fn index_mut(&mut self, index: (isize, isize)) -> &mut Self::Output {
//         self.map[self.shape.idx(index.0, index.1)]
//     }
// }

#[derive(Clone, Debug)]
pub struct ContoursCache {
    // grid_step: usize,
    contours: HashMap<i32, Vec<[Pos2; 2]>>,
}

impl ContoursCache {
    fn new() -> Self {
        Self {
            // grid_step,
            contours: HashMap::new(),
        }
    }
}

impl TrainsApp {
    // Contours are always rendered with a cache now.
    // pub fn render_contours(
    //     &self,
    //     painter: &Painter,
    //     to_pos2: &impl Fn(Pos2) -> Pos2,
    //     contour_grid_step: usize,
    // ) {
    //     if self.show_grid {
    //         render_grid(painter, &self.heightmap.shape, to_pos2, contour_grid_step);
    //     }

    //     if !self.show_contours {
    //         return;
    //     }

    //     self.heightmap
    //         .process_contours(contour_grid_step, |level, points| {
    //             let points = [to_pos2(points[0]), to_pos2(points[1])];

    //             let line_width = if level % 4 == 0 { 1.5 } else { 1. };
    //             let r = 127 + (level * 32).wrapping_rem_euclid(128);

    //             painter.line_segment(points, (line_width, Color32::from_rgb(r as u8, 0, 0)));
    //         });
    // }

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

        let level = (-(self.transform.scale().log2() / HEIGHTMAP_LEVEL_SCALE.log2() as f32).floor()
            + 1.)
            .clamp(0., HEIGHTMAP_LEVELS as f32) as usize;

        for (key, tile) in &self.heightmap.tiles {
            if key.level != level {
                continue;
            }
            let scale = HEIGHTMAP_LEVEL_SCALE.powi(key.level as i32) as f32;
            let offset =
                egui::vec2(key.pos[0] as f32, key.pos[1] as f32) * TILE_SIZE as f32 * scale;
            let cache = &tile.contours;
            HeightMapTile::render_with_cache(painter, cache, &|pos| to_pos2(pos * scale + offset));
        }
    }

    pub(super) fn render_heightmaps(
        &mut self,
        painter: &Painter,
        paint_transform: &PaintTransform,
    ) {
        let level = (-(self.transform.scale().log2() / HEIGHTMAP_LEVEL_SCALE.log2() as f32).floor()
            + 1.)
            .clamp(0., HEIGHTMAP_LEVELS as f32) as usize;
        let heightmap_scale = (HEIGHTMAP_LEVEL_SCALE as f32).powi(level as i32);
        let lt = paint_transform.from_pos2(painter.clip_rect().left_top());
        let rb = paint_transform.from_pos2(painter.clip_rect().right_bottom());
        let divisor = TILE_SIZE as f64 * heightmap_scale as f64;
        // top is greater than bottom in screen coordinates, which is flipped from Cartesian
        let (x0, y0, x1, y1);
        if self.heightmap.params.limited_size {
            x0 = lt.x.max(0.);
            y0 = rb.y.max(0.);
            x1 = rb.x.min(self.heightmap.params.width as f64);
            y1 = lt.y.min(self.heightmap.params.height as f64);
        } else {
            x0 = lt.x;
            y0 = rb.y;
            x1 = rb.x;
            y1 = lt.y;
        }
        let x0 = x0.div_euclid(divisor) as i32;
        let y0 = y0.div_euclid(divisor) as i32;
        let x1 = x1.div_euclid(divisor) as i32;
        let y1 = y1.div_euclid(divisor) as i32;
        for y in y0..=y1 {
            for x in x0..=x1 {
                let _ = self
                    .bg
                    .entry(HeightMapKey { level, pos: [x, y] })
                    .or_insert_with(BgImage::new)
                    .paint(
                        &painter,
                        (),
                        |_| {
                            self.heightmap.get_image(
                                &HeightMapKey { level, pos: [x, y] },
                                if matches!(self.click_mode, ClickMode::ConnectBelt) {
                                    Some(BELT_MAX_SLOPE)
                                } else {
                                    None
                                },
                            )
                        },
                        &paint_transform,
                        heightmap_scale,
                        egui::pos2(
                            x as f32 * heightmap_scale * TILE_SIZE as f32,
                            (y + 1) as f32 * heightmap_scale * TILE_SIZE as f32,
                        ),
                    );
            }
        }
    }
}

impl HeightMapTile {
    fn new(map: Vec<f32>, contour_grid_step: usize) -> Self {
        let contours = Self::cache_contours(&map, contour_grid_step);
        Self { map, contours }
    }

    pub fn cache_contours(map: &[f32], contour_grid_step: usize) -> ContoursCache {
        let mut ret = ContoursCache::new();
        Self::process_contours(map, contour_grid_step, |level, points| {
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

    fn process_contours(map: &[f32], contour_grid_step: usize, mut f: impl FnMut(i32, &[Pos2; 2])) {
        let contours_grid_size = TILE_SIZE / contour_grid_step + 1;
        let downsampled: Vec<_> = (0..contours_grid_size.pow(2))
            .map(|i| {
                let x = (i % contours_grid_size) * contour_grid_step;
                let y = (i / contours_grid_size) * contour_grid_step;
                map[x + y * TILE_WITH_MARGIN_SIZE]
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
            (TILE_SIZE / contour_grid_step) as isize + 1,
            (TILE_SIZE / contour_grid_step) as isize + 1,
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
