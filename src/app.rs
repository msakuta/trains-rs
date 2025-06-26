mod heightmap;
mod train;

use std::collections::HashMap;

use eframe::{
    egui::{self, Align2, Color32, FontId, Frame, Painter, Rect, Ui, vec2},
    epaint::PathShape,
};
use heightmap::DOWNSAMPLE;

pub(crate) use self::heightmap::HeightMap;
use self::heightmap::{ContoursCache, HeightMapParams};

use crate::{
    bg_image::BgImage,
    perlin_noise::Xorshift64Star,
    structure::{BeltConnection, Belts, MAX_BELT_LENGTH, Structure},
    train::Train,
    train_tracks::{SelectedPathNode, Station, StationType, TrainTracks},
    transform::{PaintTransform, Transform, half_rect},
    vec2::Vec2,
};

pub(crate) const AREA_WIDTH: usize = 512;
pub(crate) const AREA_HEIGHT: usize = 512;
// const AREA_SHAPE: Shape = (AREA_WIDTH as isize, AREA_HEIGHT as isize);
const SELECT_PIXEL_RADIUS: f64 = 20.;
const MAX_NUM_CARS: usize = 10;
const MAX_CONTOURS_GRID_STEP: usize = 100;
const STRUCTURE_SIZE: f32 = 10.;
const TRAIN_JSON: &str = "train.json";
const TRAIN_KEY: &str = "train";
const TRACKS_KEY: &str = "train_tracks";
const HEIGHTMAP_KEY: &str = "heightmap";
const BELTS_KEY: &str = "belts";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClickMode {
    None,
    GentleCurve,
    TightCurve,
    StraightLine,
    BezierCurve,
    DeleteSegment,
    AddStation,
    ConnectBelt,
}

pub(crate) struct TrainsApp {
    transform: Transform,
    heightmap: HeightMap,
    heightmap_params: HeightMapParams,
    contours_cache: Option<ContoursCache>,
    contour_grid_step: usize,
    bg: BgImage,
    show_contours: bool,
    show_grid: bool,
    use_cached_contours: bool,
    show_debug_slope: bool,
    focus_on_train: bool,
    click_mode: ClickMode,
    belt_connection: Option<(BeltConnection, Vec2<f64>)>,
    tracks: TrainTracks,
    train: Train,
    selected_station: Option<usize>,
    new_station: String,
    station_type: StationType,
    structures: HashMap<usize, Structure>,
    belts: Belts,
    credits: u32,
    error_msg: Option<(String, f64)>,
}

impl TrainsApp {
    pub fn new() -> Self {
        let contour_grid_step = DOWNSAMPLE;

        let (train, tracks, heightmap_params, belts) = std::fs::File::open(TRAIN_JSON)
            .and_then(|train_json| {
                let mut value: serde_json::Value =
                    serde_json::from_reader(std::io::BufReader::new(train_json))
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
                let train_value = value
                    .get_mut(TRAIN_KEY)
                    .ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::Other, "train doesn't exist")
                    })?
                    .take();
                let train: Train = serde_json::from_value(train_value)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

                let tracks_value = value
                    .get_mut(TRACKS_KEY)
                    .ok_or_else(|| {
                        std::io::Error::new(std::io::ErrorKind::Other, "train_track doesn't exist")
                    })?
                    .take();

                let tracks: TrainTracks = serde_json::from_value(tracks_value)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

                let heightmap_value = value
                    .get_mut(HEIGHTMAP_KEY)
                    .ok_or_else(|| {
                        std::io::Error::new(
                            std::io::ErrorKind::Other,
                            format!("{HEIGHTMAP_KEY}  doesn't exist"),
                        )
                    })?
                    .take();

                let heightmap: HeightMapParams = serde_json::from_value(heightmap_value)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;

                let belts: Belts = serde_json::from_value(
                    value
                        .get_mut(BELTS_KEY)
                        .ok_or_else(|| {
                            std::io::Error::new(std::io::ErrorKind::Other, "belts doesn't exist")
                        })?
                        .take(),
                )?;

                Ok((train, tracks, heightmap, belts))
            })
            .unwrap_or_else(|e| {
                eprintln!("Failed to load train data, falling back to default: {e}");
                (
                    Train::new(),
                    TrainTracks::new(),
                    HeightMapParams::new(),
                    Belts::new(),
                )
            });

        // Fall back to the default params if the heightmap failed to generate with the params
        let heightmap = HeightMap::new(&heightmap_params)
            .or_else(|_| HeightMap::new(&HeightMapParams::new()))
            .unwrap();

        let mut rng = Xorshift64Star::new(40925612);
        let structures = (0..10)
            .filter_map(|i| {
                let x = rng.next() * heightmap_params.width as f64;
                let y = rng.next() * heightmap_params.height as f64;
                let pos = Vec2::new(x, y);
                if heightmap.is_water(&pos) {
                    None
                } else {
                    Some((i, Structure::new_ore_mine(pos)))
                }
            })
            .collect();

        let contours_cache = heightmap.cache_contours(contour_grid_step);

        Self {
            transform: Transform::new(1.),
            heightmap,
            heightmap_params,
            contours_cache: Some(contours_cache),
            contour_grid_step,
            bg: BgImage::new(),
            show_contours: true,
            show_grid: false,
            use_cached_contours: true,
            show_debug_slope: false,
            focus_on_train: true,
            click_mode: ClickMode::None,
            belt_connection: None,
            tracks,
            train,
            selected_station: None,
            new_station: "New Station".to_string(),
            station_type: StationType::Loading,
            structures,
            belts,
            credits: 0,
            error_msg: None,
        }
    }

    fn process_result(&mut self, pos: Vec2<f64>, res: Result<(), String>) {
        if let Err(e) = res {
            self.error_msg = Some((e, 10.));
        } else {
            println!("Added point {pos:?}");
        }
    }

    fn find_belt_con(&self, pos: Vec2<f64>) -> (BeltConnection, Vec2<f64>) {
        const SELECT_THRESHOLD: f64 = 10.;
        let scale = 1. / self.transform.scale();
        for (i, structure) in &self.structures {
            let dist2 = (structure.pos - pos).length2();
            if dist2 / (scale.powi(2) as f64) < SELECT_THRESHOLD.powi(2) {
                return (BeltConnection::Structure(*i), structure.pos);
            }
        }
        (BeltConnection::Pos, pos)
    }

    fn render(&mut self, ui: &mut Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::click());

        if self.focus_on_train {
            if let Some(offset) = self.tracks.s_pos(self.train.path_id(), self.train.s()) {
                self.transform
                    .set_offset([-offset.x as f32, -offset.y as f32]);
            }
        }

        if ui.ui_contains_pointer() {
            ui.input(|i| self.transform.handle_mouse(i, half_rect(&response.rect)));
        }

        let paint_transform = self.transform.into_paint(&response);

        let thresh = SELECT_PIXEL_RADIUS / self.transform.scale() as f64;

        if response.clicked() {
            if let Some(pointer) = response.interact_pointer_pos() {
                match self.click_mode {
                    ClickMode::None => {
                        let pos = paint_transform.from_pos2(pointer);
                        let _res = self.tracks.select_node(pos, thresh);
                    }
                    ClickMode::GentleCurve => {
                        let pos = paint_transform.from_pos2(pointer);
                        if self.tracks.has_selected_node() {
                            // self.train.control_points.push(pos);
                            let res = self
                                .tracks
                                .add_gentle(pos, &self.heightmap, &mut self.train);
                            self.process_result(pos, res);
                        } else {
                            let _res = self.tracks.select_node(pos, thresh);
                        }
                    }
                    ClickMode::TightCurve => {
                        let pos = paint_transform.from_pos2(pointer);
                        if self.tracks.has_selected_node() {
                            let res = self.tracks.add_tight(pos, &self.heightmap, &mut self.train);
                            self.process_result(pos, res);
                        } else {
                            let _res = self.tracks.select_node(pos, thresh);
                        }
                    }
                    ClickMode::StraightLine => {
                        let pos = paint_transform.from_pos2(pointer);
                        if self.tracks.has_selected_node() {
                            let res =
                                self.tracks
                                    .add_straight(pos, &self.heightmap, &mut self.train);
                            self.process_result(pos, res);
                        } else {
                            let _res = self.tracks.select_node(pos, thresh);
                        }
                    }
                    ClickMode::BezierCurve => {
                        let pos = paint_transform.from_pos2(pointer);
                        if self.tracks.has_selected_node() {
                            let res = self
                                .tracks
                                .add_bezier(pos, &self.heightmap, &mut self.train);
                            self.process_result(pos, res);
                        } else {
                            let _res = self.tracks.select_node(pos, thresh);
                        }
                    }
                    ClickMode::DeleteSegment => {
                        let pos = paint_transform.from_pos2(pointer);
                        let res = self.tracks.delete_segment(pos, thresh, &mut self.train);
                        self.process_result(pos, res);
                    }
                    ClickMode::AddStation => {
                        let pos = paint_transform.from_pos2(pointer);
                        let next_name = (0..).find_map(|i| {
                            let cand = format!("New Station {i}");
                            if !self.tracks.stations.values().any(|s| s.name == cand)
                                && self.new_station != cand
                            {
                                Some(cand)
                            } else {
                                None
                            }
                        });
                        if let Some(name) = next_name {
                            self.tracks.add_station(
                                std::mem::replace(&mut self.new_station, name),
                                pos,
                                thresh,
                                self.station_type,
                            );
                        }
                    }
                    ClickMode::ConnectBelt => {
                        let pos = paint_transform.from_pos2(pointer);
                        if let Some((start_con, start_pos)) = &self.belt_connection {
                            let (end_con, end_pos) = self.find_belt_con(pos);
                            // Disallow connection to itself
                            if end_con != *start_con
                                && (end_pos - *start_pos).length2() < MAX_BELT_LENGTH.powi(2)
                            {
                                self.belts
                                    .add_belt(*start_pos, *start_con, end_pos, end_con);
                                self.belt_connection = None;
                            }
                        } else {
                            self.belt_connection = Some(self.find_belt_con(pos));
                        }
                    }
                }
            }
        }

        let _ = self.bg.paint(
            &painter,
            (),
            |_| self.heightmap.get_image(),
            &paint_transform,
        );

        if self.use_cached_contours {
            self.render_contours_with_cache(
                &painter,
                &|p| paint_transform.transform_pos2(p),
                self.contour_grid_step,
            );
        } else {
            self.render_contours(
                &painter,
                &|p| paint_transform.transform_pos2(p),
                self.contour_grid_step,
            );
        }

        self.render_track(&painter, &paint_transform);

        for (_, structure) in &self.structures {
            let pos = paint_transform.to_pos2(structure.pos);
            painter.rect_filled(
                Rect::from_center_size(pos, vec2(STRUCTURE_SIZE, STRUCTURE_SIZE)),
                0.,
                Color32::from_rgb(0, 127, 191),
            );
        }

        match self.click_mode {
            ClickMode::None => self.tracks.ghost_path = None,
            ClickMode::GentleCurve => {
                if self.tracks.has_selected_node() {
                    if let Some(pos) = response.hover_pos() {
                        self.tracks.ghost_gentle(paint_transform.from_pos2(pos));
                    } else {
                        self.tracks.ghost_path = None;
                    }
                }
            }
            ClickMode::StraightLine => {
                if self.tracks.has_selected_node() {
                    if let Some(pos) = response.hover_pos() {
                        self.tracks.ghost_straight(paint_transform.from_pos2(pos));
                    } else {
                        self.tracks.ghost_path = None;
                    }
                }
            }
            ClickMode::TightCurve => {
                if self.tracks.has_selected_node() {
                    if let Some(pos) = response.hover_pos() {
                        self.tracks.ghost_tight(paint_transform.from_pos2(pos));
                    } else {
                        self.tracks.ghost_path = None;
                    }
                }
            }
            ClickMode::BezierCurve => {
                if self.tracks.has_selected_node() {
                    if let Some(pos) = response.hover_pos() {
                        self.tracks.ghost_bezier(paint_transform.from_pos2(pos));
                    } else {
                        self.tracks.ghost_path = None;
                    }
                }
            }
            ClickMode::DeleteSegment => {
                let found_node = response.hover_pos().and_then(|pointer| {
                    let thresh = SELECT_PIXEL_RADIUS / self.transform.scale() as f64;
                    self.tracks
                        .find_path_node(paint_transform.from_pos2(pointer), thresh)
                });
                if let Some((path_id, seg_id, _)) = found_node {
                    let color = Color32::from_rgba_premultiplied(127, 0, 127, 63);
                    if let Some(path) = self.tracks.paths.get(&path_id) {
                        let seg_track = path.seg_track(seg_id);
                        self.render_track_detail(seg_track, &painter, &paint_transform, 5., color);
                    }
                }
            }
            ClickMode::AddStation => {
                if let Some(pointer) = response.hover_pos() {
                    let pos = paint_transform.from_pos2(pointer);
                    if let Some((path_id, _, node_id)) = self.tracks.find_path_node(pos, thresh) {
                        let station = Station::new(
                            self.new_station.clone(),
                            path_id,
                            node_id as f64,
                            self.station_type,
                        );
                        self.render_station(&painter, &paint_transform, &station, false, true);
                    }
                }
            }
            ClickMode::ConnectBelt => {
                if let Some(pointer) = response.hover_pos() {
                    let pos = paint_transform.from_pos2(pointer);
                    let (end_con, end_pos) = self.find_belt_con(pos);
                    if let Some((start_con, start_pos)) = &self.belt_connection {
                        if matches!(end_con, BeltConnection::Structure(_)) && end_con != *start_con
                        {
                            painter.rect_filled(
                                Rect::from_center_size(
                                    paint_transform.to_pos2(end_pos),
                                    vec2(STRUCTURE_SIZE, STRUCTURE_SIZE),
                                ),
                                0.,
                                Color32::from_rgb(255, 127, 191),
                            );
                        }
                        let color = if (end_pos - *start_pos).length2() < MAX_BELT_LENGTH.powi(2) {
                            Color32::from_rgb(255, 0, 255)
                        } else {
                            Color32::RED
                        };
                        painter.line_segment(
                            [
                                paint_transform.to_pos2(end_pos),
                                paint_transform.to_pos2(*start_pos),
                            ],
                            (2., color),
                        );
                    } else if matches!(end_con, BeltConnection::Structure(_)) {
                        painter.rect_filled(
                            Rect::from_center_size(
                                paint_transform.to_pos2(end_pos),
                                vec2(STRUCTURE_SIZE, STRUCTURE_SIZE),
                            ),
                            0.,
                            Color32::from_rgb(255, 127, 191),
                        );
                    }
                }
            }
        }

        if let Some(pointer) = response.hover_pos() {
            let pos = paint_transform.from_pos2(pointer);
            if let Some((id, node_pos)) = self
                .tracks
                .find_segment_node(pos, thresh)
                .and_then(|id| Some((id, self.tracks.node_position(id)?.0)))
            {
                painter.rect_stroke(
                    Rect::from_center_size(
                        paint_transform.to_pos2(node_pos),
                        egui::Vec2::splat(10.),
                    ),
                    0.,
                    (1., Color32::from_rgba_premultiplied(255, 0, 255, 127)),
                    egui::StrokeKind::Middle,
                );

                self.render_path_direction(&painter, &paint_transform, &id);
            }
        }

        if let Some(sel_node) = self.tracks.selected_node() {
            painter.rect_stroke(
                Rect::from_center_size(paint_transform.to_pos2(sel_node.0), egui::Vec2::splat(10.)),
                0.,
                (2., Color32::from_rgb(255, 0, 255)),
                egui::StrokeKind::Middle,
            );
        }

        for belt in self.belts.belts.values() {
            let start = paint_transform.to_pos2(belt.start);
            let end = paint_transform.to_pos2(belt.end);
            painter.arrow(start, end - start, (2., Color32::BLUE));
        }

        for station in self.tracks.stations.values() {
            // let i_ptr = &*station.borrow() as *const _;
            // let is_target = if let TrainTask::Goto(target) = &self.train.train_task {
            //     if let Some(rc) = target.upgrade() {
            //         let target_ref = rc.borrow();
            //         let target_ptr = &*target_ref as *const _;
            //         i_ptr == target_ptr
            //     } else {
            //         false
            //     }
            // } else {
            //     false
            // };
            let is_target = false;
            self.render_station(&painter, &paint_transform, &station, is_target, false);
        }

        self.render_train(&painter, &paint_transform);

        if let Some((ref err, _)) = self.error_msg {
            painter.text(
                response.rect.center(),
                Align2::CENTER_CENTER,
                err,
                FontId::default(),
                Color32::RED,
            );
        }
    }

    fn render_station(
        &self,
        painter: &Painter,
        paint_transform: &PaintTransform,
        station: &Station,
        is_target: bool,
        is_ghost: bool,
    ) {
        const STATION_HEIGHT: f64 = 5.;

        let Some(pos) = self.tracks.s_pos(station.path_id, station.s) else {
            return;
        };
        let alpha = if is_ghost { 63 } else { 255 };
        painter.line_segment(
            [
                paint_transform.to_pos2(pos),
                paint_transform.to_pos2(pos + Vec2::new(0., STATION_HEIGHT)),
            ],
            (3., Color32::from_rgba_premultiplied(0, 127, 63, alpha)),
        );

        painter.add(PathShape::convex_polygon(
            [[0., 0.], [2., -1.], [0., -2.]]
                .into_iter()
                .map(|ofs| {
                    paint_transform.to_pos2(pos + Vec2::new(ofs[0], STATION_HEIGHT + ofs[1]))
                })
                .collect(),
            Color32::from_rgba_premultiplied(63, 95, 0, alpha),
            (1., Color32::from_rgba_premultiplied(31, 63, 0, alpha)),
        ));

        painter.text(
            paint_transform.to_pos2(pos),
            Align2::CENTER_BOTTOM,
            &station.name,
            FontId::proportional(16.),
            if is_target {
                Color32::RED
            } else if is_ghost {
                Color32::from_rgba_premultiplied(0, 0, 255, 255)
            } else {
                Color32::BLACK
            },
        );
    }

    fn render_path_direction(
        &self,
        painter: &Painter,
        paint_transform: &PaintTransform,
        selected_segment: &SelectedPathNode,
    ) {
        let Some((pos, angle)) = self.tracks.node_position(*selected_segment) else {
            return;
        };

        painter.add(PathShape::convex_polygon(
            [[0., 0.], [-10., -6.], [-10., 6.]]
                .into_iter()
                .map(|ofs| {
                    let x = ofs[0] * angle.cos() + ofs[1] * angle.sin();
                    let y = -ofs[0] * angle.sin() + ofs[1] * angle.cos();
                    paint_transform.to_pos2(pos) + egui::vec2(x as f32, y as f32)
                })
                .collect(),
            Color32::from_rgba_premultiplied(255, 255, 0, 255),
            (2., Color32::from_rgba_premultiplied(127, 127, 0, 255)),
        ));
    }

    fn ui_panel(&mut self, ui: &mut Ui) {
        ui.checkbox(&mut self.show_contours, "Show contour lines");
        ui.horizontal(|ui| {
            ui.label("Contours grid step:");
            ui.add(egui::Slider::new(
                &mut self.contour_grid_step,
                1..=MAX_CONTOURS_GRID_STEP,
            ));
        });
        if self
            .contours_cache
            .as_ref()
            .is_some_and(|c| c.grid_step() != self.contour_grid_step)
        {
            self.contours_cache = Some(self.heightmap.cache_contours(self.contour_grid_step));
        }
        ui.checkbox(&mut self.show_grid, "Show grid");
        ui.checkbox(&mut self.use_cached_contours, "Use cached contours");
        ui.checkbox(&mut self.show_debug_slope, "Show debug slope");
        ui.checkbox(&mut self.focus_on_train, "Focus on train");
        ui.horizontal(|ui| {
            ui.label("Num cars:");
            ui.add(egui::Slider::new(
                &mut self.train.cars.len(),
                1..=MAX_NUM_CARS,
            ));
        });
        ui.group(|ui| {
            ui.label("Terrain generation params");
            self.heightmap_params.params_ui(ui);
            if ui.button("Regenerate").clicked() {
                match HeightMap::new(&self.heightmap_params) {
                    Ok(map) => {
                        self.heightmap = map;
                        self.contours_cache =
                            Some(self.heightmap.cache_contours(self.contour_grid_step));
                        self.bg.clear();
                    }
                    Err(e) => {
                        self.error_msg = Some((e.to_string(), 10.));
                    }
                }
            }
        });
        ui.group(|ui| {
            ui.label("Click mode:");
            ui.radio_value(&mut self.click_mode, ClickMode::None, "None");
            ui.radio_value(&mut self.click_mode, ClickMode::GentleCurve, "Gentle Curve");
            ui.radio_value(&mut self.click_mode, ClickMode::TightCurve, "Tight Curve");
            ui.radio_value(
                &mut self.click_mode,
                ClickMode::StraightLine,
                "Straight Line",
            );
            ui.radio_value(&mut self.click_mode, ClickMode::BezierCurve, "Bezier Curve");
            ui.radio_value(&mut self.click_mode, ClickMode::DeleteSegment, "Delete");
            ui.radio_value(&mut self.click_mode, ClickMode::AddStation, "Add Station");
            ui.radio_value(&mut self.click_mode, ClickMode::ConnectBelt, "Connect Belt");
            if !matches!(self.click_mode, ClickMode::ConnectBelt) {
                self.belt_connection = None;
            }
        });
        ui.group(|ui| {
            ui.label("Stations:");
            for (i, station) in &self.tracks.stations {
                ui.radio_value(
                    &mut self.selected_station,
                    Some(*i),
                    &format!(
                        "{} ({}, {}) {:?}",
                        station.name, station.path_id, station.s, station.ty
                    ),
                );
            }
            if ui.button("Schedule station").clicked() {
                if let Some(target) = self.selected_station {
                    self.train.schedule.push(target);
                }
            }
            if ui.button("Delete station").clicked() {
                if let Some(target) = self.selected_station {
                    self.tracks.stations.remove(&target);
                }
            }
            ui.text_edit_singleline(&mut self.new_station);
            ui.group(|ui| {
                ui.label("Station type:");
                ui.radio_value(&mut self.station_type, StationType::Loading, "Loading");
                ui.radio_value(&mut self.station_type, StationType::Unloading, "Unloading");
            });
        });
        ui.group(|ui| {
            ui.label("Trains:");
            for (i, car) in self.train.cars.iter().enumerate() {
                ui.label(&format!(
                    "[{i}] {:.03}, {:.03}, {:.03}, {:?}",
                    car.path_id, car.s, car.speed, car.direction
                ));
            }
            ui.label(format!(
                "Total transported items: {}",
                self.train.total_transported
            ));
        });
    }
}

impl eframe::App for TrainsApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        ctx.request_repaint();

        // Decay error message even if paused
        if let Some((_, ref mut time)) = self.error_msg {
            let dt = ctx.input(|i| i.raw.predicted_dt);
            *time = *time - dt as f64;
            if *time < 0. {
                self.error_msg = None;
            }
        }

        eframe::egui::SidePanel::right("side_panel")
            .min_width(200.)
            .show(ctx, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| self.ui_panel(ui))
            });

        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            Frame::canvas(ui.style()).show(ui, |ui| {
                self.render(ui);
            });
        });

        let mut thrust = 0.;
        ctx.input(|input| {
            for key in input.keys_down.iter() {
                match key {
                    egui::Key::W => thrust += self.train.direction().signum(),
                    egui::Key::S => thrust -= self.train.direction().signum(),
                    _ => {}
                }
            }
            if self.train.can_switch() {
                if input.key_pressed(egui::Key::A) {
                    self.train.switch_path = self.train.switch_path.saturating_add(1)
                }
                if input.key_pressed(egui::Key::D) {
                    self.train.switch_path = self.train.switch_path.saturating_sub(1)
                }
            }
        });
        self.train.update(thrust, &self.heightmap, &self.tracks);
        for (_, structure) in &mut self.structures {
            structure.update(&mut self.credits);
        }
    }
}

impl std::ops::Drop for TrainsApp {
    fn drop(&mut self) {
        println!("TrainsApp dropped");
        let mut map = serde_json::Map::new();
        map.insert(
            TRAIN_KEY.to_string(),
            serde_json::to_value(&self.train).unwrap(),
        );
        map.insert(
            TRACKS_KEY.to_string(),
            serde_json::to_value(&self.tracks).unwrap(),
        );
        map.insert(
            HEIGHTMAP_KEY.to_string(),
            serde_json::to_value(&self.heightmap_params).unwrap(),
        );
        map.insert(
            BELTS_KEY.to_string(),
            serde_json::to_value(&self.belts).unwrap(),
        );
        let _ = serde_json::to_writer_pretty(
            std::io::BufWriter::new(std::fs::File::create(TRAIN_JSON).unwrap()),
            &map,
        );
    }
}
