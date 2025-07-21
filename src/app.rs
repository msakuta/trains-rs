mod heightmap;
mod structure;
mod train;

use std::collections::HashMap;

use eframe::{
    egui::{self, Align2, Color32, FontId, Frame, Painter, Rect, Ui},
    epaint::PathShape,
};
use egui_extras::install_image_loaders;

pub(crate) use self::heightmap::HeightMap;
use self::heightmap::{CONTOURS_GRID_STEPE, HeightMapKey, HeightMapParams};

use crate::{
    bg_image::BgImage,
    structure::{
        BeltConnection, OreType, OreVein, PipeConnection, PowerStats, Structure, StructureId,
        StructureType, Structures,
    },
    train::Train,
    train_tracks::{SelectedPathNode, Station, TrainTracks},
    transform::{PaintTransform, Transform, half_rect},
    vec2::Vec2,
};

pub(crate) const AREA_WIDTH: usize = 256;
pub(crate) const AREA_HEIGHT: usize = 256;
// const AREA_SHAPE: Shape = (AREA_WIDTH as isize, AREA_HEIGHT as isize);
const SELECT_PIXEL_RADIUS: f64 = 20.;
const MAX_NUM_CARS: usize = 10;
const MAX_CONTOURS_GRID_STEP: usize = 100;
const TRAIN_JSON: &str = "train.json";
const TRAIN_KEY: &str = "train";
const TRACKS_KEY: &str = "train_tracks";
const HEIGHTMAP_KEY: &str = "heightmap";
const STRUCTURES_KEY: &str = "structures";
const CREDITS_KEY: &str = "credits";
const ORE_URL: &str = "bytes://ore.png";
const INGOT_URL: &str = "bytes://metal.png";
const COAL_URL: &str = "bytes://coal.png";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClickMode {
    None,
    GentleCurve,
    TightCurve,
    StraightLine,
    BezierCurve,
    DeleteSegment,
    AddStation,
    AddOreMine,
    AddSmelter,
    AddLoader,
    AddUnloader,
    AddSplitter,
    AddMerger,
    AddWaterPump,
    AddBoiler,
    AddSteamEngine,
    ConnectBelt,
    ConnectPipe,
    ConnectWire,
    DeleteStructure,
}

pub(crate) struct TrainsApp {
    transform: Transform,
    heightmap: HeightMap,
    heightmap_params: HeightMapParams,
    contour_grid_step: usize,
    bg: HashMap<HeightMapKey, BgImage>,
    show_contours: bool,
    show_grid: bool,
    show_debug_slope: bool,
    focus_on_train: bool,
    click_mode: ClickMode,
    cursor: Option<Vec2>,
    belt_connection: Option<(BeltConnection, Vec2<f64>)>,
    pipe_connection: Option<(PipeConnection, Vec2<f64>)>,
    building_structure: Option<Vec2>,
    wire_start: Option<(Vec2, StructureId)>,
    tracks: TrainTracks,
    train: Train,
    selected_station: Option<usize>,
    new_station: String,
    ore_veins: Vec<OreVein>,
    structures: Structures,
    power_stats: PowerStats,
    credits: u32,
    time: u32,
    error_msg: Option<(String, f64)>,
}

impl TrainsApp {
    pub fn new() -> Self {
        let contour_grid_step = CONTOURS_GRID_STEPE;

        let (train, tracks, heightmap_params, heightmap, structures, credits) = Self::deserialize()
            .unwrap_or_else(|e| {
                eprintln!("Failed to load train data, falling back to default: {e}");
                let heightmap_params = HeightMapParams::new();

                // Fall back to the default params if the heightmap failed to generate with the params
                let heightmap = HeightMap::new(&heightmap_params)
                    .or_else(|_| HeightMap::new(&HeightMapParams::new()))
                    .unwrap();

                let mut structures = Structures::new(HashMap::new());
                structures.add_structure(Structure::new_sink(Vec2::new(0., 0.), 0.));

                (
                    Train::new(),
                    TrainTracks::new(),
                    heightmap_params,
                    heightmap,
                    structures,
                    0,
                )
            });

        Self {
            transform: Transform::new(1.),
            heightmap,
            heightmap_params,
            contour_grid_step,
            bg: HashMap::new(),
            show_contours: true,
            show_grid: false,
            show_debug_slope: false,
            focus_on_train: false,
            click_mode: ClickMode::None,
            cursor: None,
            belt_connection: None,
            pipe_connection: None,
            building_structure: None,
            wire_start: None,
            tracks,
            train,
            selected_station: None,
            new_station: "New Station".to_string(),
            ore_veins: vec![],
            structures,
            power_stats: PowerStats::default(),
            credits,
            time: 0,
            error_msg: None,
        }
    }

    fn deserialize() -> Result<
        (
            Train,
            TrainTracks,
            HeightMapParams,
            HeightMap,
            Structures,
            u32,
        ),
        String,
    > {
        let train_json = std::fs::File::open(TRAIN_JSON).map_err(|e| e.to_string())?;
        let mut value: serde_json::Value =
            serde_json::from_reader(std::io::BufReader::new(train_json))
                .map_err(|e| e.to_string())?;
        let train_value = value
            .get_mut(TRAIN_KEY)
            .ok_or_else(|| "train doesn't exist".to_string())?
            .take();
        let train: Train = serde_json::from_value(train_value).map_err(|e| e.to_string())?;

        let tracks_value = value
            .get_mut(TRACKS_KEY)
            .ok_or_else(|| "train_track doesn't exist".to_string())?
            .take();

        let tracks: TrainTracks =
            serde_json::from_value(tracks_value).map_err(|e| e.to_string())?;

        let heightmap_value = value
            .get_mut(HEIGHTMAP_KEY)
            .ok_or_else(|| format!("{HEIGHTMAP_KEY}  doesn't exist"))?
            .take();

        let heightmap_params: HeightMapParams =
            serde_json::from_value(heightmap_value).map_err(|e| e.to_string())?;

        // Fall back to the default params if the heightmap failed to generate with the params
        let heightmap = HeightMap::new(&heightmap_params)
            .or_else(|_| HeightMap::new(&HeightMapParams::new()))
            .unwrap();

        let structures: Structures = serde_json::from_value(
            value
                .get_mut(STRUCTURES_KEY)
                .ok_or_else(|| "structures doesn't exist".to_string())?
                .take(),
        )
        .map_err(|e| e.to_string())?;

        let credits = value
            .get(CREDITS_KEY)
            .ok_or_else(|| "credits doesn't exist".to_string())?
            .as_u64()
            .ok_or_else(|| "credits is not a number".to_string())? as u32;

        Ok((
            train,
            tracks,
            heightmap_params,
            heightmap,
            structures,
            credits,
        ))
    }

    fn process_result(&mut self, pos: Vec2<f64>, res: Result<(), String>) {
        if let Err(e) = res {
            self.error_msg = Some((e, 10.));
        } else {
            println!("Added point {pos:?}");
        }
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
                            );
                        }
                    }
                    ClickMode::AddOreMine => {
                        if let Err(e) = self.add_ore_mine(paint_transform.from_pos2(pointer)) {
                            self.error_msg = Some((e, 10.));
                        }
                    }
                    ClickMode::AddSmelter => {
                        if let Some(pos) = self.building_structure {
                            let delta = pos - paint_transform.from_pos2(pointer);
                            let orient = delta.y.atan2(delta.x) - std::f64::consts::PI * 0.5;
                            self.structures
                                .add_structure(Structure::new_smelter(pos, orient));
                            self.building_structure = None;
                        } else if !self.heightmap.is_water(&paint_transform.from_pos2(pointer)) {
                            self.building_structure = Some(paint_transform.from_pos2(pointer));
                        } else {
                            self.error_msg = Some(("Cannot build in water".to_string(), 10.));
                        }
                    }
                    ClickMode::AddLoader | ClickMode::AddUnloader => {
                        // Loaders orientation is determined by the track normal, so we do not need two-step method to
                        // insert one.
                        let pos = paint_transform.from_pos2(pointer);
                        if let Some((st_id, car_idx, pos, _, orient)) =
                            self.tracks.find_loader_position(pos)
                        {
                            self.structures.add_structure(
                                if matches!(self.click_mode, ClickMode::AddLoader) {
                                    Structure::new_loader(pos, orient, st_id, car_idx)
                                } else {
                                    Structure::new_unloader(
                                        pos,
                                        orient + std::f64::consts::PI,
                                        st_id,
                                        car_idx,
                                    )
                                },
                            );
                            self.building_structure = None;
                        }
                    }
                    ClickMode::AddSplitter | ClickMode::AddMerger => {
                        if let Some(pos) = self.building_structure {
                            let delta = pos - paint_transform.from_pos2(pointer);
                            let orient = delta.y.atan2(delta.x) - std::f64::consts::PI * 0.5;
                            self.structures.add_structure(Structure::new_structure(
                                match self.click_mode {
                                    ClickMode::AddSplitter => StructureType::Splitter,
                                    ClickMode::AddMerger => StructureType::Merger,
                                    _ => unreachable!(),
                                },
                                pos,
                                orient,
                            ));
                            self.building_structure = None;
                        } else if !self.heightmap.is_water(&paint_transform.from_pos2(pointer)) {
                            self.building_structure = Some(paint_transform.from_pos2(pointer));
                        } else {
                            self.error_msg = Some(("Cannot build in water".to_string(), 10.));
                        }
                    }
                    ClickMode::AddWaterPump => {
                        if let Err(e) = self.add_water_pump(paint_transform.from_pos2(pointer)) {
                            self.error_msg = Some((e, 10.));
                        }
                    }
                    ClickMode::AddBoiler => {
                        if let Err(e) = self.add_hydrophoric_structure(
                            paint_transform.from_pos2(pointer),
                            StructureType::Boiler,
                        ) {
                            self.error_msg = Some((e, 10.));
                        }
                    }
                    ClickMode::AddSteamEngine => {
                        if let Err(e) = self.add_hydrophoric_structure(
                            paint_transform.from_pos2(pointer),
                            StructureType::SteamEngine,
                        ) {
                            self.error_msg = Some((e, 10.));
                        }
                    }
                    ClickMode::ConnectBelt => {
                        if let Err(e) = self.add_belt(paint_transform.from_pos2(pointer)) {
                            self.error_msg = Some((e, 10.));
                        }
                    }
                    ClickMode::ConnectPipe => {
                        if let Err(e) = self.add_pipe(paint_transform.from_pos2(pointer)) {
                            self.error_msg = Some((e, 10.));
                        }
                    }
                    ClickMode::ConnectWire => {
                        let clicked_pos = paint_transform.from_pos2(pointer);
                        if let Some((wire_start, start_st)) = self.wire_start
                            && let Some(end_st) = self.find_structure(clicked_pos)
                        {
                            self.structures.add_wire(start_st, end_st);
                            self.wire_start = None;
                        } else if let Some(start_st) = self.find_structure(clicked_pos) {
                            self.wire_start = Some((clicked_pos, start_st));
                        }
                    }
                    ClickMode::DeleteStructure => {
                        const SELECT_THRESHOLD: f64 = 10.;
                        let pos = paint_transform.from_pos2(pointer);
                        self.structures
                            .delete(pos, SELECT_THRESHOLD / self.transform.scale() as f64);
                    }
                }
            }
        }

        self.render_heightmaps(&painter, &paint_transform);

        // Contours are always rendered with a cache now.
        // if self.use_cached_contours {
        self.render_contours_with_cache(
            &painter,
            &|p| paint_transform.transform_pos2(p),
            self.contour_grid_step,
        );
        // } else {
        //     self.render_contours(
        //         &painter,
        //         &|p| paint_transform.transform_pos2(p),
        //         self.contour_grid_step,
        //     );
        // }

        if 1. < self.transform.scale() {
            for ore_vein in &self.ore_veins {
                painter.circle(
                    paint_transform.to_pos2(ore_vein.pos),
                    7.5,
                    match ore_vein.ty {
                        OreType::Iron => Color32::from_rgb(127, 127, 191),
                        OreType::Coal => Color32::from_rgb(127, 127, 63),
                    },
                    (2., Color32::BLACK),
                );
            }
        }

        self.render_track(&painter, &paint_transform);

        self.render_structures(&painter, &paint_transform);

        self.render_belts(&painter, &paint_transform);

        self.render_pipes(&painter, &paint_transform);

        self.render_wires(&painter, &paint_transform);

        self.cursor = if let Some(pos) = response.hover_pos() {
            Some(paint_transform.from_pos2(pos))
        } else {
            None
        };

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
                        let station =
                            Station::new(self.new_station.clone(), path_id, node_id as f64);
                        self.render_station(&painter, &paint_transform, &station, false, true);
                    }
                }
            }
            ClickMode::AddOreMine => {
                if let Some(pointer) = response.hover_pos() {
                    self.preview_ore_mine(pointer, &painter, &paint_transform);
                }
            }
            ClickMode::AddSmelter
            | ClickMode::AddSplitter
            | ClickMode::AddMerger
            | ClickMode::AddWaterPump
            | ClickMode::AddBoiler
            | ClickMode::AddSteamEngine => {
                if let Some(pointer) = response.hover_pos() {
                    let ty = match self.click_mode {
                        ClickMode::AddSmelter => StructureType::Smelter,
                        ClickMode::AddSplitter => StructureType::Splitter,
                        ClickMode::AddMerger => StructureType::Merger,
                        ClickMode::AddWaterPump => StructureType::WaterPump,
                        ClickMode::AddBoiler => StructureType::Boiler,
                        ClickMode::AddSteamEngine => StructureType::SteamEngine,
                        _ => unreachable!(),
                    };
                    if let Some(pos) = self.building_structure {
                        let delta = pos - paint_transform.from_pos2(pointer);
                        let orient = delta.y.atan2(delta.x) - std::f64::consts::PI * 0.5;
                        Self::render_structure(
                            paint_transform.to_pos2(pos),
                            orient,
                            true,
                            ty,
                            &painter,
                            &paint_transform,
                        );
                    } else {
                        Self::render_structure(pointer, 0., true, ty, &painter, &paint_transform);
                    }
                }
            }
            ClickMode::AddLoader | ClickMode::AddUnloader => {
                if let Some(pointer) = response.hover_pos() {
                    let pos = paint_transform.from_pos2(pointer);
                    if let Some((_, _, pos, _, orient)) = self.tracks.find_loader_position(pos) {
                        let (ty, orient) = if matches!(self.click_mode, ClickMode::AddLoader) {
                            (StructureType::Loader, orient)
                        } else {
                            (StructureType::Unloader, orient + std::f64::consts::PI)
                        };
                        Self::render_structure(
                            paint_transform.to_pos2(pos),
                            orient,
                            true,
                            ty,
                            &painter,
                            &paint_transform,
                        );
                    }
                }
            }
            ClickMode::ConnectBelt => {
                if let Some(pointer) = response.hover_pos() {
                    let pos = paint_transform.from_pos2(pointer);
                    self.preview_belt(pos, &painter, &paint_transform);
                }
            }
            ClickMode::ConnectPipe => {
                if let Some(pointer) = response.hover_pos() {
                    let pos = paint_transform.from_pos2(pointer);
                    self.preview_pipe(pos, &painter, &paint_transform);
                }
            }
            ClickMode::ConnectWire => {
                self.preview_wire(response.hover_pos(), &painter, &paint_transform);
            }
            ClickMode::DeleteStructure => {
                if let Some(pointer) = response.hover_pos() {
                    self.preview_delete_structure(pointer, &painter, &paint_transform);
                }
            }
        }

        // Clear the state of inserting structure when the player select another mode
        if !matches!(
            self.click_mode,
            ClickMode::AddOreMine
                | ClickMode::AddSmelter
                | ClickMode::AddSplitter
                | ClickMode::AddMerger
                | ClickMode::AddWaterPump
                | ClickMode::AddBoiler
                | ClickMode::AddSteamEngine
        ) {
            self.building_structure = None;
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
        ui.collapsing("View options", |ui| {
            ui.checkbox(&mut self.show_contours, "Show contour lines");
            ui.horizontal(|ui| {
                ui.label("Contours grid step:");
                ui.add(egui::Slider::new(
                    &mut self.contour_grid_step,
                    1..=MAX_CONTOURS_GRID_STEP,
                ));
            });
            ui.checkbox(&mut self.show_grid, "Show grid");
            ui.checkbox(&mut self.show_debug_slope, "Show debug slope");
            ui.checkbox(&mut self.focus_on_train, "Focus on train");
            ui.horizontal(|ui| {
                ui.label("Num cars:");
                ui.add(egui::Slider::new(
                    &mut self.train.cars.len(),
                    1..=MAX_NUM_CARS,
                ));
            });
        });
        ui.collapsing("Terrain generation params", |ui| {
            self.heightmap_params.params_ui(ui);
            if ui.button("Regenerate").clicked() {
                match HeightMap::new(&self.heightmap_params) {
                    Ok(map) => {
                        self.heightmap = map;
                        self.bg.clear();
                    }
                    Err(e) => {
                        self.error_msg = Some((e.to_string(), 10.));
                    }
                }
            }
        });
        ui.group(|ui| {
            let was_belt_mode = matches!(self.click_mode, ClickMode::ConnectBelt);
            ui.label("Click mode:");
            ui.radio_value(&mut self.click_mode, ClickMode::None, "None");
            ui.radio_value(
                &mut self.click_mode,
                ClickMode::GentleCurve,
                "Add Rail (Gentle Curve)",
            );
            ui.radio_value(
                &mut self.click_mode,
                ClickMode::TightCurve,
                "Add Rail (Tight Curve)",
            );
            ui.radio_value(
                &mut self.click_mode,
                ClickMode::StraightLine,
                "Add Rail (Straight Line)",
            );
            ui.radio_value(
                &mut self.click_mode,
                ClickMode::BezierCurve,
                "Add Rail (Bezier Curve)",
            );
            ui.radio_value(
                &mut self.click_mode,
                ClickMode::DeleteSegment,
                "Delete Rail Segment",
            );
            ui.radio_value(&mut self.click_mode, ClickMode::AddStation, "Add Station");
            ui.radio_value(&mut self.click_mode, ClickMode::AddOreMine, "Add Ore Mine");
            ui.radio_value(&mut self.click_mode, ClickMode::AddSmelter, "Add Smelter");
            ui.radio_value(&mut self.click_mode, ClickMode::AddLoader, "Add Loader");
            ui.radio_value(&mut self.click_mode, ClickMode::AddUnloader, "Add Unloader");
            ui.radio_value(&mut self.click_mode, ClickMode::AddSplitter, "Add Splitter");
            ui.radio_value(&mut self.click_mode, ClickMode::AddMerger, "Add Merger");
            ui.radio_value(
                &mut self.click_mode,
                ClickMode::AddWaterPump,
                "Add Water Pump",
            );
            ui.radio_value(&mut self.click_mode, ClickMode::AddBoiler, "Add Boiler");
            ui.radio_value(
                &mut self.click_mode,
                ClickMode::AddSteamEngine,
                "Add Steam Engine",
            );
            ui.radio_value(&mut self.click_mode, ClickMode::ConnectBelt, "Connect Belt");
            ui.radio_value(&mut self.click_mode, ClickMode::ConnectPipe, "Connect Pipe");
            ui.radio_value(&mut self.click_mode, ClickMode::ConnectWire, "Connect Wire");
            ui.radio_value(
                &mut self.click_mode,
                ClickMode::DeleteStructure,
                "Delete Structures",
            );
            let is_belt_mode = matches!(self.click_mode, ClickMode::ConnectBelt);
            if !is_belt_mode {
                self.belt_connection = None;
            }
            if is_belt_mode != was_belt_mode {
                // Re-render the map to highlight areas that exceeds slope.
                // ui.radio_value().changed() can detect changes by clicking the option, but cannot detect when
                // the option is passively deselected by clicking another.
                self.bg.clear();
            }
            let is_pipe_mode = matches!(self.click_mode, ClickMode::ConnectPipe);
            if !is_pipe_mode {
                self.pipe_connection = None;
            }
        });
        ui.group(|ui| {
            ui.label("Stations:");
            for (i, station) in &self.tracks.stations {
                ui.radio_value(
                    &mut self.selected_station,
                    Some(*i),
                    &format!("{} ({}, {})", station.name, station.path_id, station.s),
                );
            }
            ui.checkbox(&mut self.train.auto_drive, "Auto drive");
            if ui.button("Schedule station").clicked() {
                if let Some(target) = self.selected_station {
                    self.train.schedule.push(target);
                    self.train.auto_drive = true;
                }
            }
            if ui.button("Delete station").clicked() {
                if let Some(target) = self.selected_station {
                    self.tracks.stations.remove(&target);
                }
            }
            ui.text_edit_singleline(&mut self.new_station);
        });
        ui.group(|ui| {
            ui.label(if let Some(cursor) = &self.cursor {
                format!(
                    "Cursor: ({:.3},{:.3}) ({:?})\nheight: {:.3}\nis_water: {}",
                    cursor.x,
                    cursor.y,
                    HeightMap::key_from_pos(cursor),
                    self.heightmap.get_height(cursor),
                    self.heightmap.is_water(cursor)
                )
            } else {
                format!("Cursor: None")
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
            ui.label(&format!("Train task: {:?}", self.train.train_task));
            ui.label("Train schedule:");
            for (i, sched) in self.train.schedule.iter().rev().enumerate() {
                ui.label(&format!(
                    "  {}{}",
                    if i == 0 { "* " } else { "  " },
                    self.tracks
                        .stations
                        .get(sched)
                        .map_or("(None)", |st| &st.name)
                ));
            }
            ui.label(format!(
                "Total transported items: {}",
                self.train.total_transported
            ));
        });
        ui.group(|ui| {
            ui.label("Global stats");
            ui.label(format!("Power demand: {:.1} kW", self.power_stats.demand));
            ui.label(format!("Power supply: {:.1} kW", self.power_stats.supply));
            ui.label(format!(
                "Power sufficiency: {:.3} %",
                self.power_stats.sufficiency * 100.
            ));
            ui.label(format!("Credits: {}", self.credits));
        });
    }
}

impl eframe::App for TrainsApp {
    fn update(&mut self, ctx: &eframe::egui::Context, _frame: &mut eframe::Frame) {
        install_image_loaders(ctx);

        ctx.include_bytes(ORE_URL, include_bytes!("../img/ore.png"));
        ctx.include_bytes(INGOT_URL, include_bytes!("../img/metal.png"));
        ctx.include_bytes(COAL_URL, include_bytes!("../img/coal-ore.png"));

        ctx.request_repaint();

        // Give the heightmap object an opportunity to process queued map generations
        match self.heightmap.update(self.contour_grid_step) {
            Ok(tiles_to_update) => {
                for pos in tiles_to_update {
                    if let Some(tile) = self.heightmap.tiles.get(&HeightMapKey { pos, level: 0 }) {
                        for mut ov in self.heightmap.gen_ore_veins(pos, &tile) {
                            ov.occupied_miner =
                                self.structures.structures.iter_mut().find_map(|(id, st)| {
                                    if st.pos == ov.pos {
                                        st.ore_type = Some(ov.ty);
                                        Some(*id)
                                    } else {
                                        None
                                    }
                                });
                            self.ore_veins.push(ov);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("Error heightmap update: {e}");
            }
        }

        // Decay error message even if paused
        if let Some((_, ref mut time)) = self.error_msg {
            let dt = ctx.input(|i| i.raw.predicted_dt);
            *time = *time - dt as f64;
            if *time < 0. {
                self.error_msg = None;
            }
        }

        self.time += 1;

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

        self.power_stats = self.structures.update(&mut self.credits, &mut self.train);
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
            STRUCTURES_KEY.to_string(),
            serde_json::to_value(&self.structures).unwrap(),
        );
        map.insert(
            CREDITS_KEY.to_string(),
            serde_json::Value::Number(self.credits.into()),
        );
        let _ = serde_json::to_writer_pretty(
            std::io::BufWriter::new(std::fs::File::create(TRAIN_JSON).unwrap()),
            &map,
        );
    }
}
