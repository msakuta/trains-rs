mod heightmap;

use std::rc::Rc;

use eframe::{
    egui::{self, Align2, Color32, FontId, Frame, Painter, Pos2, Ui},
    epaint::PathShape,
};
use heightmap::DOWNSAMPLE;

pub(crate) use self::heightmap::HeightMap;
use self::heightmap::{ContoursCache, HeightMapParams};

use crate::{
    bg_image::BgImage,
    train::{Station, Train},
    transform::{PaintTransform, Transform, half_rect},
    vec2::Vec2,
};

pub(crate) const AREA_WIDTH: usize = 512;
pub(crate) const AREA_HEIGHT: usize = 512;
// const AREA_SHAPE: Shape = (AREA_WIDTH as isize, AREA_HEIGHT as isize);
const SELECT_PIXEL_RADIUS: f64 = 20.;
const MAX_NUM_CARS: usize = 10;
const MAX_CONTOURS_GRID_STEP: usize = 100;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ClickMode {
    None,
    GentleCurve,
    TightCurve,
    StraightLine,
    DeleteSegment,
    AddStation,
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
    click_mode: ClickMode,
    train: Train,
    selected_station: Option<usize>,
    new_station: String,
    error_msg: Option<(String, f64)>,
}

impl TrainsApp {
    pub fn new() -> Self {
        let heightmap_params = HeightMapParams::new();
        let heightmap = HeightMap::new(&heightmap_params).unwrap();
        let contour_grid_step = DOWNSAMPLE;

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
            click_mode: ClickMode::None,
            train: Train::new(),
            selected_station: None,
            new_station: "New Station".to_string(),
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

    fn render(&mut self, ui: &mut Ui) {
        let (response, painter) = ui.allocate_painter(ui.available_size(), egui::Sense::click());

        if ui.ui_contains_pointer() {
            ui.input(|i| self.transform.handle_mouse(i, half_rect(&response.rect)));
        }

        let paint_transform = self.transform.into_paint(&response);

        if response.clicked() {
            if let Some(pointer) = response.interact_pointer_pos() {
                match self.click_mode {
                    ClickMode::None => {}
                    ClickMode::GentleCurve => {
                        let pos = paint_transform.from_pos2(pointer);
                        // self.train.control_points.push(pos);
                        let res = self.train.add_gentle(pos);
                        self.process_result(pos, res);
                    }
                    ClickMode::TightCurve => {
                        let pos = paint_transform.from_pos2(pointer);
                        let res = self.train.add_tight(pos);
                        self.process_result(pos, res);
                    }
                    ClickMode::StraightLine => {
                        let pos = paint_transform.from_pos2(pointer);
                        let res = self.train.add_straight(pos);
                        self.process_result(pos, res);
                    }
                    ClickMode::DeleteSegment => {
                        let pos = paint_transform.from_pos2(pointer);
                        let thresh = SELECT_PIXEL_RADIUS / self.transform.scale() as f64;
                        let res = self.train.delete_segment(pos, thresh);
                        self.process_result(pos, res);
                    }
                    ClickMode::AddStation => {
                        let pos = paint_transform.from_pos2(pointer);
                        let thresh = SELECT_PIXEL_RADIUS / self.transform.scale() as f64;
                        let next_name = (0..).find_map(|i| {
                            let cand = format!("New Station {i}");
                            if !self.train.stations.iter().any(|s| s.borrow().name == cand)
                                && self.new_station != cand
                            {
                                Some(cand)
                            } else {
                                None
                            }
                        });
                        if let Some(name) = next_name {
                            self.train.add_station(
                                std::mem::replace(&mut self.new_station, name),
                                pos,
                                thresh,
                            );
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

        if 1. < self.transform.scale() {
            if let Some(ghost_segments) = &self.train.ghost_path {
                let is_intersecting_water = ghost_segments
                    .track
                    .iter()
                    .any(|p| self.heightmap.is_water(p));
                let color = Color32::from_rgba_premultiplied(255, 0, 255, 63);
                self.render_track_detail(
                    &ghost_segments.track,
                    &painter,
                    &paint_transform,
                    1.,
                    color,
                );
            }
            let color = Color32::from_rgba_premultiplied(255, 0, 255, 255);
            for bundle in self.train.paths.values() {
                self.render_track_detail(&bundle.track, &painter, &paint_transform, 1., color);
            }
        } else {
            if let Some(ghost_segments) = &self.train.ghost_path {
                self.render_track(&ghost_segments.track, &painter, &paint_transform, 63);
            }
            for bundle in self.train.paths.values() {
                self.render_track(&bundle.track, &painter, &paint_transform, 255);
            }
        }

        let rotation_matrix = |angle: f32| {
            [
                angle.cos() as f32,
                angle.sin() as f32,
                -angle.sin() as f32,
                angle.cos() as f32,
            ]
        };
        let rotate_vec = |rotation: &[f32; 4], ofs: &[f32; 2]| {
            [
                rotation[0] * ofs[0] + rotation[1] * ofs[1],
                rotation[2] * ofs[0] + rotation[3] * ofs[1],
            ]
        };
        let scale_vec = |scale: f32, vec: &[f32; 2]| [vec[0] * scale, vec[1] * scale];

        match self.click_mode {
            ClickMode::None => self.train.ghost_path = None,
            ClickMode::GentleCurve => {
                if let Some(pos) = response.hover_pos() {
                    self.train.ghost_gentle(paint_transform.from_pos2(pos));
                } else {
                    self.train.ghost_path = None;
                }
            }
            ClickMode::StraightLine => {
                if let Some(pos) = response.hover_pos() {
                    self.train.ghost_straight(paint_transform.from_pos2(pos));
                } else {
                    self.train.ghost_path = None;
                }
            }
            ClickMode::TightCurve => {
                if let Some(pos) = response.hover_pos() {
                    self.train.ghost_tight(paint_transform.from_pos2(pos));
                } else {
                    self.train.ghost_path = None;
                }
            }
            ClickMode::DeleteSegment => {
                let found_node = response.hover_pos().and_then(|pointer| {
                    let thresh = SELECT_PIXEL_RADIUS / self.transform.scale() as f64;
                    self.train
                        .find_path_node(paint_transform.from_pos2(pointer), thresh)
                });
                if let Some((path_id, seg_id, _)) = found_node {
                    let color = Color32::from_rgba_premultiplied(127, 0, 127, 63);
                    if let Some(path) = self.train.paths.get(&path_id) {
                        let seg_track = path.seg_track(seg_id);
                        self.render_track_detail(seg_track, &painter, &paint_transform, 5., color);
                    }
                }
            }
            ClickMode::AddStation => {
                if let Some(pointer) = response.hover_pos() {
                    let pos = paint_transform.from_pos2(pointer);
                    let thresh = SELECT_PIXEL_RADIUS / self.transform.scale() as f64;
                    if let Some((path_id, _, node_id)) = self.train.find_path_node(pos, thresh) {
                        let station =
                            Station::new(self.new_station.clone(), path_id, node_id as f64);
                        self.render_station(&painter, &paint_transform, &station, false, true);
                    }
                }
            }
        }

        for (_i, station) in self.train.stations.iter().enumerate() {
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
            self.render_station(
                &painter,
                &paint_transform,
                &station.borrow(),
                is_target,
                false,
            );
        }

        let paint_train = |pos: &Vec2<f64>, heading: f64, tangent: &Vec2<f64>| {
            let base_pos = paint_transform.to_pos2(*pos).to_vec2();
            let rotation = rotation_matrix(heading as f32);
            let transform_delta =
                |ofs: &[f32; 2]| scale_vec(self.transform.scale(), &rotate_vec(&rotation, ofs));
            let transform_vec = |ofs: &[f32; 2]| Pos2::from(transform_delta(ofs)) + base_pos;
            let convert_to_poly = |vertices: &[[f32; 2]]| {
                PathShape::closed_line(
                    vertices.into_iter().map(|ofs| transform_vec(ofs)).collect(),
                    (1., Color32::RED),
                )
            };

            painter.add(convert_to_poly(&[
                [-2., -2.],
                [6., -2.],
                [6., 2.],
                [-2., 2.],
            ]));

            // let paint_wheel = |ofs: &[f32; 2], rotation: &[f32; 4]| {
            //     use eframe::emath::Vec2;
            //     let middle = transform_vec(ofs);
            //     let front =
            //         middle + Vec2::from(rotate_vec(rotation, &[self.transform.scale(), 0.]));
            //     let back = middle - Vec2::from(rotate_vec(rotation, &[self.transform.scale(), 0.]));

            //     painter.line_segment([front, back], (2., Color32::BLACK));
            // };

            // paint_wheel(&[0., 0.], &rotation);

            if self.show_debug_slope {
                let grad = self.heightmap.gradient(pos);
                let tangent = tangent.normalized();
                let accel = -grad.dot(tangent);
                let start = paint_transform.to_pos2(*pos);
                let end = paint_transform.to_pos2(*pos - grad * 100.);
                let tangent_end = paint_transform.to_pos2(*pos + tangent * accel * 100.);
                painter.line_segment([start, end], (2., Color32::RED));
                painter.line_segment([start, tangent_end], (2., Color32::BLUE));
            }
        };

        for i in 0..self.train.num_cars {
            if let Some(((train_pos, train_heading), tangent)) = self
                .train
                .train_pos(i)
                .zip(self.train.heading(i))
                .zip(self.train.tangent(i))
            {
                paint_train(&train_pos, train_heading, &tangent);
            }
        }

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

        let Some(pos) = self.train.s_pos(station.path_id, station.s) else {
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

    fn render_track_detail(
        &self,
        track: &[Vec2<f64>],
        painter: &Painter,
        paint_transform: &PaintTransform,
        line_width: f32,
        color: Color32,
    ) {
        let parallel_offset = |ofs| {
            let paint_transform = &paint_transform;
            move |(prev, next): (&Vec2<f64>, &Vec2<f64>)| {
                let delta = (*next - *prev).normalized();
                Pos2::from(paint_transform.to_pos2(delta.left90() * ofs + *prev))
            }
        };

        const RAIL_HALFWIDTH: f64 = 1.25;
        // const TIE_HALFLENGTH: f64 = 1.5;
        // const TIE_HALFWIDTH: f64 = 0.3;
        // const TIE_INTERPOLATES: usize = 3;

        for ofs in [RAIL_HALFWIDTH, -RAIL_HALFWIDTH] {
            let left_rail_points = track
                .iter()
                .zip(track.iter().skip(1))
                .map(parallel_offset(ofs))
                .collect();
            let left_rail = PathShape::line(left_rail_points, (line_width, color));
            painter.add(left_rail);
        }

        // if self.show_rail_ties {
        //     for (prev, next) in track.iter().zip(track.iter().skip(1)) {
        //         let delta = *next - *prev;
        //         let tangent = delta.normalized();
        //         for i in 0..TIE_INTERPOLATES {
        //             let offset = *prev + delta * i as f64 / TIE_INTERPOLATES as f64;
        //             let left = tangent.left90() * TIE_HALFLENGTH + offset;
        //             let right = tangent.left90() * -TIE_HALFLENGTH + offset;
        //             let left_front = left + tangent * TIE_HALFWIDTH;
        //             let left_back = left + tangent * -TIE_HALFWIDTH;
        //             let right_front = right + tangent * TIE_HALFWIDTH;
        //             let right_back = right + tangent * -TIE_HALFWIDTH;
        //             let tie = PathShape::closed_line(
        //                 [left_front, right_front, right_back, left_back]
        //                     .into_iter()
        //                     .map(|v| paint_transform.to_pos2(v))
        //                     .collect(),
        //                 (line_width, color),
        //             );
        //             painter.add(tie);
        //         }
        //     }
        // }
    }

    fn render_track(
        &self,
        track: &[Vec2<f64>],
        painter: &Painter,
        paint_transform: &PaintTransform,
        alpha: u8,
    ) {
        let track_points: Vec<_> = track
            .iter()
            .map(|ofs| Pos2::from(paint_transform.to_pos2(*ofs)))
            .collect();

        let color = Color32::from_rgba_premultiplied(255, 0, 255, alpha);

        // if self.show_track_nodes {
        //     for track_point in &track_points {
        //         painter.circle_filled(*track_point, 3., color);
        //     }
        // }

        let track_line = PathShape::line(track_points, (2., color));
        painter.add(track_line);
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
        ui.horizontal(|ui| {
            ui.label("Num cars:");
            ui.add(egui::Slider::new(
                &mut self.train.num_cars,
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
            ui.radio_value(&mut self.click_mode, ClickMode::DeleteSegment, "Delete");
            ui.radio_value(&mut self.click_mode, ClickMode::AddStation, "Add Station");
        });
        ui.group(|ui| {
            ui.label("Stations:");
            for (i, rc_station) in self.train.stations.iter().enumerate() {
                let station = rc_station.borrow();
                ui.radio_value(
                    &mut self.selected_station,
                    Some(i),
                    &format!("{} ({}, {})", station.name, station.path_id, station.s),
                );
            }
            if ui.button("Schedule station").clicked() {
                if let Some(target) = self.selected_station {
                    self.train
                        .schedule
                        .push(Rc::downgrade(&self.train.stations[target]));
                }
            }
            if ui.button("Delete station").clicked() {
                if let Some(target) = self.selected_station {
                    self.train.stations.remove(target);
                }
            }
            ui.text_edit_singleline(&mut self.new_station);
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
            .show(ctx, |ui| self.ui_panel(ui));

        eframe::egui::CentralPanel::default().show(ctx, |ui| {
            Frame::canvas(ui.style()).show(ui, |ui| {
                self.render(ui);
            });
        });

        let mut thrust = 0.;
        ctx.input(|input| {
            for key in input.keys_down.iter() {
                match key {
                    egui::Key::W => thrust += 1.,
                    egui::Key::S => thrust -= 1.,
                    _ => {}
                }
            }
        });
        self.train.update(thrust, &self.heightmap);
    }
}
