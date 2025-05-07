mod heightmap;

use self::heightmap::init_heightmap;
use eframe::{
    egui::{Align2, Color32, FontId, Frame, Painter, Pos2, Ui, pos2},
    epaint::PathShape,
};

use crate::{
    bg_image::BgImage,
    train::{Station, Train},
    transform::{PaintTransform, Transform, half_rect},
    vec2::Vec2,
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
    train: Train,
}

impl TrainsApp {
    pub fn new() -> Self {
        Self {
            transform: Transform::new(1.),
            heightmap: init_heightmap(),
            bg: BgImage::new(),
            show_contours: true,
            show_grid: false,
            train: Train::new(),
        }
    }

    fn render(&mut self, ui: &mut Ui) {
        let (response, painter) =
            ui.allocate_painter(ui.available_size(), eframe::egui::Sense::hover());

        if ui.ui_contains_pointer() {
            ui.input(|i| self.transform.handle_mouse(i, half_rect(&response.rect)));
        }

        let paint_transform = self.transform.into_paint(&response);

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
            paint_transform
                .transform_pos2(pos2(0., AREA_HEIGHT as f32))
                .into(),
            self.transform.scale(),
        );

        self.render_contours(&painter, &|p| paint_transform.transform_pos2(p));

        if 1. < self.transform.scale() {
            if let Some(ghost_segments) = &self.train.ghost_path {
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

        for (_i, station) in self.train.stations.iter().enumerate() {
            let i_ptr = &*station.borrow() as *const _;
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
        const TIE_HALFLENGTH: f64 = 1.5;
        const TIE_HALFWIDTH: f64 = 0.3;
        const TIE_INTERPOLATES: usize = 3;

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
