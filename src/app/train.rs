//! Train and train tracks rendering logic.

use eframe::{
    egui::{self, Color32, Painter, Pos2},
    epaint::PathShape,
};

use crate::{
    train::TrainCar,
    train_tracks::{PathBundle, SegmentDirection},
    transform::PaintTransform,
    vec2::Vec2,
};

use super::TrainsApp;

const RAIL_HALFWIDTH: f64 = 1.25;
// const TIE_HALFLENGTH: f64 = 1.5;
// const TIE_HALFWIDTH: f64 = 0.3;
// const TIE_INTERPOLATES: usize = 3;

impl TrainsApp {
    pub(super) fn render_track(&self, painter: &Painter, paint_transform: &PaintTransform) {
        let ghost_path = self.tracks.ghost_path.as_ref().map(|ghost_segments| {
            let is_intersecting_water = ghost_segments
                .track
                .iter()
                .any(|p| self.heightmap.is_water(p));

            let ghost_color = if is_intersecting_water {
                Color32::from_rgba_premultiplied(255, 0, 0, 127)
            } else {
                Color32::from_rgba_premultiplied(255, 0, 255, 63)
            };

            (ghost_segments, ghost_color)
        });

        let color = Color32::from_rgba_premultiplied(255, 0, 255, 255);

        if 1. < self.transform.scale() {
            if let Some((ghost_segments, color)) = ghost_path {
                self.render_track_detail(
                    &ghost_segments.track,
                    &painter,
                    &paint_transform,
                    1.,
                    color,
                );
            }
            for (id, bundle) in &self.tracks.paths {
                self.render_track_detail(&bundle.track, &painter, &paint_transform, 1., color);
                self.render_track_direction(*id, &bundle, &painter, &paint_transform);
            }
        } else {
            if let Some((ghost_segments, color)) = ghost_path {
                self.render_track_simple(&ghost_segments.track, &painter, &paint_transform, color);
            }
            for bundle in self.tracks.paths.values() {
                self.render_track_simple(&bundle.track, &painter, &paint_transform, color);
            }
        }
    }

    pub(super) fn render_track_detail(
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

        for ofs in [RAIL_HALFWIDTH, -RAIL_HALFWIDTH] {
            let mut left_rail_points: Vec<Pos2> = track
                .iter()
                .zip(track.iter().skip(1))
                .map(parallel_offset(ofs))
                .collect();

            // Extend the last segment using the same normal vector from the last segment.
            // This could be inaccurate, but is better than disconnection.
            if let Some([last2, last]) = track.get(track.len().saturating_sub(2)..) {
                let delta = (*last - *last2).normalized();
                left_rail_points.push(Pos2::from(
                    paint_transform.to_pos2(delta.left90() * ofs + *last),
                ));
            }

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

    fn render_track_direction(
        &self,
        path_id: usize,
        path: &PathBundle,
        painter: &Painter,
        paint_transform: &PaintTransform,
    ) {
        let ofs = RAIL_HALFWIDTH * 2.;

        let is_start_node_switching =
            self.tracks
                .nodes
                .get(&path.start_node.node_id)
                .is_some_and(|node| {
                    let prev_paths = node.paths_in_direction(!path.start_node.direction);
                    let cur_paths = node.paths_in_direction(path.start_node.direction);
                    prev_paths.iter().any(|p| p.path_id == self.train.path_id())
                        && cur_paths
                            .get(
                                self.train
                                    .switch_path
                                    .min(cur_paths.len().saturating_sub(1)),
                            )
                            .is_some_and(|p| p.path_id == path_id)
                });

        let is_end_node_switching =
            self.tracks
                .nodes
                .get(&path.end_node.node_id)
                .is_some_and(|node| {
                    let next_paths = node.paths_in_direction(!path.end_node.direction);
                    let cur_paths = node.paths_in_direction(path.end_node.direction);
                    next_paths.iter().any(|p| p.path_id == self.train.path_id())
                        && cur_paths
                            .get(
                                self.train
                                    .switch_path
                                    .min(cur_paths.len().saturating_sub(1)),
                            )
                            .is_some_and(|p| p.path_id == path_id)
                });

        let track = &path.track;

        let parallel_offset = |ofs| {
            let paint_transform = &paint_transform;
            move |(prev, next): (&Vec2<f64>, &Vec2<f64>)| {
                let delta = (*next - *prev).normalized();
                Pos2::from(paint_transform.to_pos2(delta.left90() * ofs + *prev))
            }
        };

        let perpendicular_offset = |ofs| {
            let scale = paint_transform.scale();
            move |(prev, next): (&Vec2<f64>, &Vec2<f64>)| {
                let delta = (*next - *prev).normalized();
                let v = delta * ofs;
                egui::vec2(v.x as f32, -v.y as f32) * scale
            }
        };

        const RENDER_DIRECTION_COUNT: usize = 5;

        if is_start_node_switching {
            for (prev, next) in track
                .iter()
                .take(RENDER_DIRECTION_COUNT)
                .zip(track.iter().skip(1))
            {
                // let delta = (*next - *prev).normalized();
                // let heading = delta.y.atan2(delta.x);
                let prev_pos = paint_transform.to_pos2(*prev);
                let next_pos = paint_transform.to_pos2(*next);
                painter.arrow(prev_pos, next_pos - prev_pos, (3., Color32::RED));
            }
        }

        if is_end_node_switching {
            for (prev, next) in track
                .iter()
                .rev()
                .take(RENDER_DIRECTION_COUNT)
                .zip(track.iter().rev().skip(1))
            {
                // let delta = (*next - *prev).normalized();
                // let heading = delta.y.atan2(delta.x);
                let prev_pos = paint_transform.to_pos2(*prev);
                let next_pos = paint_transform.to_pos2(*next);
                painter.arrow(
                    prev_pos,
                    next_pos - prev_pos,
                    (3., Color32::from_rgb(0, 191, 0)),
                );
            }
        }

        if let Some((first, second)) = track.get(0).zip(track.get(1)) {
            let par = parallel_offset(ofs)((first, second));
            painter.arrow(
                par,
                perpendicular_offset(ofs * 2.)((&first, &second)),
                (2., Color32::RED),
            );
        }

        if let [first, second] = track[track.len().saturating_sub(2)..] {
            let par = parallel_offset(ofs)((&first, &second));
            painter.arrow(
                par,
                perpendicular_offset(ofs * 2.)((&first, &second)),
                (2., Color32::from_rgb(0, 191, 0)),
            );
        }
    }

    fn render_track_simple(
        &self,
        track: &[Vec2<f64>],
        painter: &Painter,
        paint_transform: &PaintTransform,
        color: Color32,
    ) {
        let track_points: Vec<_> = track
            .iter()
            .map(|ofs| Pos2::from(paint_transform.to_pos2(*ofs)))
            .collect();

        // if self.show_track_nodes {
        //     for track_point in &track_points {
        //         painter.circle_filled(*track_point, 3., color);
        //     }
        // }

        let track_line = PathShape::line(track_points, (2., color));
        painter.add(track_line);
    }

    pub(super) fn render_train(&self, painter: &Painter, paint_transform: &PaintTransform) {
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

        let paint_car = |car: &TrainCar| {
            let pos = car.pos(&self.tracks.paths)?;
            let heading = car.heading(&self.tracks.paths)?;
            let tangent = car.tangent(&self.tracks.paths)?;
            let base_pos = paint_transform.to_pos2(pos).to_vec2();
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
                [-4., -2.],
                [4., -2.],
                [4., 2.],
                [-4., 2.],
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

            // When the user zooms in enough, draw the direction arrow.
            if 2. < self.transform.scale() {
                let direction = match car.direction {
                    SegmentDirection::Forward => egui::Vec2::from(transform_delta(&[3., 0.])),
                    SegmentDirection::Backward => egui::Vec2::from(transform_delta(&[-3., 0.])),
                };
                painter.arrow(base_pos.to_pos2(), direction, (2., Color32::WHITE));
            }

            if self.show_debug_slope {
                let grad = self.heightmap.gradient(&pos);
                let tangent = tangent.normalized();
                let accel = -grad.dot(tangent);
                let start = paint_transform.to_pos2(pos);
                let end = paint_transform.to_pos2(pos - grad * 100.);
                let tangent_end = paint_transform.to_pos2(pos + tangent * accel * 100.);
                painter.line_segment([start, end], (2., Color32::RED));
                painter.line_segment([start, tangent_end], (2., Color32::BLUE));
            }

            Some(())
        };

        for car in &self.train.cars {
            paint_car(car);
        }
    }
}
