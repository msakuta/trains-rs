use eframe::{
    egui::{self, Color32, Painter, Pos2},
    epaint::PathShape,
};

use crate::{transform::PaintTransform, vec2::Vec2};

use super::TrainsApp;

const RAIL_HALFWIDTH: f64 = 1.25;
// const TIE_HALFLENGTH: f64 = 1.5;
// const TIE_HALFWIDTH: f64 = 0.3;
// const TIE_INTERPOLATES: usize = 3;

impl TrainsApp {
    pub(super) fn render_track(&self, painter: &Painter, paint_transform: &PaintTransform) {
        let ghost_path = self.train.ghost_path.as_ref().map(|ghost_segments| {
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
            for bundle in self.train.paths.values() {
                self.render_track_detail(&bundle.track, &painter, &paint_transform, 1., color);
            }
        } else {
            if let Some((ghost_segments, color)) = ghost_path {
                self.render_track_simple(&ghost_segments.track, &painter, &paint_transform, color);
            }
            for bundle in self.train.paths.values() {
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
            if let Some((last, last2)) = track.last().zip(track.get(track.len() - 2)) {
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
        track: &[Vec2<f64>],
        painter: &Painter,
        paint_transform: &PaintTransform,
    ) {
        let ofs = RAIL_HALFWIDTH * 2.;

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

        if let Some((first, second)) = track.get(0).zip(track.get(1)) {
            let par = parallel_offset(ofs)((first, second));
            painter.arrow(
                par,
                perpendicular_offset(ofs * 2.)((&first, &second)),
                (2., Color32::RED),
            );
        }

        if let [first, second] = track[track.len() - 2..] {
            let par = parallel_offset(ofs)((&first, &second));
            painter.arrow(
                par,
                perpendicular_offset(ofs * 2.)((&first, &second)),
                (2., Color32::from_rgb(0, 127, 0)),
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
}
