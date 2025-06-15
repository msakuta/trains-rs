//! Methods related to tight paths, which are a pair of minimum radius arc and a straight segment.

use crate::{
    app::HeightMap,
    path_utils::{CircleArc, wrap_angle_offset},
    train::Train,
    vec2::Vec2,
};

use super::{MIN_RADIUS, NodeConnection, PathBundle, PathSegment, TrainTracks};

impl TrainTracks {
    pub fn add_tight(
        &mut self,
        pos: Vec2<f64>,
        heightmap: &HeightMap,
        train: &mut Train,
    ) -> Result<(), String> {
        let path_segments = self.compute_tight(pos)?;

        if path_segments.track.iter().any(|p| heightmap.is_water(p)) {
            return Err("Cannot build tracks through water".to_string());
        }

        self.add_segment(path_segments, None, train)
    }

    pub fn ghost_tight(&mut self, pos: Vec2<f64>) {
        self.ghost_path = self.compute_tight(pos).ok();
    }

    fn compute_tight(&self, pos: Vec2<f64>) -> Result<PathBundle, String> {
        let Some((prev_pos, prev_angle)) = self.selected_node() else {
            return Err("Select a node first".to_string());
        };

        let tangent = Vec2::new(prev_angle.cos(), prev_angle.sin());
        let normal_left = tangent.left90();
        let mut tangent_angle: Option<(f64, f64, Vec2<f64>)> = None;
        for cur in [-1., 1.] {
            let normal = normal_left * cur;
            let start_angle = (-normal.y).atan2(-normal.x);
            let arc_center = normal * MIN_RADIUS + prev_pos;
            let arc_dest = pos - arc_center;
            let arc_dest_len = arc_dest.length();
            if arc_dest_len < MIN_RADIUS {
                continue;
            }
            let beta = (MIN_RADIUS / arc_dest_len).acos();
            let arc_dest_angle = arc_dest.y.atan2(arc_dest.x);
            let angle = arc_dest_angle - cur * beta;
            let end_angle = start_angle
                + wrap_angle_offset(angle - start_angle, (1. - cur) * std::f64::consts::PI);
            if !tangent_angle
                .is_some_and(|acc| (acc.1 - acc.0).abs() < (start_angle - end_angle).abs())
            {
                tangent_angle = Some((end_angle, start_angle, arc_center));
            }
        }
        if let Some((end_angle, start_angle, a)) = tangent_angle {
            let tangent_pos = a + Vec2::new(end_angle.cos(), end_angle.sin()) * MIN_RADIUS;
            let path_segments = [
                PathSegment::Arc(CircleArc::new(a, MIN_RADIUS, start_angle, end_angle)),
                PathSegment::Line([tangent_pos, pos]),
            ];
            Ok(PathBundle::multi(
                path_segments,
                NodeConnection::default(),
                NodeConnection::default(),
            ))
        } else {
            Err("Clicked point requires tighter curvature radius than allowed".to_string())
        }
    }

    #[allow(dead_code, unused_variables)]
    fn compute_tight_orient(&self, pos: Vec2<f64>, end_angle: f64) -> Result<PathBundle, String> {
        let Some((prev_pos, prev_angle)) = self.selected_node() else {
            return Err("Select a node first".to_string());
        };

        let start_tangent = Vec2::new(prev_angle.cos(), prev_angle.sin());
        let normal_left = start_tangent.left90();
        let end_normal_left = start_tangent.left90();
        let mut tangent_angle: Option<(f64, f64, Vec2<f64>)> = None;
        for start_side in [-1., 1.] {
            let normal = normal_left * start_side;
            let start_angle = (-normal.y).atan2(-normal.x);
            let arc_center = normal * MIN_RADIUS + prev_pos;
            let arc_dest = pos - arc_center;
            let arc_dest_len = arc_dest.length();
            if arc_dest_len < MIN_RADIUS {
                continue;
            }
            let beta = (MIN_RADIUS / arc_dest_len).acos();
            let arc_dest_angle = arc_dest.y.atan2(arc_dest.x);
            let angle = arc_dest_angle - start_side * beta;
            let start_end_angle = start_angle
                + wrap_angle_offset(
                    angle - start_angle,
                    (1. - start_side) * std::f64::consts::PI,
                );
            for end_side in [-1., 1.] {
                let end_normal = end_normal_left * end_side;
                if !tangent_angle.is_some_and(|acc| {
                    (acc.1 - acc.0).abs() < (start_angle - start_end_angle).abs()
                }) {
                    tangent_angle = Some((end_angle, start_angle, arc_center));
                }
            }
        }
        if let Some((end_angle, start_angle, a)) = tangent_angle {
            let tangent_pos = a + Vec2::new(end_angle.cos(), end_angle.sin()) * MIN_RADIUS;
            let path_segments = [
                PathSegment::Arc(CircleArc::new(a, MIN_RADIUS, start_angle, end_angle)),
                PathSegment::Line([tangent_pos, pos]),
            ];
            Ok(PathBundle::multi(
                path_segments,
                NodeConnection::default(),
                NodeConnection::default(),
            ))
        } else {
            Err("Clicked point requires tighter curvature radius than allowed".to_string())
        }
    }
}
