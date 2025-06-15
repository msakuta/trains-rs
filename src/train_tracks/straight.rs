//! Methods for straight path logic. It is almost trivial, but is separated for consistency.

use crate::{app::HeightMap, path_utils::PathSegment, train::Train, vec2::Vec2};

use super::{PathBundle, TrainTracks};

impl TrainTracks {
    pub fn add_straight(
        &mut self,
        pos: Vec2<f64>,
        heightmap: &HeightMap,
        train: &mut Train,
    ) -> Result<(), String> {
        let path_segments = self.compute_straight(pos)?;

        if path_segments.track.iter().any(|p| heightmap.is_water(p)) {
            return Err("Cannot build tracks through water".to_string());
        }

        self.add_segment(path_segments, None, train)
    }

    pub fn ghost_straight(&mut self, pos: Vec2<f64>) {
        self.ghost_path = self.compute_straight(pos).ok();
    }

    fn compute_straight(&self, pos: Vec2<f64>) -> Result<PathBundle, String> {
        let Some((prev_pos, prev_angle)) = self.selected_node() else {
            return Err("Select a node first".to_string());
        };

        let delta = pos - prev_pos;
        let tangent = Vec2::new(prev_angle.cos(), prev_angle.sin());
        let dot = tangent.dot(delta);
        if dot < 0. {
            return Err(
                "Straight line cannot connect behind the current track direction".to_string(),
            );
        }
        let perpendicular_foot = prev_pos + tangent * dot;
        let path_segment = PathSegment::Line([prev_pos, perpendicular_foot]);
        Ok(PathBundle::single(path_segment, 0, 0))
    }
}
