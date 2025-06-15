//! Methods related to gentle paths

use crate::{app::HeightMap, path_utils::wrap_angle, train::Train};

use super::{CircleArc, MAX_RADIUS, MIN_RADIUS, PathBundle, PathSegment, TrainTracks, Vec2};

impl TrainTracks {
    pub fn add_gentle(
        &mut self,
        pos: Vec2<f64>,
        heightmap: &HeightMap,
        train: &mut Train,
    ) -> Result<(), String> {
        match self.compute_gentle(pos) {
            Ok(path_segments) => {
                if path_segments.track.iter().any(|p| heightmap.is_water(p)) {
                    return Err("Cannot build tracks through water".to_string());
                }
                self.add_segment(path_segments, None, train)?;
            }
            Err(e) => {
                self.ghost_path = None;
                return Err(e);
            }
        }
        Ok(())
    }

    pub fn ghost_gentle(&mut self, pos: Vec2<f64>) {
        self.ghost_path = self.compute_gentle(pos).ok();
    }

    fn compute_gentle(&self, pos: Vec2<f64>) -> Result<PathBundle, String> {
        let Some((prev_pos, prev_angle)) = self.selected_node() else {
            return Err("Select a node first".to_string());
        };

        let delta = pos - prev_pos;
        let normal = Vec2::new(-prev_angle.sin(), prev_angle.cos());
        let angle = delta.y.atan2(delta.x);
        let phi = wrap_angle(angle - prev_angle);
        let radius = delta.length() / 2. / phi.sin();
        if radius.abs() < MIN_RADIUS {
            return Err("Clicked point requires tighter curvature radius than allowed".to_string());
        }
        if MAX_RADIUS < radius.abs() {
            return Err("Clicked point requires too large radius".to_string());
        }
        let start = wrap_angle(prev_angle - radius.signum() * std::f64::consts::PI * 0.5);
        let end = start + phi * 2.;
        let path_segment = PathSegment::Arc(CircleArc::new(
            prev_pos + normal * radius,
            radius.abs(),
            start,
            end,
        ));
        Ok(PathBundle::single(path_segment, 0, 0))
    }
}
