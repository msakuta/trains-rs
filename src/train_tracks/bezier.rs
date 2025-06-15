use crate::{app::HeightMap, train::Train, vec2::Vec2};

use super::{PathBundle, PathSegment, SegmentDirection, SelectedNode, TrainTracks};

impl TrainTracks {
    pub fn ghost_bezier(&mut self, pos: Vec2<f64>) {
        self.ghost_path = self.compute_bezier(pos).ok().map(|(path, _)| path);
    }

    pub fn add_bezier(
        &mut self,
        pos: Vec2<f64>,
        heightmap: &HeightMap,
        train: &mut Train,
    ) -> Result<(), String> {
        let (path_segments, node_id) = self.compute_bezier(pos)?;

        if path_segments.track.iter().any(|p| heightmap.is_water(p)) {
            return Err("Cannot build tracks through water".to_string());
        }

        self.add_segment(path_segments, node_id, train)
    }

    /// Returns a pair of PathBundle and optional end node id if it closes a loop.
    fn compute_bezier(
        &self,
        mut pos: Vec2<f64>,
    ) -> Result<(PathBundle, Option<SelectedNode>), String> {
        let Some((prev_pos, prev_angle)) = self.selected_node() else {
            return Err("Select a node first".to_string());
        };

        if let Some((node_id, node, next_angle)) = self
            .nodes
            .iter()
            .find(|(_, node)| (node.pos - pos).length2() < (10f64).powi(2))
            .and_then(|(node_id, node)| Some((*node_id, node, self.node_angle(node)?)))
        {
            println!(
                "connecting {pos:?} -> {:?}, {}",
                node.pos,
                next_angle * 180. / std::f64::consts::PI
            );
            let delta = pos - node.pos;
            let end_tangent = Vec2::new(next_angle.cos(), next_angle.sin());
            let dot = delta.dot(end_tangent);
            let (direction, sign) = if dot < 0. {
                (SegmentDirection::Forward, -1.)
            } else {
                (SegmentDirection::Backward, 1.)
            };

            pos = node.pos;

            let start_tangent = Vec2::new(prev_angle.cos(), prev_angle.sin());
            let p1 = prev_pos + start_tangent * 50.;
            let p2 = pos + end_tangent * sign * 50.;
            let path = PathBundle::single(
                PathSegment::CubicBezier([prev_pos, p1, p2, pos]),
                0,
                node_id,
            );
            Ok((path, Some(SelectedNode { node_id, direction })))
        } else {
            let tangent = Vec2::new(prev_angle.cos(), prev_angle.sin());
            let p1 = prev_pos + tangent * 50.;
            let path = PathBundle::single(PathSegment::Bezier([prev_pos, p1, pos]), 0, 0);
            Ok((path, None))
        }
    }
}
