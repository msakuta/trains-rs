mod path_bundle;

use std::{
    cell::RefCell,
    collections::HashMap,
    rc::{Rc, Weak},
};

use crate::{
    app::HeightMap,
    path_utils::{CircleArc, PathSegment, interpolate_path, wrap_angle, wrap_angle_offset},
    train::Train,
    vec2::Vec2,
};

pub(crate) use self::path_bundle::{ConnectPoint, PathBundle, PathConnection};

use serde::{Deserialize, Serialize};

const MIN_RADIUS: f64 = 50.;
const MAX_RADIUS: f64 = 10000.;
const SEGMENT_LENGTH: f64 = 10.;
pub(crate) const _C_POINTS: [Vec2<f64>; 11] = [
    Vec2::new(0., 0.),
    Vec2::new(50., 0.),
    Vec2::new(100., 0.),
    Vec2::new(200., 0.),
    Vec2::new(300., 100.),
    Vec2::new(400., 200.),
    Vec2::new(500., 200.),
    Vec2::new(550., 200.),
    Vec2::new(600., 200.),
    Vec2::new(700., 200.),
    Vec2::new(700., 100.),
];

pub(crate) const PATH_SEGMENTS: [PathSegment; 5] = [
    PathSegment::Line([Vec2::new(50., 50.), Vec2::new(150., 50.)]),
    PathSegment::Arc(CircleArc::new(
        Vec2::new(150., 150.),
        100.,
        std::f64::consts::PI * 1.5,
        std::f64::consts::PI * 2.,
    )),
    PathSegment::Line([Vec2::new(250., 150.), Vec2::new(250., 250.)]),
    PathSegment::Arc(CircleArc::new(
        Vec2::new(150., 250.),
        100.,
        std::f64::consts::PI * 0.,
        std::f64::consts::PI * 0.5,
    )),
    PathSegment::Line([Vec2::new(150., 350.), Vec2::new(50., 350.)]),
];

pub(crate) struct Station {
    pub name: String,
    pub path_id: usize,
    pub s: f64,
}

impl Station {
    pub fn new(name: impl Into<String>, path_id: usize, s: f64) -> Self {
        Self {
            name: name.into(),
            path_id,
            s,
        }
    }
}

#[derive(Clone, Default)]
pub(crate) enum TrainTask {
    #[default]
    Idle,
    Goto(Weak<RefCell<Station>>),
    Wait(usize),
}

#[derive(Serialize, Deserialize)]
pub(crate) struct TrainNode {
    pos: Vec2<f64>,
    pub(crate) forward_paths: Vec<PathConnection>,
    pub(crate) backward_paths: Vec<PathConnection>,
}

impl TrainNode {
    fn new(pos: Vec2<f64>) -> Self {
        Self {
            pos,
            forward_paths: vec![],
            backward_paths: vec![],
        }
    }

    fn is_connected_to(&self, path_id: usize) -> bool {
        self.forward_paths.iter().any(|p| p.path_id == path_id)
            || self.backward_paths.iter().any(|p| p.path_id == path_id)
    }

    fn count_connections(&self) -> usize {
        self.forward_paths.len() + self.backward_paths.len()
    }
}

pub(crate) type Paths = HashMap<usize, PathBundle>;

/// Represents the whole train track network. Can be serialized to a save file.
#[derive(Serialize, Deserialize)]
pub(crate) struct TrainTracks {
    // pub control_points: Vec<Vec2<f64>>,
    /// A collection of paths. A path is an edge as in graph theory.
    /// It could be a vec of `Rc`s, but we want serializable data structure.
    /// We may replace it with a generational id arena to keep the ordering stable.
    pub paths: Paths,
    /// The next id of the path
    pub path_id_gen: usize,
    /// A collection of train track nodes, which connects paths. A node as in graph theory.
    /// Used for connectivity calculations.
    pub nodes: HashMap<usize, TrainNode>,
    /// The next id of the node
    pub node_id_gen: usize,
    /// Selected segment node to add a segment to. Skipped from serde since it is not a game state par se.
    #[serde(skip)]
    pub selected_node: Option<SelectedPathNode>,
    /// Build ghost segment, which is not actually built yet
    pub ghost_path: Option<PathBundle>,
    #[serde(skip)]
    pub stations: Vec<Rc<RefCell<Station>>>,
}

impl TrainTracks {
    pub fn new() -> Self {
        let mut paths = HashMap::new();
        paths.insert(0, PathBundle::multi(PATH_SEGMENTS.to_vec(), 0, 1));
        let mut nodes = HashMap::new();
        let mut first_node = TrainNode::new(PATH_SEGMENTS.first().unwrap().start());
        first_node
            .forward_paths
            .push(PathConnection::new(0, ConnectPoint::Start));
        nodes.insert(0, first_node);
        let mut last_node = TrainNode::new(PATH_SEGMENTS.last().unwrap().end());
        last_node
            .backward_paths
            .push(PathConnection::new(0, ConnectPoint::End));
        nodes.insert(1, last_node);
        Self {
            // control_points: C_POINTS.to_vec(),
            paths,
            path_id_gen: 1,
            selected_node: None,
            nodes,
            node_id_gen: 2,
            ghost_path: None,
            stations: [Station::new("Start", 0, 10.), Station::new("Goal", 0, 70.)]
                .into_iter()
                .map(|s| Rc::new(RefCell::new(s)))
                .collect(),
        }
    }

    pub fn control_points(&self) -> Vec<Vec2<f64>> {
        self.paths
            .values()
            .map(|b| b.segments.iter())
            .flatten()
            .map(|seg| seg.end())
            .collect()
    }

    pub fn s_pos(&self, path_id: usize, s: f64) -> Option<Vec2<f64>> {
        interpolate_path(&self.paths.get(&path_id)?.track, s)
    }

    pub fn add_station(&mut self, name: impl Into<String>, pos: Vec2<f64>, thresh: f64) {
        let Some((path_id, _seg_id, node_id)) = self.find_path_node(pos, thresh) else {
            return;
        };
        self.stations.push(Rc::new(RefCell::new(Station::new(
            name,
            path_id,
            node_id as f64,
        ))));
    }

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
                self.add_segment(path_segments, train)?;
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

    /// Attempt to add a segment from the selected node. If it was at the end of a path,
    /// extend it, otherwise split the path in the middle and attach a new path.
    /// This way we maintain that any path does not have a branch.
    fn add_segment(
        &mut self,
        mut path_bundle: PathBundle,
        train: &mut Train,
    ) -> Result<(), String> {
        let Some(selected) = self.selected_node else {
            return Err("Select a node first".to_string());
        };
        let Some(path) = self.paths.get_mut(&selected.path_id) else {
            return Err("Path not found; perhaps deleted".to_string());
        };
        // If it was the first pathnode...
        if selected.pathnode_id == 0 {
            let start_node = self.nodes.get_mut(&path.start_node).unwrap();
            // ... and it does not have any other path, just prepend a segment, ...
            if start_node.count_connections() < 2 {
                let len = path.append(path_bundle.segments);
                self.offset_path(selected.path_id, len as f64);
            } else {
                // ... unless there are already connected paths in which case we can't just extend.
                // Allocate path ids for the new paths
                let new_path_id = self.path_id_gen;
                self.path_id_gen += 1;
                let new_end_node_id = self.node_id_gen;
                self.node_id_gen += 1;

                start_node
                    .backward_paths
                    .push(PathConnection::new(new_path_id, ConnectPoint::Start));

                // Set up the new end node
                let mut new_end_node = TrainNode::new(path_bundle.end());

                // Then, add a new node for the end of the new path.
                new_end_node
                    .backward_paths
                    .push(PathConnection::new(new_path_id, ConnectPoint::End));
                self.nodes.insert(new_end_node_id, new_end_node);
                path_bundle.start_node = path.start_node;
                path_bundle.end_node = new_end_node_id;

                // Lastly, add the new path for the new segment.
                self.paths.insert(new_path_id, path_bundle);
            }
        } else if selected.pathnode_id == path.segments.len() {
            // If it was the last segment ...

            let end_node = self.nodes.get_mut(&path.end_node).unwrap();
            // ... and it does not have any other path, just extend it ...
            if end_node.count_connections() < 2 {
                path.extend(&path_bundle.segments);
                // Continue extending from the added segment
                self.selected_node = Some(SelectedPathNode::new(
                    selected.path_id,
                    selected.pathnode_id + path_bundle.segments.len(),
                    selected.direction,
                ));
            } else {
                // ... unless there are already connected paths in which case we can't just extend.

                // Allocate path ids for the new paths
                let new_path_id = self.path_id_gen;
                self.path_id_gen += 1;
                let new_end_node_id = self.node_id_gen;
                self.node_id_gen += 1;

                end_node
                    .forward_paths
                    .push(PathConnection::new(new_path_id, ConnectPoint::Start));

                // Set up the new start node
                let mut new_end_node = TrainNode::new(path_bundle.end());

                // Then, add a new node for the end of the new path.
                new_end_node
                    .backward_paths
                    .push(PathConnection::new(new_path_id, ConnectPoint::End));
                self.nodes.insert(new_end_node_id, new_end_node);
                path_bundle.start_node = path.end_node;
                path_bundle.end_node = new_end_node_id;

                // Lastly, add the new path for the new segment.
                self.paths.insert(new_path_id, path_bundle);
            }
        } else {
            // Othewise, split the path at the node and add a new path starting from the selected pathnode,
            // whose sole member is the new segments.
            // It can be quite complicated, so see the figures below to understand the desired before and after state
            // along with the variable names.
            //
            // Before:
            //
            //          * end_node
            //           \
            //            \
            //             \
            //              | selected.path_id
            //              |
            //              |
            //              * start_node
            //
            // After:
            //
            // end_node *       * new_end_node
            //           \     /
            // split_path \   / new_path
            //             \ /
            //              * split_node
            //              |
            //              | selected.path_id
            //              |
            //              * start_node
            //

            // Allocate path ids for the new paths
            let split_path_id = self.path_id_gen;
            self.path_id_gen += 1;
            let new_path_id = self.path_id_gen;
            self.path_id_gen += 1;
            let split_node_id = self.node_id_gen;
            self.node_id_gen += 1;
            let new_end_node_id = self.node_id_gen;
            self.node_id_gen += 1;

            // First, create a path for the segments after the selected node.
            let split_path = PathBundle::multi(
                path.segments[selected.pathnode_id..]
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
                split_node_id,
                path.end_node,
            );

            // Set up the split node. Note that the new path can have different direction depending on
            // selected.direction.
            let mut split_node = TrainNode::new(split_path.start());
            split_node
                .forward_paths
                .push(PathConnection::new(split_path_id, ConnectPoint::Start));
            split_node
                .backward_paths
                .push(PathConnection::new(selected.path_id, ConnectPoint::End));
            match selected.direction {
                SegmentDirection::Forward => {
                    split_node
                        .forward_paths
                        .push(PathConnection::new(new_path_id, ConnectPoint::Start));
                }
                SegmentDirection::Backward => {
                    split_node
                        .backward_paths
                        .push(PathConnection::new(new_path_id, ConnectPoint::Start));
                }
            }
            self.nodes.insert(split_node_id, split_node);

            // Next, truncate the selected path after the selected node.
            path.truncate(selected.pathnode_id);
            if let Some(node) = self.nodes.get_mut(&path.end_node) {
                node.forward_paths.retain(|p| p.path_id != selected.path_id);
                node.backward_paths
                    .retain(|p| p.path_id != selected.path_id);
                node.backward_paths
                    .push(PathConnection::new(split_path_id, ConnectPoint::End));
            }
            path.end_node = split_node_id;

            // Move the stations after the split point to the split path and subtract the first half path
            let new_path_len = path.track.len() as f64;
            for station in &self.stations {
                let mut station = station.borrow_mut();
                if station.path_id == selected.path_id && new_path_len < station.s {
                    station.path_id = split_path_id;
                    station.s -= new_path_len;
                }
            }
            if train.path_id == selected.path_id && new_path_len < train.s {
                train.path_id = split_path_id;
                train.s -= new_path_len;
            }

            // Add the split path after the selected path is modified, in order to avoid the borrow checker.
            self.paths.insert(split_path_id, split_path);

            // Set up the new end node
            let mut new_end_node = TrainNode::new(path_bundle.end());

            // Then, add a new node for the end of the new path.
            new_end_node
                .backward_paths
                .push(PathConnection::new(new_path_id, ConnectPoint::End));
            self.nodes.insert(new_end_node_id, new_end_node);
            path_bundle.start_node = split_node_id;
            path_bundle.end_node = new_end_node_id;
            let next_pathnode = path_bundle.segments.len();

            // Lastly, add the new path for the new segment.
            self.paths.insert(new_path_id, path_bundle);

            // Select the segment just added to allow continuing extending
            self.selected_node = Some(SelectedPathNode::new(
                new_path_id,
                next_pathnode,
                SegmentDirection::Forward,
            ));

            // Sort the connection list after new connections are added.
            // Note that new_end_node does not have to sort because there is only 1 connection.
            self.sort_node_connections(split_node_id);
        }
        Ok(())
    }

    fn sort_node_connections(&mut self, node_id: usize) {
        let Some(node) = self.nodes.get_mut(&node_id) else {
            return;
        };

        let key = |con: &PathConnection| {
            self.paths
                .get(&con.path_id)
                .map(|p| match con.connect_point {
                    ConnectPoint::Start => {
                        p.track
                            .get(p.track.len().saturating_sub(5))
                            .map_or(0., |offset_pos| {
                                let delta = *offset_pos - node.pos;
                                delta.y.atan2(delta.x)
                            })
                    }
                    ConnectPoint::End => p.track.get(5).map_or(0., |offset_pos| {
                        let delta = *offset_pos - node.pos;
                        delta.y.atan2(delta.x)
                    }),
                })
        };

        node.forward_paths.sort_by(|lhs, rhs| {
            let lhs = key(lhs).unwrap_or(0.);
            let rhs = key(rhs).unwrap_or(0.);
            // Ignore nans for now
            lhs.partial_cmp(&rhs).unwrap()
        });

        for (i, con) in node.forward_paths.iter().enumerate() {
            println!("forward con {i} {:?}", key(con));
        }

        node.backward_paths.sort_by(|lhs, rhs| {
            let lhs = key(lhs).unwrap_or(0.);
            let rhs = key(rhs).unwrap_or(0.);
            // Ignore nans for now
            lhs.partial_cmp(&rhs).unwrap()
        });

        for (i, con) in node.backward_paths.iter().enumerate() {
            println!("backward con {i} {:?}", key(con));
        }
    }

    fn offset_path(&mut self, path_id: usize, s: f64) {
        for station in &mut self.stations {
            let mut station = station.borrow_mut();
            if station.path_id == path_id {
                println!(
                    "Adding offset {s} to station {} at {}",
                    station.name, station.s
                );
                station.s += s;
            }
        }
    }

    pub fn has_selected_node(&self) -> bool {
        self.selected_node.is_some()
    }

    /// Returns the position and the angle of the selected node
    pub fn selected_node(&self) -> Option<(Vec2<f64>, f64)> {
        self.selected_node
            .and_then(|selected| self.node_position(selected))
    }

    /// Returns the position and the angle of the given node
    pub fn node_position(&self, selected: SelectedPathNode) -> Option<(Vec2<f64>, f64)> {
        let Some(path) = self.paths.get(&selected.path_id) else {
            return None;
        };
        match selected.direction {
            SegmentDirection::Forward => {
                if selected.pathnode_id == 0 {
                    let Some(seg) = path.segments.first() else {
                        return None;
                    };
                    return Some((
                        seg.start(),
                        wrap_angle(seg.start_angle() + std::f64::consts::PI),
                    ));
                }
                let Some(seg) = path.segments.get(selected.pathnode_id - 1) else {
                    return None;
                };
                return Some((seg.end(), seg.end_angle()));
            }
            SegmentDirection::Backward => {
                let Some(seg) = path.segments.get(selected.pathnode_id) else {
                    return None;
                };
                return Some((seg.start(), seg.start_angle()));
            }
        }
    }

    pub fn select_node(&mut self, pos: Vec2<f64>, thresh: f64) -> Option<SelectedPathNode> {
        let found_node = self.find_segment_node(pos, thresh);
        self.selected_node = found_node;
        found_node
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

        self.add_segment(path_segments, train)
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

        self.add_segment(path_segments, train)
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
            Ok(PathBundle::multi(path_segments, 0, 0))
        } else {
            Err("Clicked point requires tighter curvature radius than allowed".to_string())
        }
    }

    pub fn ghost_bezier(&mut self, pos: Vec2<f64>) {
        self.ghost_path = self.compute_bezier(pos).ok();
    }

    pub fn add_bezier(
        &mut self,
        pos: Vec2<f64>,
        heightmap: &HeightMap,
        train: &mut Train,
    ) -> Result<(), String> {
        let path_segments = self.compute_bezier(pos)?;

        if path_segments.track.iter().any(|p| heightmap.is_water(p)) {
            return Err("Cannot build tracks through water".to_string());
        }

        self.add_segment(path_segments, train)
    }

    fn compute_bezier(&self, mut pos: Vec2<f64>) -> Result<PathBundle, String> {
        let Some((prev_pos, prev_angle)) = self.selected_node() else {
            return Err("Select a node first".to_string());
        };

        if let Some((node, next_angle)) = self
            .nodes
            .iter()
            .find(|(_, node)| (node.pos - pos).length2() < (10f64).powi(2))
            .and_then(|(_, node)| Some((node, self.node_angle(node)?)))
        {
            println!(
                "connecting {pos:?} -> {:?}, {}",
                node.pos,
                next_angle * 180. / std::f64::consts::PI
            );
            pos = node.pos;

            let start_tangent = Vec2::new(prev_angle.cos(), prev_angle.sin());
            let end_tangent = Vec2::new(next_angle.cos(), next_angle.sin());
            let p1 = prev_pos + start_tangent * 50.;
            let p2 = prev_pos - end_tangent * 50.;
            let path = PathBundle::single(PathSegment::CubicBezier([prev_pos, p1, p2, pos]), 0, 0);
            Ok(path)
        } else {
            let tangent = Vec2::new(prev_angle.cos(), prev_angle.sin());
            let p1 = prev_pos + tangent * 50.;
            let path = PathBundle::single(PathSegment::Bezier([prev_pos, p1, pos]), 0, 0);
            Ok(path)
        }
    }

    /// Forward angle of a node in radians. It is not directly stored in the node, so we need to look up a path that is
    /// connected to it.
    /// A node should always have a connected path, but the data structure allows a node without connected paths, so
    /// this function can return None in that case.
    fn node_angle(&self, node: &TrainNode) -> Option<f64> {
        let path_angle = |con: &PathConnection, path: &PathBundle| match con.connect_point {
            ConnectPoint::Start => return Some(path.segments.first()?.start_angle()),
            ConnectPoint::End => return Some(path.segments.last()?.end_angle()),
        };
        if let Some(con) = node.forward_paths.first() {
            if let Some(path) = self.paths.get(&con.path_id) {
                return path_angle(con, path);
            }
        } else if let Some(con) = node.backward_paths.first() {
            if let Some(path) = self.paths.get(&con.path_id) {
                return path_angle(con, path);
            }
        }
        None
    }

    #[allow(dead_code)]
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
            Ok(PathBundle::multi(path_segments, 0, 0))
        } else {
            Err("Clicked point requires tighter curvature radius than allowed".to_string())
        }
    }

    /// Returns a tuple of (path id, segment id, node id)
    pub fn find_path_node(&self, pos: Vec2<f64>, thresh: f64) -> Option<(usize, usize, usize)> {
        self.paths.iter().find_map(|(path_id, path)| {
            let (seg_id, node_id) = path.find_node(pos, thresh)?;
            Some((*path_id, seg_id, node_id))
        })
    }

    /// Finds a segment node and returns its id. A segment node is a point between segments, not within one.
    pub fn find_segment_node(&self, pos: Vec2<f64>, thresh: f64) -> Option<SelectedPathNode> {
        self.paths.iter().find_map(|(path_id, path)| {
            // 0th segment node is the start of the first segment
            if path
                .segments
                .first()
                .is_some_and(|seg| (seg.start() - pos).length2() < thresh.powi(2))
            {
                return Some(SelectedPathNode::new(
                    *path_id,
                    0,
                    SegmentDirection::Backward,
                ));
            }
            path.segments.iter().enumerate().find_map(|(seg_id, seg)| {
                if (seg.end() - pos).length2() < thresh.powi(2) {
                    let angle = seg.end_angle();
                    let delta = seg.end() - pos;
                    if 0. < delta.dot(Vec2::new(angle.cos(), angle.sin())) {
                        return Some(SelectedPathNode::new(
                            *path_id,
                            seg_id + 1,
                            SegmentDirection::Forward,
                        ));
                    }
                }

                if (seg.start() - pos).length2() < thresh.powi(2) {
                    let angle = seg.start_angle();
                    let delta = seg.start() - pos;
                    if 0. < delta.dot(Vec2::new(angle.cos(), angle.sin())) {
                        return Some(SelectedPathNode::new(
                            *path_id,
                            seg_id,
                            SegmentDirection::Backward,
                        ));
                    }
                }

                None
            })
        })
    }

    pub fn delete_segment(
        &mut self,
        pos: Vec2<f64>,
        dist_thresh: f64,
        train: &mut Train,
    ) -> Result<(), String> {
        let found_node = self.paths.iter_mut().find_map(|(id, path)| {
            let (seg, _) = path.find_node(pos, dist_thresh)?;
            Some((id, path, seg))
        });
        if let Some((&path_id, path, seg)) = found_node {
            if path_id == train.path_id && seg == path.find_seg_by_s(train.s as usize) {
                return Err("You can't delete a segment while a train is on it".to_string());
            }
            let delete_begin = if 0 < seg {
                path.track_ranges[seg - 1] as f64
            } else {
                0.
            };
            let delete_end = path.track_ranges[seg] as f64;
            println!("Delete node range: {}, {}", delete_begin, delete_end);
            let new_path = path.delete_segment(seg, |added_node| {
                let node_id = self.node_id_gen;
                self.nodes.insert(node_id, TrainNode::new(added_node));
                self.node_id_gen += 1;
                node_id
            });
            let path_len = path.segments.len();
            if let Some(new_path) = new_path {
                let new_id = self.path_id_gen;
                self.paths.insert(new_id, new_path);
                self.path_id_gen += 1;

                let move_s = |path_id: &mut usize, s: &mut f64, name: &str| {
                    if delete_end <= *s {
                        println!(
                            "Moving {name} path: {}, {}",
                            new_id,
                            (*s - delete_end).max(0.)
                        );
                        *path_id = new_id;
                        *s = (*s - delete_end).max(0.);
                        true
                    } else {
                        *s <= delete_begin
                    }
                };

                move_s(&mut train.path_id, &mut train.s, "train");
                self.stations.retain(|station| {
                    let mut station = station.borrow_mut();
                    let station = &mut *station;
                    if station.path_id != path_id {
                        return true;
                    }
                    let station_name = format!("station {}", station.name);
                    move_s(&mut station.path_id, &mut station.s, &station_name)
                });
            } else {
                self.stations.retain(|station| {
                    let station = station.borrow();
                    if station.path_id != path_id {
                        return true;
                    }
                    station.s <= delete_begin || delete_end <= station.s
                });
            }
            if path_len == 0 {
                println!("Path {path_id} length is 0! deleting path");
                self.paths.remove(&path_id);
            }
        } else {
            return Err("No segment is selected for deleting".to_string());
        }
        Ok(())
    }
}

/// A selected path node with direction. A path node is a point between segments, including both ends of the path.
/// 0th and nth pathnodes are at the ends, where n is the number of segments.
///
/// This type is used to store the position and the direction of a segment to add.
#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) struct SelectedPathNode {
    /// The path id, an index into HashMap
    pub path_id: usize,
    /// The node index between segments. 0th pathnode is the start of the first segment and nth is the end of (n-1)th segment.
    /// Note that if the path has n segments, it has n+1 pathnodes.
    pub pathnode_id: usize,
    /// Direction, used to determine angle of the segment to add.
    pub direction: SegmentDirection,
}

impl SelectedPathNode {
    pub fn new(path_id: usize, pathnode_id: usize, direction: SegmentDirection) -> Self {
        Self {
            path_id,
            pathnode_id,
            direction,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize, Default)]
pub(crate) enum SegmentDirection {
    /// Follow the direction of the increasing s
    #[default]
    Forward,
    /// Reverse of forward
    Backward,
}

impl std::ops::Not for SegmentDirection {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            Self::Forward => Self::Backward,
            Self::Backward => Self::Forward,
        }
    }
}

impl SegmentDirection {
    pub fn signum(&self) -> f64 {
        match self {
            Self::Forward => 1.,
            Self::Backward => -1.,
        }
    }
}
