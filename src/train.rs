mod path_bundle;

use std::{
    cell::RefCell,
    collections::HashMap,
    rc::{Rc, Weak},
};

use crate::{
    app::HeightMap,
    path_utils::{
        CircleArc, PathSegment, interpolate_path, interpolate_path_heading,
        interpolate_path_tangent, wrap_angle, wrap_angle_offset,
    },
    vec2::Vec2,
};

pub(crate) use path_bundle::PathBundle;
use path_bundle::{ConnectPoint, PathConnection};

use serde::{Deserialize, Serialize};

const CAR_LENGTH: f64 = 1.;
const TRAIN_ACCEL: f64 = 0.001;
const MAX_SPEED: f64 = 1.;
const THRUST_ACCEL: f64 = 0.001;
const GRAD_ACCEL: f64 = 0.0002;
const MIN_RADIUS: f64 = 50.;
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
    forward_paths: Vec<PathConnection>,
    backward_paths: Vec<PathConnection>,
}

impl TrainNode {
    fn new(pos: Vec2<f64>) -> Self {
        Self {
            pos,
            forward_paths: vec![],
            backward_paths: vec![],
        }
    }
}

/// Represents the whole train track network. Can be serialized to a save file.
#[derive(Serialize, Deserialize)]
pub(crate) struct TrainTracks {
    // pub control_points: Vec<Vec2<f64>>,
    /// A collection of paths. A path is an edge as in graph theory.
    /// It could be a vec of `Rc`s, but we want serializable data structure.
    /// We may replace it with a generational id arena to keep the ordering stable.
    pub paths: HashMap<usize, PathBundle>,
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
    /// The index of the path_bundle that the train is on
    pub path_id: usize,
    /// Position along the track
    pub s: f64,
    /// Speed along s
    pub speed: f64,
    pub num_cars: usize,
    #[serde(skip)]
    pub stations: Vec<Rc<RefCell<Station>>>,
    #[serde(skip)]
    pub train_task: TrainTask,
    #[serde(skip)]
    pub schedule: Vec<Weak<RefCell<Station>>>,
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
            s: 0.,
            speed: 0.,
            num_cars: 3,
            path_id: 0,
            stations: [Station::new("Start", 0, 10.), Station::new("Goal", 0, 70.)]
                .into_iter()
                .map(|s| Rc::new(RefCell::new(s)))
                .collect(),
            train_task: TrainTask::Idle,
            schedule: vec![],
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

    pub fn train_pos(&self, car_idx: usize) -> Option<Vec2<f64>> {
        interpolate_path(
            &self.paths[&self.path_id].track,
            self.s - car_idx as f64 * CAR_LENGTH,
        )
    }

    pub fn heading(&self, car_idx: usize) -> Option<f64> {
        interpolate_path_heading(
            &self.paths[&self.path_id].track,
            self.s - car_idx as f64 * CAR_LENGTH,
        )
    }

    pub fn tangent(&self, car_idx: usize) -> Option<Vec2<f64>> {
        interpolate_path_tangent(
            &self.paths[&self.path_id].track,
            self.s - car_idx as f64 * CAR_LENGTH,
        )
    }

    pub fn update(&mut self, thrust: f64, heightmap: &HeightMap) {
        if let TrainTask::Wait(timer) = &mut self.train_task {
            *timer -= 1;
            if *timer <= 1 {
                self.train_task = TrainTask::Idle;
            }
        }
        if matches!(self.train_task, TrainTask::Idle) {
            if let Some(first) = self.schedule.first().cloned() {
                self.train_task = TrainTask::Goto(first.clone());
                self.schedule.remove(0);
                self.schedule.push(first);
            }
        }
        if let TrainTask::Goto(target) = &self.train_task {
            if let Some(target) = target.upgrade() {
                let target = target.borrow();
                let target_s = target.s;
                if (target_s - self.s).abs() < 1. && self.speed.abs() < TRAIN_ACCEL {
                    self.speed = 0.;
                    self.train_task = TrainTask::Wait(120);
                } else if target_s < self.s {
                    // speed / accel == t
                    // speed * t / 2 == speed^2 / accel / 2 == dist
                    // accel = sqrt(2 * dist)
                    if self.speed < 0. && self.s - target_s < 0.5 * self.speed.powi(2) / TRAIN_ACCEL
                    {
                        self.speed += TRAIN_ACCEL;
                    } else {
                        self.speed -= TRAIN_ACCEL;
                    }
                } else {
                    if 0. < self.speed && target_s - self.s < 0.5 * self.speed.powi(2) / TRAIN_ACCEL
                    {
                        self.speed -= TRAIN_ACCEL;
                    } else {
                        self.speed += TRAIN_ACCEL;
                    }
                }
            } else {
                // The station that cease to exist should be rotated to the end of the queue.
                self.schedule.pop();
                self.train_task = TrainTask::Idle;
            }
        }
        self.speed = (self.speed + thrust * THRUST_ACCEL).clamp(-MAX_SPEED, MAX_SPEED);

        // Acceleration from terrain slope
        for i in 0..self.num_cars {
            if let Some((tangent, pos)) = self.tangent(i).zip(self.train_pos(i)) {
                let grad = heightmap.gradient(&pos);
                self.speed -= grad.dot(tangent) * GRAD_ACCEL / self.num_cars as f64;
            }
        }

        if let Some(path) = self.paths.get(&self.path_id) {
            if self.s == 0. && self.speed < 0. {
                if let Some(prev_path) = self.nodes.get(&path.start_node).and_then(|node| {
                    let forward = node.forward_paths.iter().any(|p| p.path_id == self.path_id);
                    // If we came from forward, we should continue on backward
                    if forward {
                        node.backward_paths.first()
                    } else {
                        node.forward_paths.first()
                    }
                }) {
                    match prev_path.connect_point {
                        ConnectPoint::Start => {
                            self.path_id = prev_path.path_id;
                            self.s = 0.;
                            self.speed *= -1.;
                        }
                        ConnectPoint::End => {
                            self.path_id = prev_path.path_id;
                            self.s = prev_path_ref.track.len() as f64;
                        }
                    }
                } else {
                    self.speed = 0.;
                }
            }

            if path.track.len() as f64 <= self.s && 0. < self.speed {
                if let Some(next_path) = self.nodes.get(&path.end_node).and_then(|node| {
                    let forward = node.forward_paths.iter().any(|p| p.path_id == self.path_id);
                    // If we came from forward, we should continue on backward
                    if forward {
                        node.backward_paths.first()
                    } else {
                        node.forward_paths.first()
                    }
                }) {
                    match next_path.connect_point {
                        ConnectPoint::Start => {
                            self.path_id = next_path.path_id;
                            self.s = 0.;
                        }
                        ConnectPoint::End => {
                            self.path_id = next_path.path_id;
                            self.s = path.track.len() as f64;
                            self.speed *= -1.;
                        }
                    }
                } else {
                    self.speed = 0.;
                }
            }

            // Acquire path again because it may have changed
            let path = &self.paths[&self.path_id];
            self.s = (self.s + self.speed).clamp(0., path.track.len() as f64);
        }
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

    pub fn add_gentle(&mut self, pos: Vec2<f64>, heightmap: &HeightMap) -> Result<(), String> {
        match self.compute_gentle(pos) {
            Ok(path_segments) => {
                if path_segments.track.iter().any(|p| heightmap.is_water(p)) {
                    return Err("Cannot build tracks through water".to_string());
                }
                self.add_segment(path_segments)?;
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
    fn add_segment(&mut self, mut path_bundle: PathBundle) -> Result<(), String> {
        let Some(selected) = self.selected_node else {
            return Err("Select a node first".to_string());
        };
        let Some(path) = self.paths.get_mut(&selected.path_id) else {
            return Err("Path not found; perhaps deleted".to_string());
        };
        // If it was the first pathnode, just prepend a segment
        if selected.pathnode_id == 0 {
            let len = path.append(path_bundle.segments);
            self.offset_path(selected.path_id, len as f64);
        } else
        // If it was the last segment, just extend it
        if selected.pathnode_id == path.segments.len() {
            path.extend(&path_bundle.segments);
            // Continue extending from the added segment
            self.selected_node = Some(SelectedPathNode::new(
                selected.path_id,
                selected.pathnode_id + path_bundle.segments.len(),
                selected.direction,
            ));
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
            let mut split_path = PathBundle::multi(
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
            if self.path_id == selected.path_id && new_path_len < self.s {
                self.path_id = split_path_id;
                self.s -= new_path_len;
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
        }
        Ok(())
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

        if self.path_id == path_id {
            self.s += s;
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

    pub fn add_straight(&mut self, pos: Vec2<f64>, heightmap: &HeightMap) -> Result<(), String> {
        let path_segments = self.compute_straight(pos)?;

        if path_segments.track.iter().any(|p| heightmap.is_water(p)) {
            return Err("Cannot build tracks through water".to_string());
        }

        self.add_segment(path_segments)
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

    pub fn add_tight(&mut self, pos: Vec2<f64>, heightmap: &HeightMap) -> Result<(), String> {
        let path_segments = self.compute_tight(pos)?;

        if path_segments.track.iter().any(|p| heightmap.is_water(p)) {
            return Err("Cannot build tracks through water".to_string());
        }

        self.add_segment(path_segments)
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

    fn compute_tight_orient(&self, pos: Vec2<f64>, end_angle: f64) -> Result<PathBundle, String> {
        let Some(prev) = self.paths[&self.path_id].segments.last() else {
            return Err("Path needs at least one segment to connect to".to_string());
        };
        let prev_pos = prev.end();
        let prev_angle = prev.end_angle();
        let start_tangent = Vec2::new(prev_angle.cos(), prev_angle.sin());
        let normal_left = start_tangent.left90();
        let end_tangent = Vec2::new(end_angle.cos(), end_angle.sin());
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

    pub fn delete_segment(&mut self, pos: Vec2<f64>, dist_thresh: f64) -> Result<(), String> {
        let found_node = self.paths.iter_mut().find_map(|(id, path)| {
            let (seg, _) = path.find_node(pos, dist_thresh)?;
            Some((id, path, seg))
        });
        if let Some((&path_id, path, seg)) = found_node {
            if path_id == self.path_id && seg == path.find_seg_by_s(self.s as usize) {
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

                move_s(&mut self.path_id, &mut self.s, "train");
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

#[derive(Clone, Copy, Serialize, Deserialize)]
pub(crate) enum SegmentDirection {
    /// Follow the direction of the increasing s
    Forward,
    /// Reverse of forward
    Backward,
}
