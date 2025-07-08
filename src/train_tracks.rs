mod bezier;
mod gentle;
mod path_bundle;
mod straight;
mod tight;

use std::collections::HashMap;

use crate::{
    path_utils::{CircleArc, PathSegment, interpolate_path, interpolate_path_tangent, wrap_angle},
    train::{CAR_LENGTH, Train},
    vec2::Vec2,
};

pub(crate) use self::path_bundle::{ConnectPoint, NodeConnection, PathBundle, PathConnection};

use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

pub(crate) const RAIL_HALFWIDTH: f64 = 1.25;
pub(crate) const RAIL_WIDTH: f64 = RAIL_HALFWIDTH * 2.;
const MIN_RADIUS: f64 = 50.;
const MAX_RADIUS: f64 = 10000.;
pub(crate) const SEGMENT_LENGTH: f64 = 10.;

pub(crate) const PATH_SEGMENTS: [PathSegment; 4] = [
    PathSegment::Arc(CircleArc::new(
        Vec2::new(-50., 0.),
        50.,
        std::f64::consts::PI * 0.5,
        std::f64::consts::PI * 1.5,
    )),
    PathSegment::Line([Vec2::new(-50., -50.), Vec2::new(50., -50.)]),
    PathSegment::Arc(CircleArc::new(
        Vec2::new(50., 0.),
        50.,
        std::f64::consts::PI * 1.5,
        std::f64::consts::PI * 2.5,
    )),
    PathSegment::Line([Vec2::new(50., 50.), Vec2::new(-50., 50.)]),
];

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Station {
    pub name: String,
    pub path_id: usize,
    pub s: f64,
    pub ty: StationType,
}

impl Station {
    pub fn new(name: impl Into<String>, path_id: usize, s: f64, ty: StationType) -> Self {
        Self {
            name: name.into(),
            path_id,
            s,
            ty,
        }
    }
}

/// This could be a part of TrainTask instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum StationType {
    Loading,
    Unloading,
}

pub(crate) type StationId = usize;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) enum TrainTask {
    #[default]
    Idle,
    Goto(StationId),
    Wait(usize, StationId),
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

    fn _is_connected_to(&self, path_id: usize) -> bool {
        self.forward_paths.iter().any(|p| p.path_id == path_id)
            || self.backward_paths.iter().any(|p| p.path_id == path_id)
    }

    fn count_connections(&self) -> usize {
        self.forward_paths.len() + self.backward_paths.len()
    }

    pub(crate) fn paths_in_direction(&self, direction: SegmentDirection) -> &[PathConnection] {
        match direction {
            SegmentDirection::Forward => &self.forward_paths,
            SegmentDirection::Backward => &self.backward_paths,
        }
    }

    pub(crate) fn paths_in_direction_mut(
        &mut self,
        direction: SegmentDirection,
    ) -> &mut Vec<PathConnection> {
        match direction {
            SegmentDirection::Forward => &mut self.forward_paths,
            SegmentDirection::Backward => &mut self.backward_paths,
        }
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
    /// The next id of the station
    pub station_id_gen: usize,
    /// A collection of stations in the train track network.
    pub stations: HashMap<usize, Station>,
}

impl TrainTracks {
    pub fn new() -> Self {
        let mut paths = HashMap::new();
        paths.insert(
            0,
            PathBundle::multi(
                PATH_SEGMENTS.to_vec(),
                NodeConnection::new(0, SegmentDirection::Forward),
                NodeConnection::new(0, SegmentDirection::Backward),
            ),
        );
        let mut nodes = HashMap::new();
        let mut first_node = TrainNode::new(PATH_SEGMENTS.first().unwrap().start());
        first_node
            .forward_paths
            .push(PathConnection::new(0, ConnectPoint::Start));
        first_node
            .backward_paths
            .push(PathConnection::new(0, ConnectPoint::End));
        nodes.insert(0, first_node);
        Self {
            paths,
            path_id_gen: 1,
            selected_node: None,
            nodes,
            node_id_gen: 1,
            ghost_path: None,
            station_id_gen: 2,
            stations: [
                Station::new("Start", 0, 10., StationType::Loading),
                Station::new("Goal", 0, 30., StationType::Unloading),
            ]
            .into_iter()
            .enumerate()
            .collect(),
        }
    }

    pub fn s_pos(&self, path_id: usize, s: f64) -> Option<Vec2<f64>> {
        interpolate_path(&self.paths.get(&path_id)?.track, s)
    }

    pub fn add_station(
        &mut self,
        name: impl Into<String>,
        pos: Vec2<f64>,
        thresh: f64,
        ty: StationType,
    ) {
        let Some((path_id, _seg_id, node_id)) = self.find_path_node(pos, thresh) else {
            return;
        };
        self.stations.insert(
            self.station_id_gen,
            Station::new(name, path_id, node_id as f64, ty),
        );
        self.station_id_gen += 1;
    }

    /// Attempt to add a segment from the selected node. The behavior is different in 3 cases:
    ///
    /// * If the node was at either end of a path, and has no other connected paths, extend it.
    /// * If the node was at either end of a path, and has other connected paths, create a new path and connect it to
    ///   the node.
    /// * If the node was in the middle of a path, split the path at that point and attach a new path.
    ///
    /// This way we maintain the state that any path does not have a branch.
    fn add_segment(
        &mut self,
        mut path_bundle: PathBundle,
        end_node_sel: Option<SelectedNode>,
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
            let start_node = self.nodes.get_mut(&path.start_node.node_id).unwrap();
            // ... and it does not have any other path, just prepend a segment, ...
            if start_node.count_connections() < 2 {
                if let Some(end_node_sel) = end_node_sel {
                    path.start_node = end_node_sel;
                    self.nodes
                        .get_mut(&end_node_sel.node_id)
                        .ok_or_else(|| format!("Node {} not found", end_node_sel.node_id))?
                        .paths_in_direction_mut(end_node_sel.direction)
                        .push(PathConnection::new(selected.path_id, ConnectPoint::Start));
                } else {
                    start_node.pos = path_bundle.end();
                }

                let len = path.prepend(path_bundle.segments);
                self.offset_path(selected.path_id, len as f64);
                train.offset(selected.path_id, len as f64);
            } else {
                // ... unless there are already connected paths in which case we can't just extend.
                // Allocate path ids for the new paths
                let new_path_id = self.path_id_gen;
                self.path_id_gen += 1;
                let new_end_node_id = end_node_sel.unwrap_or_else(|| {
                    let id = self.node_id_gen;
                    self.node_id_gen += 1;
                    SelectedNode {
                        node_id: id,
                        direction: SegmentDirection::Backward,
                    }
                });

                start_node
                    .paths_in_direction_mut(!path.start_node.direction)
                    .push(PathConnection::new(new_path_id, ConnectPoint::Start));

                // Set up the new end node
                let new_end_node = self
                    .nodes
                    .entry(new_end_node_id.node_id)
                    .or_insert_with(|| TrainNode::new(path_bundle.end()));

                // Then, add a new node for the end of the new path.
                new_end_node
                    .paths_in_direction_mut(new_end_node_id.direction)
                    .push(PathConnection::new(new_path_id, ConnectPoint::End));
                path_bundle.start_node = path.start_node.reversed();
                path_bundle.end_node = new_end_node_id;

                // Lastly, add the new path for the new segment.
                self.paths.insert(new_path_id, path_bundle);
            }
        } else if selected.pathnode_id == path.segments.len() {
            // If it was the last segment ...

            let end_node = self.nodes.get_mut(&path.end_node.node_id).unwrap();
            // ... and it does not have any other path, just extend it ...
            if end_node.count_connections() < 2 {
                if let Some(node_sel) = end_node_sel {
                    path.end_node = node_sel;
                    self.nodes
                        .get_mut(&node_sel.node_id)
                        .ok_or_else(|| format!("Node {} not found", node_sel.node_id))?
                        .paths_in_direction_mut(node_sel.direction)
                        .push(PathConnection::new(selected.path_id, ConnectPoint::End));
                } else {
                    end_node.pos = path_bundle.end();
                }

                path.extend(&path_bundle.segments);
                // Continue extending from the added segment
                self.selected_node = Some(SelectedPathNode::new(
                    selected.path_id,
                    selected.pathnode_id + path_bundle.segments.len(),
                    selected.direction,
                ));
            } else {
                // ... unless there are already connected paths in which case we need to insert a new path.

                // Allocate a path id for the new path
                let new_path_id = self.path_id_gen;
                self.path_id_gen += 1;
                let new_end_node_id = end_node_sel.unwrap_or_else(|| {
                    let id = self.node_id_gen;
                    self.node_id_gen += 1;
                    SelectedNode {
                        node_id: id,
                        direction: SegmentDirection::Backward,
                    }
                });

                end_node
                    .paths_in_direction_mut(!path.end_node.direction)
                    .push(PathConnection::new(new_path_id, ConnectPoint::Start));

                // Set up the new end node
                let new_end_node = self
                    .nodes
                    .entry(new_end_node_id.node_id)
                    .or_insert_with(|| TrainNode::new(path_bundle.end()));

                // Then, add a new node for the end of the new path.
                new_end_node
                    .paths_in_direction_mut(new_end_node_id.direction)
                    .push(PathConnection::new(new_path_id, ConnectPoint::End));
                path_bundle.start_node = path.end_node.reversed();
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
            let new_end_node_id = end_node_sel.unwrap_or_else(|| {
                let id = self.node_id_gen;
                self.node_id_gen += 1;
                SelectedNode {
                    node_id: id,
                    direction: SegmentDirection::Forward,
                }
            });

            // First, create a path for the segments after the selected node.
            let split_path = PathBundle::multi(
                path.segments[selected.pathnode_id..]
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>(),
                NodeConnection::new(split_node_id, SegmentDirection::Forward),
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
            split_node
                .paths_in_direction_mut(selected.direction)
                .push(PathConnection::new(new_path_id, ConnectPoint::Start));
            self.nodes.insert(split_node_id, split_node);

            // Next, truncate the selected path after the selected node.
            path.truncate(selected.pathnode_id);
            if let Some(node) = self.nodes.get_mut(&path.end_node.node_id) {
                let connections = node.paths_in_direction_mut(path.end_node.direction);
                connections.retain(|p| p.path_id != selected.path_id);
                connections.push(PathConnection::new(split_path_id, ConnectPoint::End));
            }
            path.end_node = NodeConnection::new(split_node_id, SegmentDirection::Backward);

            // Move the stations after the split point to the split path and subtract the first half path
            let new_path_len = path.track.len() as f64;
            for station in self.stations.values_mut() {
                if station.path_id == selected.path_id && new_path_len < station.s {
                    station.path_id = split_path_id;
                    station.s -= new_path_len;
                }
            }
            train.transfer_path(selected.path_id, new_path_len);

            // Add the split path after the selected path is modified, in order to avoid the borrow checker.
            self.paths.insert(split_path_id, split_path);

            // Set up the new end node
            let new_end_node = self
                .nodes
                .entry(new_end_node_id.node_id)
                .or_insert_with(|| TrainNode::new(path_bundle.end()));

            // Then, add a new node for the end of the new path.
            new_end_node
                .backward_paths
                .push(PathConnection::new(new_path_id, ConnectPoint::End));
            path_bundle.start_node = NodeConnection::new(split_node_id, selected.direction);
            path_bundle.end_node =
                NodeConnection::new(new_end_node_id.node_id, SegmentDirection::Backward);
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
        for station in self.stations.values_mut() {
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
                return path_angle(con, path).map(|angle| wrap_angle(angle + std::f64::consts::PI));
            }
        }
        None
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

    pub(crate) fn find_loader_position(
        &self,
        pos: Vec2,
    ) -> Option<(StationId, usize, Vec2, f64, f64)> {
        self.stations
            .iter()
            .filter_map(|(id, station)| {
                let Some(path) = self.paths.get(&station.path_id) else {
                    return None;
                };
                let Some(track_pos) = path.track.get(station.s as usize) else {
                    return None;
                };
                let Some(tangent) = interpolate_path_tangent(&path.track, station.s) else {
                    return None;
                };
                let tangent = tangent.normalized();
                let normal = tangent.left90();
                let orient = tangent.y.atan2(tangent.x);
                Some(
                    (0..3)
                        .filter_map(|i| {
                            let loader_pos = *track_pos
                                - tangent * i as f64 * CAR_LENGTH * SEGMENT_LENGTH
                                + normal * RAIL_WIDTH;
                            let delta = pos - loader_pos;
                            let dist = delta.length();
                            Some((*id, i, loader_pos, NotNan::new(dist).ok()?, orient))
                        })
                        .collect::<Vec<_>>(),
                )
            })
            .flatten()
            .min_by_key(|(_, _, _, dist, _)| *dist)
            .map(|(id, idx, pos, dist, orient)| (id, idx, pos, *dist, orient))
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
            if train.in_path_segment(path_id, path, seg) {
                return Err("You can't delete a segment while a train is on it".to_string());
            }
            let delete_begin = if 0 < seg {
                path.track_ranges[seg - 1] as f64
            } else {
                0.
            };
            let delete_end = path.track_ranges[seg] as f64;
            println!("Delete node range: {}, {}", delete_begin, delete_end);

            // We accumulate added nodes in a temporary buffer since a mutable borrow cannot be shared among
            // add_node and remove_node callbacks.
            let mut node_id_gen = self.node_id_gen;
            let mut added_nodes = vec![];

            let new_path = path.delete_segment(
                seg,
                |added_node, con_point| {
                    let node_id = node_id_gen;
                    added_nodes.push((node_id, added_node, con_point));
                    node_id_gen += 1;
                    NodeConnection::new(node_id, SegmentDirection::Forward)
                },
                |node_con| {
                    if let Some(node) = self.nodes.get_mut(&node_con.node_id) {
                        node.paths_in_direction_mut(node_con.direction)
                            .retain(|path_con| path_con.path_id != path_id);
                    }
                },
            );

            // Postprocess after deleting a segment to update nodes, if it creates new nodes
            for (node_id, node_pos, con_point) in added_nodes {
                let mut node = TrainNode::new(node_pos);
                node.forward_paths
                    .push(PathConnection::new(path_id, con_point));
                self.nodes.insert(node_id, node);
            }

            self.node_id_gen = node_id_gen;

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

                train.for_each_car(|car| {
                    move_s(&mut car.path_id, &mut car.s, "train");
                });
                self.stations.retain(|_i, station| {
                    if station.path_id != path_id {
                        return true;
                    }
                    let station_name = format!("station {}", station.name);
                    move_s(&mut station.path_id, &mut station.s, &station_name)
                });
            } else {
                self.stations.retain(|_i, station| {
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

/// The selected node and the direction. It happens to be the same as `NodeConnection` so we reuse the type.
/// Note that this type cannot select a pathnode in the middle of a path, unlike `SelectedPathNode`.
pub(crate) type SelectedNode = NodeConnection;

/// A selected path node with direction. A path node is a point between segments, including both ends of the path.
/// 0th and nth pathnodes are at the ends, where n is the number of segments.
///
/// This type is used to store the position and the direction of a segment to add.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Copy, Hash, PartialEq, Eq, Serialize, Deserialize, Default, Debug)]
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
