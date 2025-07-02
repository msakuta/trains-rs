use crate::{
    path_utils::{_bezier_interp, _bezier_length, PathSegment},
    train_tracks::SegmentDirection,
    vec2::Vec2,
};

use super::SEGMENT_LENGTH;

use ordered_float::NotNan;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
/// A path and its accompanying data. A path is a sequence of segments without a branch.
/// `start_node` and `end_node` are the node ids at both ends.
pub(crate) struct PathBundle {
    /// A segment is a continuous line or curve with the same curvature
    pub(super) segments: Vec<PathSegment>,
    /// Interpolated points along the track in the interval SEGMENT_LENGTH
    pub track: Vec<Vec2<f64>>,
    /// S length is the same as track.len(), but it has fractional parts.
    pub s_length: f64,
    pub(super) track_ranges: Vec<usize>,
    /// A node id of the starting node.
    pub(crate) start_node: NodeConnection,
    /// A node id of the end node.
    pub(crate) end_node: NodeConnection,
}

impl PathBundle {
    pub(super) fn single(
        path_segment: PathSegment,
        start_node: NodeConnection,
        end_node: NodeConnection,
    ) -> Self {
        let (track, s_length, track_ranges) = compute_track_ps(&[path_segment]);
        PathBundle {
            segments: vec![path_segment],
            track,
            s_length,
            track_ranges,
            start_node,
            end_node,
        }
    }

    pub(super) fn multi(
        path_segments: impl Into<Vec<PathSegment>>,
        start_node: NodeConnection,
        end_node: NodeConnection,
    ) -> Self {
        let path_segments = path_segments.into();
        let (track, s_length, track_ranges) = compute_track_ps(&path_segments);
        PathBundle {
            segments: path_segments,
            track,
            s_length,
            track_ranges,
            start_node,
            end_node,
        }
    }

    /// Append path segments at the start. Try to avoid this since it is slow.
    /// Unlike `extend`, consumes the argument to reverse them in place.
    /// Returns a number of nodes added to the track.
    pub fn prepend(&mut self, path_segments: Vec<PathSegment>) -> usize {
        let prev_track_len = self.track.len();
        for segment in path_segments.into_iter() {
            println!("appending segment: {segment:?}");
            self.segments.insert(0, segment.reverse());
        }
        (self.track, self.s_length, self.track_ranges) = compute_track_ps(&self.segments);
        self.track.len() - prev_track_len
    }

    pub fn extend(&mut self, path_segments: &[PathSegment]) {
        self.segments.extend_from_slice(path_segments);
        (self.track, self.s_length, self.track_ranges) = compute_track_ps(&self.segments);
    }

    /// Modifies the path and update track nodes
    pub fn truncate(&mut self, node: usize) {
        if node < self.segments.len() {
            self.segments.truncate(node);
            (self.track, self.s_length, self.track_ranges) = compute_track_ps(&self.segments);
        }
    }

    pub fn _segments(&self) -> impl Iterator<Item = &PathSegment> {
        self.segments.iter()
    }

    pub fn seg_track(&self, seg: usize) -> &[Vec2<f64>] {
        let start = if seg == 0 {
            0
        } else {
            self.track_ranges[seg - 1]
        };
        // println!("start: {start}, track_ranges: {:?}, seg: {seg}/{}", self.track_ranges, self.segments.len());
        &self.track[start..self.track_ranges[seg]]
    }

    pub fn start(&self) -> Vec2<f64> {
        // It is a logical bug if a PathBundle has no segments.
        self.segments.first().unwrap().start()
    }

    pub fn end(&self) -> Vec2<f64> {
        // It is a logical bug if a PathBundle has no segments.
        self.segments.last().unwrap().end()
    }

    /// Length as in s space. For physical length, multiply with SEGMENT_LENGTH.
    pub fn s_length(&self) -> f64 {
        self.track.len() as f64
    }

    pub fn connect_node(&self, con: ConnectPoint) -> NodeConnection {
        match con {
            ConnectPoint::Start => self.start_node,
            ConnectPoint::End => self.end_node,
        }
    }

    /// Returns (segment id, node id)
    pub fn find_node(&self, pos: Vec2<f64>, dist_thresh: f64) -> Option<(usize, usize)> {
        let dist2_thresh = dist_thresh.powi(2);
        let closest_node: Option<(usize, f64)> =
            self.track.iter().enumerate().fold(None, |acc, cur| {
                let dist2 = (*cur.1 - pos).length2();
                if let Some(acc) = acc {
                    if acc.1 < dist2 {
                        Some(acc)
                    } else {
                        Some((cur.0, dist2))
                    }
                } else if dist2 < dist2_thresh {
                    Some((cur.0, dist2))
                } else {
                    None
                }
            });
        closest_node.map(|(i, _)| (self.find_seg_by_s(i), i))
    }

    pub fn find_seg_by_s(&self, i: usize) -> usize {
        let res = self.track_ranges.binary_search(&i);
        let seg = match res {
            Ok(res) => res,
            Err(res) => res,
        };
        seg
    }

    /// Deletes a segment, optionally splitting the path that contains the segment.
    ///
    /// Returns a new path created by splitting this one, if the segment was in the middle of a path.
    /// `add_node` parameter is called when adding a node becomes necessary by the splitting.
    ///
    /// Note that deleting a node by isolation doesn't make sense, because a node is a interpolated cached position
    /// from a segment parameters.
    pub(super) fn delete_segment(
        &mut self,
        seg: usize,
        mut add_node: impl FnMut(Vec2<f64>) -> NodeConnection,
    ) -> Option<PathBundle> {
        let mut new_path = vec![];
        let prev_end_node = self.end_node;
        if seg == 0 {
            self.segments.remove(0);
            if let Some(first_seg) = self.segments.first() {
                // Keep old node which will accumulate as garbage
                let new_start = add_node(first_seg.start());
                self.start_node = new_start;
            }
        } else {
            new_path = self.segments[seg + 1..].to_vec();
            self.segments.truncate(seg);
            if let Some(last_seg) = self.segments.last() {
                self.end_node = add_node(last_seg.end());
            }
        }
        (self.track, self.s_length, self.track_ranges) = compute_track_ps(&self.segments);
        if !new_path.is_empty() {
            let prev_node = new_path.first().map(|seg| add_node(seg.start()));
            prev_node.map(|prev_node| Self::multi(new_path, prev_node, prev_end_node))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub(crate) enum ConnectPoint {
    #[default]
    Start,
    End,
}

impl std::ops::Not for ConnectPoint {
    type Output = Self;

    fn not(self) -> Self::Output {
        match self {
            ConnectPoint::Start => ConnectPoint::End,
            ConnectPoint::End => ConnectPoint::Start,
        }
    }
}

/// A connection to a path. This data structure indicates only one way of the connection,
/// but the other path should have the connection in the other way to form a bidirectional graph.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct PathConnection {
    pub path_id: usize,
    /// Where does this path connects to the other path
    pub connect_point: ConnectPoint,
}

impl PathConnection {
    pub fn new(path_id: usize, connect_point: ConnectPoint) -> Self {
        Self {
            path_id,
            connect_point,
        }
    }
}

/// A connection to a node. This is similar to [`PathConnection`], but it indicates connection from a path to a
/// node. It is a pair of node id and a direction.
#[derive(Clone, Copy, Hash, PartialEq, Eq, Debug, Serialize, Deserialize, Default)]
pub(crate) struct NodeConnection {
    pub node_id: usize,
    /// Which direction of the node we are connecting to. Note that the direction would be ambiguous if the same path
    /// connects to the same node at both ends, if we didn't put this information here.
    pub direction: SegmentDirection,
}

impl NodeConnection {
    pub fn new(node_id: usize, direction: SegmentDirection) -> Self {
        Self { node_id, direction }
    }

    pub fn reversed(&self) -> Self {
        Self {
            node_id: self.node_id,
            direction: !self.direction,
        }
    }
}

pub(super) fn _compute_track(control_points: &[Vec2<f64>]) -> Vec<Vec2<f64>> {
    let segment_lengths =
        control_points
            .windows(3)
            .enumerate()
            .fold(vec![], |mut acc, (cur_i, cur)| {
                if cur_i % 2 == 0 {
                    acc.push(_bezier_length(cur).unwrap_or(0.));
                }
                acc
            });
    let cumulative_lengths: Vec<f64> = segment_lengths.iter().fold(vec![], |mut acc, cur| {
        if let Some(v) = acc.last() {
            acc.push(v + *cur);
        } else {
            acc.push(*cur);
        }
        acc
    });
    let total_length: f64 = segment_lengths.iter().copied().sum();
    let num_nodes = (total_length / SEGMENT_LENGTH) as usize;
    println!("cumulative_lengths: {cumulative_lengths:?}");
    (0..num_nodes)
        .map(|i| {
            let fidx = i as f64 * SEGMENT_LENGTH;
            let (seg_idx, _) = cumulative_lengths
                .iter()
                .enumerate()
                .find(|(_i, l)| fidx < **l)
                .unwrap();
            let (frem, _cum_len) = if seg_idx == 0 {
                (fidx, 0.)
            } else {
                let cum_len = cumulative_lengths[seg_idx - 1];
                (fidx - cum_len, cum_len)
            };
            // println!("[{}, {}, {}],", seg_idx, frem, cum_len + frem);
            let seg_len = segment_lengths[seg_idx];
            _bezier_interp(
                &control_points[seg_idx * 2..seg_idx * 2 + 3],
                frem / seg_len,
            )
            .unwrap()
        })
        .collect()
}

/// Computes the positions of the track points with the list of path segments.
/// Think of path segments as the control points to determine the shape of the track, and this function
/// returns the exact shape of the track. However, it is an approximation by sampled line segments, not an analytical
/// form.
pub(super) fn compute_track_ps(path_segments: &[PathSegment]) -> (Vec<Vec2<f64>>, f64, Vec<usize>) {
    if path_segments.is_empty() {
        return (vec![], 0., vec![]);
    }
    let segment_lengths: Vec<_> = path_segments.iter().map(|seg| seg.length()).collect();
    let cumulative_lengths: Vec<f64> = segment_lengths.iter().fold(vec![], |mut acc, cur| {
        if let Some(v) = acc.last() {
            acc.push(v + *cur);
        } else {
            acc.push(*cur);
        }
        acc
    });
    let total_length = *cumulative_lengths.last().unwrap_or(&0.);
    let num_nodes = (total_length / SEGMENT_LENGTH) as usize + 1;
    let mut last_idx = None;
    let mut track_ranges = vec![];

    let lookup_segment = |fidx: f64| {
        let seg_idx = cumulative_lengths
            .iter()
            .enumerate()
            .find(|(_i, l)| fidx < **l)
            .map(|(i, _)| i)
            .unwrap_or_else(|| cumulative_lengths.len() - 1);
        let (frem, _cum_len) = if seg_idx == 0 {
            (fidx, 0.)
        } else {
            let cum_len = cumulative_lengths[seg_idx - 1];
            (fidx - cum_len, cum_len)
        };
        (seg_idx, frem)
    };

    let path_nodes: Vec<_> = (0..=num_nodes)
        .filter_map(|i| {
            let (seg_idx, frem) = lookup_segment(i as f64 * SEGMENT_LENGTH);
            if last_idx.is_some_and(|idx| idx != seg_idx) {
                track_ranges.push(i);
            }
            last_idx = Some(seg_idx);
            let seg_len = segment_lengths[seg_idx];
            path_segments[seg_idx].interp((frem / seg_len).clamp(0., 1.))
        })
        .collect();

    // Build the measured cumulative lengths of path segments, not an heuristic estimation.
    // It makes a lot of difference in Bezier curves.
    let mut acc = 0.;
    let cumulative_lengths = path_nodes
        .iter()
        .zip(path_nodes.iter().skip(1))
        .enumerate()
        .map(|(_i, (node, next))| {
            let dist = (*next - *node).length();
            // println!("[{i}]: {dist}, {acc}");
            acc += dist;
            NotNan::new(acc).unwrap()
        })
        .collect::<Vec<_>>();

    // Reverse map from distance to s value
    let lookup = |dist: f64| -> f64 {
        let dist = NotNan::new(dist).unwrap();
        match cumulative_lengths.binary_search(&dist) {
            Ok(i) => i as f64,
            Err(i) => {
                if i == 0 {
                    i as f64
                } else if let Some([prev_length, next_length]) = cumulative_lengths.get(i - 1..=i) {
                    let segment_length = next_length - *prev_length;
                    i as f64 + ((dist - *prev_length) / segment_length).into_inner()
                } else {
                    i as f64
                }
            }
        }
    };

    let num_nodes = cumulative_lengths
        .last()
        .map_or(0, |&x| (x.into_inner() / SEGMENT_LENGTH) as usize);
    let resampled_nodes: Vec<_> = (0..=num_nodes)
        .filter_map(|i| {
            let fidx = lookup(i as f64 * SEGMENT_LENGTH) * SEGMENT_LENGTH;
            let (seg_idx, frem) = lookup_segment(fidx);
            let seg_len = segment_lengths[seg_idx];
            path_segments[seg_idx].interp((frem / seg_len).clamp(0., 1.))
        })
        .chain(std::iter::once(path_segments.last().unwrap().end()))
        .collect();

    let mut resampled_track_ranges: Vec<_> = track_ranges
        .iter()
        .map(|d| (lookup(*d as f64 * SEGMENT_LENGTH) as usize).min(resampled_nodes.len() - 1))
        .collect();

    // Recheck uniform distance
    // for (i, (node, next)) in resampled_nodes
    //     .iter()
    //     .zip(resampled_nodes.iter().skip(1))
    //     .enumerate()
    // {
    //     let dist = (*next - *node).length();
    //     println!("re[{i}]: {dist}");
    // }

    if last_idx != Some(resampled_nodes.len()) {
        resampled_track_ranges.push(resampled_nodes.len());
    }
    (
        resampled_nodes,
        cumulative_lengths
            .last()
            .map(|v| v.into_inner())
            .unwrap_or(0.)
            / SEGMENT_LENGTH,
        resampled_track_ranges,
    )
}
