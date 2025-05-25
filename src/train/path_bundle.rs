use crate::{
    path_utils::{_bezier_interp, _bezier_length, PathSegment},
    vec2::Vec2,
};

use super::SEGMENT_LENGTH;

/// A path and its accompanying data. A path is a sequence of segments without a branch.
/// `start_paths` and `end_paths` points to the paths connected to the start and end of this path, respectively.
pub(crate) struct PathBundle {
    /// A segment is a continuous line or curve with the same curvature
    pub(super) segments: Vec<PathSegment>,
    /// Interpolated points along the track in the interval SEGMENT_LENGTH
    pub track: Vec<Vec2<f64>>,
    pub(super) track_ranges: Vec<usize>,
    /// A list of connected paths to the start.
    pub(super) start_paths: Vec<PathConnection>,
    /// A list of connected paths to the start.
    pub(super) end_paths: Vec<PathConnection>,
}

impl PathBundle {
    pub(super) fn single(path_segment: PathSegment) -> Self {
        let (track, track_ranges) = compute_track_ps(&[path_segment]);
        PathBundle {
            segments: vec![path_segment],
            track,
            track_ranges,
            start_paths: vec![],
            end_paths: vec![],
        }
    }

    pub(super) fn multi(path_segments: impl Into<Vec<PathSegment>>) -> Self {
        let path_segments = path_segments.into();
        let (track, track_ranges) = compute_track_ps(&path_segments);
        PathBundle {
            segments: path_segments,
            track,
            track_ranges,
            start_paths: vec![],
            end_paths: vec![],
        }
    }

    /// Append path segments at the start. Try to avoid this since it is slow.
    /// Unlike `extend`, consumes the argument to reverse them in place.
    /// Returns a number of nodes added to the track.
    pub fn append(&mut self, path_segments: Vec<PathSegment>) -> usize {
        let prev_track_len = self.track.len();
        for segment in path_segments.into_iter() {
            println!("appending segment: {segment:?}");
            self.segments.insert(0, segment.reverse());
        }
        (self.track, self.track_ranges) = compute_track_ps(&self.segments);
        self.track.len() - prev_track_len
    }

    pub fn extend(&mut self, path_segments: &[PathSegment]) {
        self.segments.extend_from_slice(path_segments);
        (self.track, self.track_ranges) = compute_track_ps(&self.segments);
    }

    /// Modifies the path and update track nodes
    pub fn truncate(&mut self, node: usize) {
        if node < self.segments.len() {
            self.segments.truncate(node);
            (self.track, self.track_ranges) = compute_track_ps(&self.segments);
        }
    }

    pub fn segments(&self) -> impl Iterator<Item = &PathSegment> {
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
    /// Returns a tuple of a new path created by splitting this one and the node length
    ///
    /// Note that deleting a node by isolation doesn't make sense, because a node is a interpolated cached position
    /// from a segment parameters.
    pub(super) fn delete_segment(&mut self, seg: usize) -> Option<PathBundle> {
        let mut new_path = vec![];
        if seg == 0 {
            self.segments.remove(0);
        } else {
            new_path = self.segments[seg + 1..].to_vec();
            self.segments.truncate(seg);
        }
        (self.track, self.track_ranges) = compute_track_ps(&self.segments);
        if !new_path.is_empty() {
            Some(Self::multi(new_path))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy)]
pub(super) enum ConnectPoint {
    Start,
    End,
}

/// A connection to a path. This data structure indicates only one way of the connection,
/// but the other path should have the connection in the other way to form a bidirectional graph.
#[derive(Clone, Copy)]
pub(super) struct PathConnection {
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

pub(super) fn compute_track_ps(path_segments: &[PathSegment]) -> (Vec<Vec2<f64>>, Vec<usize>) {
    if path_segments.is_empty() {
        return (vec![], vec![]);
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
    let path_segments: Vec<_> = (0..=num_nodes)
        .filter_map(|i| {
            let fidx = i as f64 * SEGMENT_LENGTH;
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
            let seg_len = segment_lengths[seg_idx];
            if last_idx.is_some_and(|idx| idx != seg_idx) {
                track_ranges.push(i);
            }
            last_idx = Some(seg_idx);
            path_segments[seg_idx].interp((frem / seg_len).clamp(0., 1.))
        })
        .collect();
    if last_idx != Some(path_segments.len()) {
        track_ranges.push(path_segments.len());
    }
    (path_segments, track_ranges)
}
