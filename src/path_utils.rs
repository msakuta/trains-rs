use crate::vec2::Vec2;

use serde::{Deserialize, Serialize};

pub(super) fn find_closest_node(path: &[Vec2<f64>], pos: Vec2<f64>) -> f64 {
    let closest_two =
        path.iter()
            .enumerate()
            .fold([None; 2], |mut acc: [Option<(usize, f64)>; 2], cur| {
                let dist2 = (pos - *cur.1).length2();
                // Insertion sort up to closest 2 elements
                if let [Some(acc0), _] = acc {
                    if dist2 < acc0.1 {
                        acc[1] = Some(acc0);
                        acc[0] = Some((cur.0, dist2));
                        return acc;
                    }
                }
                if let [Some(_), None] = acc {
                    acc[1] = Some((cur.0, dist2));
                    return acc;
                }
                if let [Some(_), Some(acc1)] = acc {
                    if dist2 < acc1.1 {
                        acc[1] = Some((cur.0, dist2));
                    }
                    return acc;
                }
                [Some((cur.0, dist2)), None]
            });

    // We make a strong assumption that the shortest segment's ends are closest vertices of the whole path,
    // which is not necessarily true.
    match closest_two {
        [Some(first), Some(second)] => {
            if first.0 == second.0 + 1 || first.0 + 1 == second.0 {
                let (prev, next) = if second.0 < first.0 {
                    (second, first)
                } else {
                    (first, second)
                };
                let segment = path[next.0] - path[prev.0];
                let segment_tangent = segment.normalized();
                // let segment_normal = Vec2::new(segment_tangent.y, -segment_tangent.x);
                let pos_delta = pos - path[prev.0];
                // let segment_dist = pos_delta.dot(segment_normal).abs();
                // let segment_dist2 = dbg!(segment_dist.powi(2));
                let segment_s = pos_delta.dot(segment_tangent) / segment.length();
                if 0. < segment_s && segment_s < 1. {
                    prev.0 as f64 + segment_s
                } else {
                    first.0 as f64
                }
            } else {
                first.0 as f64
            }
        }
        [Some(first), None] => first.0 as f64,
        _ => unreachable!(),
    }
}

#[test]
fn test_closest_node() {
    let path = vec![
        Vec2::new(0., 0.),
        Vec2::new(5., 5.),
        Vec2::new(10., 10.),
        Vec2::new(15., 15.),
    ];
    assert!((find_closest_node(&path, Vec2::new(-5., 0.)) - 0.).abs() < 1e-6);
    assert!((find_closest_node(&path, Vec2::new(0., 5.)) - 0.5).abs() < 1e-6);
    assert!((find_closest_node(&path, Vec2::new(5., 10.)) - 1.5).abs() < 1e-6);
    assert!((find_closest_node(&path, Vec2::new(15., 10.)) - 2.5).abs() < 1e-6);
    assert!((find_closest_node(&path, Vec2::new(20., 20.)) - 3.).abs() < 1e-6);
}

pub(crate) fn interpolate_path(path: &[Vec2<f64>], s: f64) -> Option<Vec2<f64>> {
    if path.len() == 0 {
        return None;
    }
    if s <= 0. {
        return Some(path[0]);
    }
    if (path.len() - 1) as f64 <= s {
        return Some(path[path.len() - 1]);
    }
    let i = s as usize;
    let (prev, next) = (path[i], path[i + 1]);
    let segment_delta = next - prev;
    let fr = s.rem_euclid(1.);
    Some(prev + segment_delta * fr)
}

pub(crate) fn interpolate_path_heading(path: &[Vec2<f64>], s: f64) -> Option<f64> {
    interpolate_path_tangent(path, s).map(|tangent| tangent.y.atan2(tangent.x))
}

pub(crate) fn interpolate_path_tangent(path: &[Vec2<f64>], s: f64) -> Option<Vec2<f64>> {
    if path.len() < 2 {
        return None;
    }
    if s <= 0. {
        let delta = path[1] - path[0];
        return Some(delta);
    }
    if (path.len() - 1) as f64 <= s {
        let delta = path[path.len() - 1] - path[path.len() - 2];
        return Some(delta);
    }
    let i = s as usize;
    let (prev, next) = (path[i], path[i + 1]);
    let delta = next - prev;
    Some(delta)
}

/// Quadratic Bezier curve, uses control points
pub(crate) fn _bezier_interp(c_points: &[Vec2<f64>], s: f64) -> Option<Vec2<f64>> {
    if c_points.len() < 3 {
        return None;
    }
    if s < 0. {
        return Some(c_points[0]);
    }
    if 1. < s {
        return Some(c_points[2]);
    }
    let delta1 = c_points[1] - c_points[0];
    let interp1 = c_points[0] + delta1 * s;
    let delta2 = c_points[2] - c_points[1];
    let interp2 = c_points[1] + delta2 * s;
    let delta12 = interp2 - interp1;
    Some(interp1 + delta12 * s)
}

/// Estimate the total length
pub(crate) fn _bezier_length(c_points: &[Vec2<f64>]) -> Option<f64> {
    if c_points.len() < 3 {
        return None;
    }
    const SPLITS: usize = 32;
    Some((0..SPLITS).zip(1..=SPLITS).fold(0., |acc, (f, g)| {
        let ff = f as f64 / SPLITS as f64;
        let gg = g as f64 / SPLITS as f64;
        let Some(fpos) = _bezier_interp(c_points, ff) else {
            return acc;
        };
        let Some(gpos) = _bezier_interp(c_points, gg) else {
            return acc;
        };
        acc + (fpos - gpos).length()
    }))
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct CircleArc {
    pub center: Vec2<f64>,
    pub radius: f64,
    pub start: f64,
    pub end: f64,
}

impl CircleArc {
    pub const fn new(center: Vec2<f64>, radius: f64, start: f64, end: f64) -> Self {
        Self {
            center,
            radius,
            start,
            end,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) enum PathSegment {
    Line([Vec2<f64>; 2]),
    Arc(CircleArc),
    Bezier([Vec2<f64>; 3]),
}

impl PathSegment {
    pub(crate) fn start(&self) -> Vec2<f64> {
        match self {
            Self::Line(pts) => pts[0],
            Self::Arc(arc) => arc.center + Vec2::new(arc.start.cos(), arc.start.sin()) * arc.radius,
            Self::Bezier(pts) => pts[0],
        }
    }

    /// Tangent angle
    pub(crate) fn start_angle(&self) -> f64 {
        match self {
            Self::Line(pts) => {
                let delta = pts[0] - pts[1];
                delta.y.atan2(delta.x)
            }
            Self::Arc(arc) => {
                wrap_angle(arc.start + (arc.start - arc.end).signum() * std::f64::consts::PI * 0.5)
            }
            Self::Bezier(pts) => {
                let delta = pts[0] - pts[1];
                delta.y.atan2(delta.x)
            }
        }
    }

    pub(crate) fn end(&self) -> Vec2<f64> {
        match self {
            Self::Line(pts) => pts[1],
            Self::Arc(arc) => arc.center + Vec2::new(arc.end.cos(), arc.end.sin()) * arc.radius,
            Self::Bezier(pts) => pts[2],
        }
    }

    /// Tangent angle
    pub(crate) fn end_angle(&self) -> f64 {
        match self {
            Self::Line(pts) => {
                let delta = pts[1] - pts[0];
                delta.y.atan2(delta.x)
            }
            Self::Arc(arc) => {
                wrap_angle(arc.end + (arc.end - arc.start).signum() * std::f64::consts::PI * 0.5)
            }
            Self::Bezier(pts) => {
                let delta = pts[2] - pts[1];
                delta.y.atan2(delta.x)
            }
        }
    }

    /// Compute the total length. Unlike Bezier curve, it returns exact length.
    pub(crate) fn length(&self) -> f64 {
        match self {
            Self::Line(pts) => (pts[0] - pts[1]).length(),
            Self::Arc(arc) => arc.radius * (arc.end - arc.start).abs(),
            Self::Bezier(pts) => {
                // Heuristic to return p1-p0 + p2-p1 distances
                (pts[0] - pts[1]).length() + (pts[1] - pts[2]).length()
            }
        }
    }

    pub(crate) fn interp(&self, s: f64) -> Option<Vec2<f64>> {
        if !(0. <= s && s <= 1.) {
            return None;
        }
        Some(match self {
            Self::Line(pts) => pts[0] * (1. - s) + pts[1] * s,
            Self::Arc(arc) => {
                let phase = arc.start * (1. - s) + arc.end * s;
                let relpos = Vec2::new(phase.cos(), phase.sin()) * arc.radius;
                arc.center + relpos
            }
            Self::Bezier(pts) => {
                let p01 = pts[0] * (1. - s) + pts[1] * s;
                let p12 = pts[1] * (1. - s) + pts[2] * s;
                p01 * (1. - s) + p12 * s
            }
        })
    }

    pub(crate) fn reverse(mut self) -> Self {
        match &mut self {
            Self::Line(pts) => pts.swap(0, 1),
            Self::Arc(arc) => {
                std::mem::swap(&mut arc.start, &mut arc.end);
            }
            Self::Bezier(pts) => pts.swap(0, 2),
        }
        self
    }
}

pub(crate) fn wrap_angle(x: f64) -> f64 {
    wrap_angle_offset(x, std::f64::consts::PI)
}

pub(crate) fn wrap_angle_offset(x: f64, offset: f64) -> f64 {
    use std::f64::consts::PI;
    const TWOPI: f64 = PI * 2.;
    x - (x + offset).div_euclid(TWOPI) * TWOPI
}
