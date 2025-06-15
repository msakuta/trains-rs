mod dijkstra;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    app::HeightMap,
    path_utils::{interpolate_path, interpolate_path_heading, interpolate_path_tangent},
    train_tracks::{
        ConnectPoint, PathBundle, PathConnection, Paths, SegmentDirection, TrainTask, TrainTracks,
    },
    vec2::Vec2,
};

const CAR_LENGTH: f64 = 1.;
const TRAIN_ACCEL: f64 = 0.001;
const MAX_SPEED: f64 = 1.;
const THRUST_ACCEL: f64 = 0.001;
const GRAD_ACCEL: f64 = 0.0002;

#[derive(Serialize, Deserialize)]
pub(crate) struct Train {
    /// The index of the path_bundle that the train is on
    pub path_id: usize,
    /// Position along the track
    pub s: f64,
    /// Speed along s
    pub speed: f64,
    pub num_cars: usize,
    #[serde(skip)]
    pub train_task: TrainTask,
    #[serde(skip)]
    pub schedule: Vec<usize>,
    pub train_direction: SegmentDirection,
    /// The index of the direction of the path in the next branch.
    #[serde(skip)]
    pub switch_path: usize,
    #[serde(skip)]
    pub route: Vec<PathConnection>,
}

impl Train {
    pub fn new() -> Self {
        Self {
            s: 0.,
            speed: 0.,
            num_cars: 3,
            path_id: 0,
            train_task: TrainTask::Idle,
            schedule: vec![],
            train_direction: SegmentDirection::Forward,
            switch_path: 0,
            route: vec![],
        }
    }

    pub fn update(&mut self, thrust: f64, heightmap: &HeightMap, tracks: &TrainTracks) {
        if let TrainTask::Wait(timer) = &mut self.train_task {
            *timer -= 1;
            if *timer <= 1 {
                self.train_task = TrainTask::Idle;
            }
        }
        if matches!(self.train_task, TrainTask::Idle) {
            if let Some(first) = self.schedule.first().cloned() {
                self.train_task = TrainTask::Goto(first);
                self.schedule.remove(0);
                self.schedule.push(first);
            }
        }
        if let TrainTask::Goto(target) = &self.train_task {
            if let Some(target) = tracks.stations.get(target) {
                println!("Goto {target:?}, route: {:?}", self.route);
                if target.path_id == self.path_id {
                    self.move_to_s(target.s);
                } else {
                    self.find_path(target.path_id, tracks);
                    self.follow_route(tracks);
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
            if let Some((tangent, pos)) = self
                .tangent(i, &tracks.paths)
                .zip(self.pos(i, &tracks.paths))
            {
                let grad = heightmap.gradient(&pos);
                self.speed -= grad.dot(tangent) * GRAD_ACCEL / self.num_cars as f64;
            }
        }

        fn get_clamped(v: &[PathConnection], i: usize) -> Option<&PathConnection> {
            v.get(i.clamp(0, v.len().saturating_sub(1)))
        }

        if let Some(path) = tracks.paths.get(&self.path_id) {
            if self.s == 0. && self.speed < 0. {
                let prev_path = tracks.nodes.get(&path.start_node.node_id).and_then(|node| {
                    // If we came from forward, we should continue on backward
                    let path_con = get_clamped(
                        node.paths_in_direction(!path.start_node.direction),
                        self.switch_path,
                    )?;
                    let path_ref = tracks.paths.get(&path_con.path_id)?;
                    Some((path_con, path_ref))
                });
                if let Some((prev_path, prev_path_ref)) = prev_path {
                    match prev_path.connect_point {
                        ConnectPoint::Start => {
                            self.path_id = prev_path.path_id;
                            self.s = 0.;
                            self.speed *= -1.;
                            // Invert the direction if the track direction reversed
                            self.train_direction = !self.train_direction;
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
                if let Some(next_path) = tracks.nodes.get(&path.end_node.node_id).and_then(|node| {
                    // If we came from forward, we should continue on backward
                    get_clamped(
                        &node.paths_in_direction(!path.end_node.direction),
                        self.switch_path,
                    )
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
                            // Invert the direction if the track direction reversed
                            self.train_direction = !self.train_direction;
                        }
                    }
                } else {
                    self.speed = 0.;
                }
            }

            // Acquire path again because it may have changed
            let path = &tracks.paths[&self.path_id];
            self.s = (self.s + self.speed).clamp(0., path.track.len() as f64);
        }
    }

    /// Move to a position in the same path as the train.
    fn move_to_s(&mut self, target_s: f64) {
        if (target_s - self.s).abs() < 1. && self.speed.abs() < TRAIN_ACCEL {
            self.speed = 0.;
            self.train_task = TrainTask::Wait(120);
        } else if target_s < self.s {
            // speed / accel == t
            // speed * t / 2 == speed^2 / accel / 2 == dist
            // accel = sqrt(2 * dist)
            if self.speed < 0. && self.s - target_s < 0.5 * self.speed.powi(2) / TRAIN_ACCEL {
                self.speed += TRAIN_ACCEL;
            } else {
                self.speed -= TRAIN_ACCEL;
            }
        } else {
            if 0. < self.speed && target_s - self.s < 0.5 * self.speed.powi(2) / TRAIN_ACCEL {
                self.speed -= TRAIN_ACCEL;
            } else {
                self.speed += TRAIN_ACCEL;
            }
        }
    }

    pub fn tangent(&self, car_idx: usize, paths: &HashMap<usize, PathBundle>) -> Option<Vec2<f64>> {
        interpolate_path_tangent(
            &paths.get(&self.path_id)?.track,
            self.s - car_idx as f64 * CAR_LENGTH,
        )
    }

    pub fn pos(&self, car_idx: usize, paths: &Paths) -> Option<Vec2<f64>> {
        interpolate_path(
            &paths.get(&self.path_id)?.track,
            self.s - car_idx as f64 * CAR_LENGTH,
        )
    }

    pub fn heading(&self, car_idx: usize, paths: &Paths) -> Option<f64> {
        interpolate_path_heading(
            &paths.get(&self.path_id)?.track,
            self.s - car_idx as f64 * CAR_LENGTH,
        )
    }

    pub fn offset(&mut self, path_id: usize, s: f64) {
        if self.path_id == path_id {
            self.s += s;
        }
    }
}
