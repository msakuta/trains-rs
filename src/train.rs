mod dijkstra;

use std::collections::HashMap;

use eframe::egui::ahash::HashSet;
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
    /// Speed along s
    pub speed: f64,
    pub cars: Vec<TrainCar>,
    #[serde(skip)]
    pub train_task: TrainTask,
    #[serde(skip)]
    pub schedule: Vec<usize>,
    /// The index of the direction of the path in the next branch.
    #[serde(skip)]
    pub switch_path: usize,
    #[serde(skip)]
    pub route: Vec<PathConnection>,
}

impl Train {
    pub fn new() -> Self {
        let cars = (0..3)
            .map(|i| TrainCar {
                path_id: 0,
                s: i as f64 * CAR_LENGTH,
                direction: SegmentDirection::Forward,
            })
            .collect();
        Self {
            speed: 0.,
            cars,
            train_task: TrainTask::Idle,
            schedule: vec![],
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
                if target.path_id == self.path_id() {
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
        for (i, car) in self.cars.iter().enumerate() {
            if let Some((tangent, pos)) = self
                .tangent(i, &tracks.paths)
                .zip(self.pos(i, &tracks.paths))
            {
                let grad = heightmap.gradient(&pos);
                self.speed -= grad.dot(tangent) * GRAD_ACCEL / self.cars.len() as f64;
            }
        }

        for (i, car) in self.cars.iter_mut().enumerate() {
            let mut speed = self.speed;
            car.update(&mut speed, self.switch_path, tracks);
            if i == 0 {
                self.speed = speed;
            }
        }
    }

    pub fn path_id(&self) -> usize {
        // A train should have one or more cars.
        self.cars[0].path_id
    }

    fn path_ids(&self) -> HashSet<usize> {
        // A train should have one or more cars.
        self.cars.iter().map(|car| car.path_id).collect()
    }

    pub fn s(&self) -> f64 {
        self.cars[0].s
    }

    pub fn direction(&self) -> SegmentDirection {
        self.cars[0].direction
    }

    pub fn in_path_segment(&self, path_id: usize, path: &PathBundle, seg_id: usize) -> bool {
        self.cars
            .iter()
            .any(|car| car.path_id == path_id && seg_id == path.find_seg_by_s(car.s as usize))
    }

    pub fn for_each_car(&mut self, mut f: impl FnMut(&mut TrainCar)) {
        for car in &mut self.cars {
            f(car);
        }
    }

    /// Move to a position in the same path as the train.
    fn move_to_s(&mut self, target_s: f64) {
        if (target_s - self.s()).abs() < 1. && self.speed.abs() < TRAIN_ACCEL {
            self.speed = 0.;
            self.train_task = TrainTask::Wait(120);
        } else if target_s < self.s() {
            // speed / accel == t
            // speed * t / 2 == speed^2 / accel / 2 == dist
            // accel = sqrt(2 * dist)
            if self.speed < 0. && self.s() - target_s < 0.5 * self.speed.powi(2) / TRAIN_ACCEL {
                self.speed += TRAIN_ACCEL;
            } else {
                self.speed -= TRAIN_ACCEL;
            }
        } else {
            if 0. < self.speed && target_s - self.s() < 0.5 * self.speed.powi(2) / TRAIN_ACCEL {
                self.speed -= TRAIN_ACCEL;
            } else {
                self.speed += TRAIN_ACCEL;
            }
        }
    }

    pub fn tangent(&self, car_idx: usize, paths: &Paths) -> Option<Vec2<f64>> {
        self.cars.get(car_idx)?.tangent(paths)
    }

    pub fn pos(&self, car_idx: usize, paths: &Paths) -> Option<Vec2<f64>> {
        self.cars.get(car_idx)?.pos(paths)
    }

    pub fn heading(&self, car_idx: usize, paths: &Paths) -> Option<f64> {
        self.cars.get(car_idx)?.heading(paths)
    }

    pub fn offset(&mut self, path_id: usize, s: f64) {
        for car in &mut self.cars {
            if car.path_id == path_id {
                car.s += s;
            }
        }
    }

    /// Move cars at path_id and has position further than from_s.
    /// Call it when you split a path at from_s.
    pub fn transfer_path(&mut self, path_id: usize, from_s: f64) {
        for car in &mut self.cars {
            if car.path_id == path_id && from_s < car.s {
                car.path_id = path_id;
                car.s -= from_s;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct TrainCar {
    /// The index of the path_bundle that the train is on
    pub path_id: usize,
    /// Position along the track
    pub s: f64,
    pub direction: SegmentDirection,
}

impl TrainCar {
    pub fn tangent(&self, paths: &Paths) -> Option<Vec2<f64>> {
        interpolate_path_tangent(&paths.get(&self.path_id)?.track, self.s)
    }

    pub fn pos(&self, paths: &Paths) -> Option<Vec2<f64>> {
        interpolate_path(&paths.get(&self.path_id)?.track, self.s)
    }

    pub fn heading(&self, paths: &Paths) -> Option<f64> {
        interpolate_path_heading(&paths.get(&self.path_id)?.track, self.s)
    }

    fn update(&mut self, speed: &mut f64, switch_path: usize, tracks: &TrainTracks) {
        if let Some(path) = tracks.paths.get(&self.path_id) {
            if self.s == 0. && *speed < 0. {
                let prev_path = tracks.nodes.get(&path.start_node.node_id).and_then(|node| {
                    // If we came from forward, we should continue on backward
                    let path_con = get_clamped(
                        node.paths_in_direction(!path.start_node.direction),
                        switch_path,
                    )?;
                    let path_ref = tracks.paths.get(&path_con.path_id)?;
                    Some((path_con, path_ref))
                });
                if let Some((prev_path, prev_path_ref)) = prev_path {
                    match prev_path.connect_point {
                        ConnectPoint::Start => {
                            self.path_id = prev_path.path_id;
                            self.s = 0.;
                            *speed *= -1.;
                            // Invert the direction if the track direction reversed
                            self.direction = !self.direction;
                        }
                        ConnectPoint::End => {
                            self.path_id = prev_path.path_id;
                            self.s = prev_path_ref.track.len() as f64;
                        }
                    }
                } else {
                    *speed = 0.;
                }
            }

            if path.track.len() as f64 <= self.s && 0. < *speed {
                if let Some(next_path) = tracks.nodes.get(&path.end_node.node_id).and_then(|node| {
                    // If we came from forward, we should continue on backward
                    get_clamped(
                        &node.paths_in_direction(!path.end_node.direction),
                        switch_path,
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
                            *speed *= -1.;
                            // Invert the direction if the track direction reversed
                            self.direction = !self.direction;
                        }
                    }
                } else {
                    *speed = 0.;
                }
            }
        }

        // Acquire path again because it may have changed
        let path = &tracks.paths[&self.path_id];
        self.s = (self.s + *speed).clamp(0., path.track.len() as f64);
    }
}

fn get_clamped(v: &[PathConnection], i: usize) -> Option<&PathConnection> {
    v.get(i.clamp(0, v.len().saturating_sub(1)))
}
