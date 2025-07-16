use std::{
    collections::{HashSet, VecDeque},
    sync::{Arc, Condvar, Mutex},
};

use super::{HeightMap, HeightMapKey, HeightMapParams, HeightMapTile};

/// The tile generator that can run tasks in parallel threads.
/// It is very high performance but does not work in Wasm.
/// (Maybe there is a way to use it on Wasm with SharedMemoryBuffer,
/// but it seems tooling and browser support are not readily available.)
pub(super) struct TileGen {
    /// The set of already requested tiles, used to avoid duplicate requests.
    requested: HashSet<HeightMapKey>,
    /// Request queue and the event to notify workers.
    notify: Arc<(Mutex<VecDeque<(HeightMapKey, usize)>>, Condvar)>,
    // We should join the handles, but we can also leave them and kill the process, since the threads only interact
    // with memory.
    _workers: Vec<std::thread::JoinHandle<()>>,
    /// The channel to receive finished tiles.
    finished: std::sync::mpsc::Receiver<(HeightMapKey, HeightMapTile)>,
}

impl TileGen {
    pub fn new(params: &HeightMapParams) -> Self {
        let num_threads = std::thread::available_parallelism().map_or(8, |v| v.into());
        let notify = Arc::new((Mutex::new(VecDeque::new()), Condvar::new()));
        let (finished_tx, finished_rx) = std::sync::mpsc::channel();
        let workers = (0..num_threads)
            .map(|_| {
                let params_copy = params.clone();
                let notify_copy = notify.clone();
                let finished_tx_copy = finished_tx.clone();
                std::thread::spawn(move || {
                    let (lock, cvar) = &*notify_copy;
                    loop {
                        let mut queue = lock.lock().unwrap();
                        while queue.is_empty() {
                            queue = cvar.wait(queue).unwrap();
                        }

                        if let Some((key, contour_grid_step)) = queue.pop_back() {
                            // Drop the lock just before heavy lifting
                            drop(queue);
                            let tile = HeightMapTile::new(
                                HeightMap::new_map(&params_copy, &key).unwrap(),
                                contour_grid_step,
                            );
                            finished_tx_copy.send((key, tile)).unwrap();
                        }
                    }
                })
            })
            .collect();
        Self {
            requested: HashSet::new(),
            notify,
            _workers: workers,
            finished: finished_rx,
        }
    }

    pub fn request_tile(&mut self, key: &HeightMapKey, contours_grid_step: usize) {
        // We could not use entry API due to a lifetime issue.
        if !self.requested.contains(key) {
            let (lock, cvar) = &*self.notify;
            let mut queue = lock.lock().unwrap();
            queue.push_back((*key, contours_grid_step));
            cvar.notify_one();
            self.requested.insert(*key);
        }
    }

    pub fn update(&mut self) -> Result<Vec<(HeightMapKey, HeightMapTile)>, String> {
        let tiles = self.finished.try_iter().collect::<Vec<_>>();

        Ok(tiles)
    }
}
