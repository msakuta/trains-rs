use std::{
    collections::{HashSet, VecDeque},
    sync::{Arc, Condvar, Mutex},
};

use super::{HeightMap, HeightMapKey, HeightMapParams, HeightMapTile};

/// The tile generator that can run tasks in parallel threads.
/// It is very high performance but does not work in Wasm.
/// (Maybe there is a way to use it on Wasm with SharedMemoryBuffer,
/// but it seems tooling and browser situations are not)
pub(super) struct TileGen {
    params: HeightMapParams,
    requested: HashSet<HeightMapKey>,
    gen_queue: VecDeque<(HeightMapKey, usize)>,
}

impl TileGen {
    pub fn new(params: &HeightMapParams) -> Self {
        Self {
            params: params.clone(),
            requested: HashSet::new(),
            gen_queue: VecDeque::new(),
        }
    }

    pub fn request_tile(&mut self, key: &HeightMapKey, contours_grid_step: usize) {
        // We could not use entry API due to a lifetime issue.
        if !self.requested.contains(key) {
            self.gen_queue.push_back((*key, contours_grid_step));
            self.requested.insert(*key);
        }
    }

    pub fn update(&mut self) -> Result<Vec<(HeightMapKey, HeightMapTile)>, String> {
        let mut ret = vec![];
        // We would like to offload the tile generation to another thread, but until we know if it works in wasm,
        // we use the main thread to do it, but progressively.
        if let Some((key, contour_grid_step)) = self.gen_queue.pop_back() {
            // Drop the lock just before heavy lifting
            ret.push((
                key,
                HeightMapTile::new(
                    HeightMap::new_map(&self.params, &key).unwrap(),
                    contour_grid_step,
                ),
            ))
        }

        Ok(ret)
    }
}
