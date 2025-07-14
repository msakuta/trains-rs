use std::collections::{HashSet, VecDeque};

use super::{HeightMap, HeightMapKey, HeightMapParams, HeightMapTile};

/// The tile generator that cannot run tasks in parallel threads.
/// It has very low performance but works in Wasm.
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
        // We use the main thread but proceed with tile generation one at a time, in order to avoid blocking the main
        // thread for too long. However, it can be too conservative and the progress may be slower than necessary, or
        // vice versa. The optimal amount of progress per frame should be determined by the platform parallelism, which
        // is what parallel_tile_gen does.
        if let Some((key, contour_grid_step)) = self.gen_queue.pop_back() {
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
