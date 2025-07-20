use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use crate::vec2::Vec2;

use super::{EntityId, Item, StructureId, StructureUpdateResult, Structures};

pub(crate) const MAX_BELT_LENGTH: f64 = 20.;
pub(crate) const ITEM_INTERVAL: f64 = 1.0;
pub(crate) const BELT_SPEED: f64 = 0.05; // length per tick
pub(crate) const BELT_MAX_SLOPE: f64 = 0.1;

pub(crate) type BeltId = usize;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Belt {
    pub(crate) start: Vec2<f64>,
    pub(crate) start_con: BeltConnection,
    pub(crate) end: Vec2<f64>,
    pub(crate) end_con: BeltConnection,
    /// Items and their positions, measured from the start.
    /// We should use scrolling trick to reduce updates, but it's an optimization for later.
    pub(crate) items: VecDeque<(Item, f64)>,
}

impl Belt {
    pub fn new(
        start: Vec2<f64>,
        start_con: BeltConnection,
        end: Vec2<f64>,
        end_con: BeltConnection,
    ) -> Self {
        Self {
            start,
            start_con,
            end,
            end_con,
            items: VecDeque::new(),
        }
    }

    pub fn try_insert(&mut self, item: Item) -> bool {
        if self
            .items
            .back()
            .is_none_or(|(_, last)| ITEM_INTERVAL < *last)
        {
            self.items.push_back((item, 0.));
            true
        } else {
            false
        }
    }

    pub fn update(&mut self) -> StructureUpdateResult {
        let mut ret = StructureUpdateResult::new();
        let length = self.length();
        let mut last_pos = length + ITEM_INTERVAL;
        let mut remove_idx = None;
        // We could use retain_mut, but it would be more efficient to pop first n elements, since we know that
        // the elements are only removed from the front.
        for (i, (item, pos)) in self.items.iter_mut().enumerate() {
            if length < *pos + BELT_SPEED {
                match self.end_con {
                    BeltConnection::BeltStart(belt_id) => {
                        ret.insert_items.push((EntityId::Belt(belt_id), *item));
                        remove_idx = Some(i);
                    }
                    BeltConnection::Structure(st, _) => {
                        ret.insert_items.push((EntityId::Structure(st), *item));
                        remove_idx = Some(i);
                    }
                    _ => {}
                }
            }
            *pos = (*pos + BELT_SPEED).min(last_pos - ITEM_INTERVAL);
            last_pos = *pos;
        }

        for _ in 0..remove_idx.unwrap_or(0) {
            self.items.pop_front();
        }
        ret
    }

    /// Attempt to remove items that were successfully deleted.
    pub fn post_update(&mut self, num: usize) {
        for _ in 0..num {
            self.items.pop_front();
        }
    }

    pub fn length(&self) -> f64 {
        (self.start - self.end).length()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum BeltConnection {
    /// An end of a belt that connects to nothing.
    Pos,
    Structure(StructureId, usize),
    /// Belt start can only connect to end and vice versa
    BeltStart(BeltId),
    BeltEnd(BeltId),
}

impl Structures {
    pub(super) fn update_belts(&mut self) {
        // We cannot use iter_mut since we need random access of the elements in belts to transfer items.
        // Technically, this logic depends on the ordering of iteration, which depends on the hashmap.
        // We also need to store the keys into a temporary container to avoid borrow checker, which is not great
        // for performance.
        // We may want stable order by using generational id arena.
        let belt_ids = self.belts.keys().copied().collect::<Vec<_>>();
        for belt_id in belt_ids {
            let Some(belt) = self.belts.get_mut(&belt_id) else {
                continue;
            };
            let res = belt.update();
            let mut moved_items = 0;
            for (dest_id, item) in res.insert_items {
                match dest_id {
                    EntityId::Belt(dest_belt_id) => {
                        let Some(dest) = self.belts.get_mut(&dest_belt_id) else {
                            continue;
                        };
                        moved_items += dest.try_insert(item) as usize;
                    }
                    EntityId::Structure(dest_st_id) => {
                        if let Some(dest) = self.structures.get_mut(&dest_st_id) {
                            moved_items += dest.try_insert(item) as usize;
                        }
                    }
                    _ => {}
                }
            }
            // Re-borrow the original belt
            let Some(belt) = self.belts.get_mut(&belt_id) else {
                continue;
            };
            belt.post_update(moved_items);
        }
    }
}
