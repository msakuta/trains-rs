use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::vec2::Vec2;

const ORE_MINE_CAPACITY: u32 = 100;
const ORE_MINE_FREQUENCY: usize = 120;
pub(crate) const MAX_BELT_LENGTH: f64 = 10.;
pub(crate) const ITEM_INTERVAL: f64 = 1.0;
const BELT_SPEED: f64 = 0.05; // length per tick

pub(crate) type StructureId = usize;

#[derive(Serialize, Deserialize)]
pub(crate) struct Structure {
    pub pos: Vec2<f64>,
    ty: StructureType,
    iron: u32,
    ingot: u32,
    cooldown: usize,
    pub output_belts: HashSet<BeltId>,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) enum StructureType {
    OreMine,
    Smelter,
    Sink,
}

impl Structure {
    pub fn new_ore_mine(pos: Vec2<f64>) -> Self {
        Self {
            pos,
            ty: StructureType::OreMine,
            iron: 0,
            ingot: 0,
            cooldown: 0,
            output_belts: HashSet::new(),
        }
    }

    pub fn new_smelter(pos: Vec2<f64>) -> Self {
        Self {
            pos,
            ty: StructureType::Smelter,
            iron: 0,
            ingot: 0,
            cooldown: 0,
            output_belts: HashSet::new(),
        }
    }

    pub fn new_sink(pos: Vec2<f64>) -> Self {
        Self {
            pos,
            ty: StructureType::Smelter,
            iron: 0,
            ingot: 0,
            cooldown: 0,
            output_belts: HashSet::new(),
        }
    }

    pub fn update(&mut self, score: &mut u32) -> StructureUpdateResult {
        let mut ret = StructureUpdateResult {
            insert_items: vec![],
            remove_items: vec![],
        };
        match self.ty {
            StructureType::OreMine => {
                if self.cooldown == 0 && self.iron < ORE_MINE_CAPACITY {
                    self.iron += 1;
                    self.cooldown = ORE_MINE_FREQUENCY;
                } else {
                    self.cooldown = self.cooldown.saturating_sub(1);
                }

                if 0 < self.iron {
                    if let Some(belt_id) = self.output_belts.iter().next() {
                        ret.insert_items.push((*belt_id, Item::IronOre));
                    }
                    self.iron -= 1;
                }
            }
            StructureType::Smelter => {
                if self.cooldown == 0 && self.ingot < ORE_MINE_CAPACITY && 0 < self.iron {
                    self.iron -= 1;
                    self.ingot -= 1;
                    self.cooldown = ORE_MINE_FREQUENCY;
                } else {
                    self.cooldown = self.cooldown.saturating_sub(1);
                }
            }
            StructureType::Sink => {
                if self.cooldown == 0 && 0 < self.ingot {
                    self.ingot -= 1;
                    *score += 1;
                    self.cooldown = ORE_MINE_FREQUENCY;
                } else {
                    self.cooldown = self.cooldown.saturating_sub(1);
                }
            }
        }
        ret
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StructureUpdateResult {
    pub insert_items: Vec<(BeltId, Item)>,
    pub remove_items: Vec<(BeltId, Item)>,
}

impl StructureUpdateResult {
    pub fn new() -> Self {
        Self {
            insert_items: vec![],
            remove_items: vec![],
        }
    }

    pub fn merge(&mut self, other: Self) {
        self.insert_items.extend_from_slice(&other.insert_items);
        self.remove_items.extend_from_slice(&other.remove_items);
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Belts {
    pub belts: HashMap<BeltId, Belt>,
    pub belt_id_gen: BeltId,
}

impl Belts {
    pub fn new() -> Self {
        Self {
            belts: HashMap::new(),
            belt_id_gen: 0,
        }
    }

    pub fn add_belt(
        &mut self,
        start_pos: Vec2,
        start_con: BeltConnection,
        end_pos: Vec2,
        end_con: BeltConnection,
    ) -> BeltId {
        let ret = self.belt_id_gen;
        self.belts.insert(
            self.belt_id_gen,
            Belt::new(start_pos, start_con, end_pos, end_con),
        );
        self.belt_id_gen += 1;
        ret
    }
}

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
                        ret.insert_items.push((belt_id, *item));
                        remove_idx = Some(i);
                    }
                    BeltConnection::Structure(st) => {
                        ret.insert_items.push((st, *item));
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
    Pos,
    Structure(StructureId),
    /// Belt start can only connect to end and vice versa
    BeltStart(BeltId),
    BeltEnd(BeltId),
}

pub(crate) struct BeltUpdateResult {
    items: Vec<Item>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Item {
    IronOre,
    Ingot,
}
