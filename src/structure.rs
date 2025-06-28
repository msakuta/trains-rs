use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};

use crate::vec2::Vec2;

pub(crate) const STRUCTURE_SIZE: f64 = 2.;
pub(crate) const ORE_MINE_CAPACITY: u32 = 10;
const ORE_MINE_FREQUENCY: usize = 120;
pub(crate) const INGOT_CAPACITY: u32 = 20;
pub(crate) const MAX_BELT_LENGTH: f64 = 10.;
pub(crate) const ITEM_INTERVAL: f64 = 1.0;
const BELT_SPEED: f64 = 0.05; // length per tick

pub(crate) type StructureId = usize;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Structure {
    pub pos: Vec2<f64>,
    pub ty: StructureType,
    pub iron: u32,
    pub ingot: u32,
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
            ty: StructureType::Sink,
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
                        ret.insert_items
                            .push((StructureOrBelt::Belt(*belt_id), Item::IronOre));
                    }
                    self.iron -= 1;
                }
            }
            StructureType::Smelter => {
                if self.cooldown == 0 && self.ingot < ORE_MINE_CAPACITY && 0 < self.iron {
                    self.iron -= 1;
                    self.ingot += 1;
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

    pub fn try_insert(&mut self, item: Item) -> bool {
        match item {
            Item::IronOre => {
                if self.iron < ORE_MINE_CAPACITY {
                    self.iron += 1;
                    return true;
                }
            }
            Item::Ingot => {
                if matches!(self.ty, StructureType::Sink) && self.ingot < ORE_MINE_CAPACITY {
                    self.ingot += 1;
                    return true;
                }
            }
        }
        false
    }

    /// Attempt to remove items that were successfully deleted.
    pub fn post_update(&mut self, remove_iron_ores: u32, remove_ingots: u32) {
        self.iron = self.iron.saturating_sub(remove_iron_ores);
        self.ingot = self.ingot.saturating_sub(remove_ingots);
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StructureUpdateResult {
    pub insert_items: Vec<(StructureOrBelt, Item)>,
    pub remove_items: Vec<(StructureOrBelt, Item)>,
}

impl StructureUpdateResult {
    pub fn new() -> Self {
        Self {
            insert_items: vec![],
            remove_items: vec![],
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum StructureOrBelt {
    Structure(StructureId),
    Belt(BeltId),
}

/// A collection of structures and belts.
///
/// Since they are deeply related to each other, we group them in the same struct.
/// We may add stations and cargo loader/unloader here, but they may be also a part of TrainTracks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Structures {
    pub structures: HashMap<usize, Structure>,
    pub structure_id_gen: StructureId,
    pub belts: HashMap<BeltId, Belt>,
    pub belt_id_gen: BeltId,
}

impl Structures {
    pub fn new(structures: HashMap<usize, Structure>) -> Self {
        let structure_id_gen = structures.keys().max().map_or(0, |id| id + 1);
        Self {
            structures,
            structure_id_gen,
            belts: HashMap::new(),
            belt_id_gen: 0,
        }
    }

    pub(crate) fn find_belt_con(
        &self,
        pos: Vec2<f64>,
        search_radius: f64,
    ) -> (BeltConnection, Vec2<f64>) {
        for (i, structure) in &self.structures {
            let dist2 = (structure.pos - pos).length2();
            if dist2 < search_radius.powi(2) {
                return (BeltConnection::Structure(*i), structure.pos);
            }
        }
        for (i, belt) in &self.belts {
            let dist2 = (belt.start - pos).length2();
            if dist2 < search_radius.powi(2) {
                return (BeltConnection::BeltStart(*i), belt.start);
            }
            let dist2 = (belt.end - pos).length2();
            if dist2 < search_radius.powi(2) {
                return (BeltConnection::BeltEnd(*i), belt.end);
            }
        }
        (BeltConnection::Pos, pos)
    }

    pub fn add_structure(&mut self, st: Structure) -> StructureId {
        let ret = self.structure_id_gen;
        self.structures.insert(self.belt_id_gen, st);
        self.belt_id_gen += 1;
        ret
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
                        ret.insert_items
                            .push((StructureOrBelt::Belt(belt_id), *item));
                        remove_idx = Some(i);
                    }
                    BeltConnection::Structure(st) => {
                        ret.insert_items
                            .push((StructureOrBelt::Structure(st), *item));
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
    Structure(StructureId),
    /// Belt start can only connect to end and vice versa
    BeltStart(BeltId),
    BeltEnd(BeltId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Item {
    IronOre,
    Ingot,
}
