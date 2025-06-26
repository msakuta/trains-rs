use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::vec2::Vec2;

const ORE_MINE_CAPACITY: u32 = 100;
pub(crate) const MAX_BELT_LENGTH: f64 = 10.;

pub(crate) type StructureId = usize;

#[derive(Serialize, Deserialize)]
pub(crate) struct Structure {
    pub pos: Vec2<f64>,
    ty: StructureType,
    iron: u32,
    ingot: u32,
    cooldown: usize,
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
        }
    }

    pub fn new_smelter(pos: Vec2<f64>) -> Self {
        Self {
            pos,
            ty: StructureType::Smelter,
            iron: 0,
            ingot: 0,
            cooldown: 0,
        }
    }

    pub fn new_sink(pos: Vec2<f64>) -> Self {
        Self {
            pos,
            ty: StructureType::Smelter,
            iron: 0,
            ingot: 0,
            cooldown: 0,
        }
    }

    pub fn update(&mut self, score: &mut u32) {
        match self.ty {
            StructureType::OreMine => {
                if self.cooldown == 0 && self.iron < ORE_MINE_CAPACITY {
                    self.iron += 1;
                    self.cooldown = 10;
                } else {
                    self.cooldown = self.cooldown.saturating_sub(1);
                }
            }
            StructureType::Smelter => {
                if self.cooldown == 0 && self.ingot < ORE_MINE_CAPACITY && 0 < self.iron {
                    self.iron -= 1;
                    self.ingot -= 1;
                    self.cooldown = 10;
                } else {
                    self.cooldown = self.cooldown.saturating_sub(1);
                }
            }
            StructureType::Sink => {
                if self.cooldown == 0 && 0 < self.ingot {
                    self.ingot -= 1;
                    *score += 1;
                    self.cooldown = 10;
                } else {
                    self.cooldown = self.cooldown.saturating_sub(1);
                }
            }
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Belts {
    pub belts: HashMap<usize, Belt>,
    pub belt_id_gen: usize,
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
    ) {
        self.belts.insert(
            self.belt_id_gen,
            Belt::new(start_pos, start_con, end_pos, end_con),
        );
        self.belt_id_gen += 1;
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Belt {
    pub(crate) start: Vec2<f64>,
    pub(crate) start_con: BeltConnection,
    pub(crate) end: Vec2<f64>,
    pub(crate) end_con: BeltConnection,
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
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum BeltConnection {
    Pos,
    Structure(StructureId),
}
