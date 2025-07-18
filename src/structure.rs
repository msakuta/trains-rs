use std::collections::{HashMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::{
    train::Train,
    train_tracks::{SegmentDirection, StationId, TrainTask},
    vec2::Vec2,
};

use std::f64::consts::PI;

/// This should depend on the type of the structure.
pub(crate) const STRUCTURE_INPUT_POS: [(Vec2, f64); 3] = [
    (Vec2::new(0., 1.), 0.),
    (Vec2::new(-1., 0.), PI / 2.),
    (Vec2::new(1., 0.), -PI / 2.),
];
pub(crate) const STRUCTURE_OUTPUT_POS: [(Vec2, f64); 3] = [
    (Vec2::new(0., -1.), 0.),
    (Vec2::new(1., 0.), PI / 2.),
    (Vec2::new(-1., 0.), -PI / 2.),
];
pub(crate) const ORE_MINE_CAPACITY: u32 = 10;
const ORE_MINE_FREQUENCY: usize = 120;
pub(crate) const INGOT_CAPACITY: u32 = 20;
pub(crate) const MAX_BELT_LENGTH: f64 = 20.;
pub(crate) const ITEM_INTERVAL: f64 = 1.0;
pub(crate) const BELT_SPEED: f64 = 0.05; // length per tick
pub(crate) const BELT_MAX_SLOPE: f64 = 0.1;

pub(crate) type StructureId = usize;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct Structure {
    pub pos: Vec2<f64>,
    pub ty: StructureType,
    pub iron: u32,
    pub ingot: u32,
    cooldown: usize,
    pub output_belts: [Option<BeltId>; 3],
    pub orientation: f64,
    pub connected_station: Option<(StationId, i32)>,
    pub next_output: u32,
}

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub(crate) enum StructureType {
    #[default]
    OreMine,
    Smelter,
    Sink,
    Loader,
    Unloader,
    Splitter,
    Merger,
}

impl StructureType {
    pub fn input_ports(&self) -> &'static [(Vec2, f64)] {
        const SINK: &[(Vec2, f64)] = &[
            (Vec2::new(0., 4.), 0.),
            (Vec2::new(-1., 4.), 0.),
            (Vec2::new(-2., 4.), 0.),
            (Vec2::new(1., 4.), 0.),
            (Vec2::new(2., 4.), 0.),
        ];
        match self {
            Self::OreMine | Self::Smelter | Self::Loader | Self::Splitter => {
                &STRUCTURE_INPUT_POS[0..1]
            }
            Self::Merger => &STRUCTURE_INPUT_POS[..],
            Self::Sink => SINK,
            Self::Unloader => &[],
        }
    }

    pub fn output_ports(&self) -> &'static [(Vec2, f64)] {
        match self {
            Self::OreMine | Self::Smelter | Self::Unloader | Self::Merger => {
                &STRUCTURE_OUTPUT_POS[0..1]
            }
            Self::Splitter => &STRUCTURE_OUTPUT_POS[..],
            Self::Loader | Self::Sink => &[],
        }
    }
}

impl Structure {
    pub fn new_ore_mine(pos: Vec2, orientation: f64) -> Self {
        Self {
            pos,
            ty: StructureType::OreMine,
            orientation,
            ..Self::default()
        }
    }

    pub fn new_smelter(pos: Vec2, orientation: f64) -> Self {
        Self {
            pos,
            ty: StructureType::Smelter,
            orientation,
            ..Self::default()
        }
    }

    pub fn new_sink(pos: Vec2, orientation: f64) -> Self {
        Self {
            pos,
            ty: StructureType::Sink,
            orientation,
            ..Self::default()
        }
    }

    pub fn new_loader(pos: Vec2, orientation: f64, station: StationId, car_idx: i32) -> Self {
        Self {
            pos,
            ty: StructureType::Loader,
            orientation,
            connected_station: Some((station, car_idx)),
            ..Self::default()
        }
    }

    pub fn new_unloader(pos: Vec2, orientation: f64, station: StationId, car_idx: i32) -> Self {
        Self {
            pos,
            ty: StructureType::Unloader,
            orientation,
            connected_station: Some((station, car_idx)),
            ..Self::default()
        }
    }

    pub fn new_structure(ty: StructureType, pos: Vec2, orientation: f64) -> Self {
        Self {
            pos,
            ty,
            orientation,
            ..Self::default()
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
                    if let Some(belt_id) = self.output_belts.iter().find_map(|belt| belt.as_ref()) {
                        ret.insert_items
                            .push((EntityId::Belt(*belt_id), Item::IronOre));
                    }
                }
            }
            StructureType::Smelter => {
                if self.cooldown == 0 && self.ingot < INGOT_CAPACITY && 0 < self.iron {
                    self.iron -= 1;
                    self.ingot += 1;
                    self.cooldown = ORE_MINE_FREQUENCY;
                } else {
                    self.cooldown = self.cooldown.saturating_sub(1);
                }

                if 0 < self.ingot {
                    if let Some(belt_id) = self.output_belts.iter().find_map(|belt| belt.as_ref()) {
                        ret.insert_items
                            .push((EntityId::Belt(*belt_id), Item::Ingot));
                    }
                }
            }
            StructureType::Sink => {
                if self.cooldown == 0 && 0 < self.ingot {
                    self.ingot -= 1;
                    *score += 1;
                    self.cooldown = 1;
                } else {
                    self.cooldown = self.cooldown.saturating_sub(1);
                }
            }
            StructureType::Loader => {
                if let Some((station, car_idx)) = self.connected_station {
                    if 0 < self.iron {
                        ret.insert_items
                            .push((EntityId::Station(station, car_idx), Item::IronOre));
                    } else if 0 < self.ingot {
                        ret.insert_items
                            .push((EntityId::Station(station, car_idx), Item::Ingot));
                    }
                }
            }
            StructureType::Unloader => {
                if let Some((station, car_idx)) = self.connected_station {
                    // Attempt to unload both item types.
                    // TODO: avoid heap allocations
                    if self.iron < ORE_MINE_CAPACITY {
                        ret.remove_items
                            .push((EntityId::Station(station, car_idx), Item::IronOre));
                    }
                    if self.ingot < INGOT_CAPACITY {
                        ret.remove_items
                            .push((EntityId::Station(station, car_idx), Item::Ingot));
                    }
                }

                if 0 < self.iron {
                    if let Some(belt_id) = self.output_belts.iter().find_map(|belt| belt.as_ref()) {
                        ret.insert_items
                            .push((EntityId::Belt(*belt_id), Item::IronOre));
                    }
                }

                if 0 < self.ingot {
                    if let Some(belt_id) = self.output_belts.iter().find_map(|belt| belt.as_ref()) {
                        ret.insert_items
                            .push((EntityId::Belt(*belt_id), Item::Ingot));
                    }
                }
            }
            StructureType::Splitter => {
                if let Some(belt_id) = self.output_belts[self.next_output as usize] {
                    if 0 < self.iron {
                        ret.insert_items
                            .push((EntityId::Belt(belt_id), Item::IronOre));
                    } else if 0 < self.ingot {
                        ret.insert_items
                            .push((EntityId::Belt(belt_id), Item::Ingot));
                    }
                    self.next_output = (self.next_output + 1) % self.output_belts.len() as u32;
                }
            }
            StructureType::Merger => {
                if let Some(belt_id) = self.output_belts[0] {
                    if 0 < self.iron {
                        ret.insert_items
                            .push((EntityId::Belt(belt_id), Item::IronOre));
                    } else if 0 < self.ingot {
                        ret.insert_items
                            .push((EntityId::Belt(belt_id), Item::Ingot));
                    }
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
                if matches!(
                    self.ty,
                    StructureType::Sink
                        | StructureType::Loader
                        | StructureType::Splitter
                        | StructureType::Merger
                ) && self.ingot < INGOT_CAPACITY
                {
                    self.ingot += 1;
                    return true;
                }
            }
        }
        false
    }

    /// Attempt to remove items that were successfully deleted.
    pub fn post_update(&mut self, remove_iron_ores: i32, remove_ingots: i32) {
        self.iron = (self.iron as i32 - remove_iron_ores).max(0) as u32;
        self.ingot = (self.ingot as i32 - remove_ingots).max(0) as u32;
    }

    /// Returns an iterator for the positions of the input belt connection.
    pub fn input_pos(&self) -> impl Iterator<Item = Vec2> {
        self.ty
            .input_ports()
            .iter()
            .map(|(pos, _)| self.relative_pos(&pos))
    }

    pub fn output_pos(&self) -> impl Iterator<Item = Vec2> {
        self.ty
            .output_ports()
            .iter()
            .map(|(pos, _)| self.relative_pos(&pos))
    }

    fn relative_pos(&self, ofs: &Vec2) -> Vec2 {
        let s = self.orientation.sin();
        let c = self.orientation.cos();
        let x = ofs.x * c - ofs.y * s;
        let y = ofs.x * s + ofs.y * c;
        self.pos + Vec2::new(x, y)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct StructureUpdateResult {
    pub insert_items: Vec<(EntityId, Item)>,
    pub remove_items: Vec<(EntityId, Item)>,
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

/// A reference type that can indicate either a structure, a belt or a train car parked on a station.
#[derive(Clone, Copy, Debug)]
pub(crate) enum EntityId {
    Structure(StructureId),
    Belt(BeltId),
    Station(StationId, i32),
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

    /// Find the structure or belt connection point that is close to the given position.
    /// Structures have different positions for the input and the output, so the parameter `input` affects it.
    pub(crate) fn find_belt_con(
        &self,
        pos: Vec2<f64>,
        search_radius: f64,
        input: bool,
    ) -> (BeltConnection, Vec2<f64>) {
        for (i, structure) in &self.structures {
            if input {
                for (idx, con_pos) in structure.input_pos().enumerate() {
                    let dist2 = (con_pos - pos).length2();
                    if dist2 < search_radius.powi(2) {
                        return (BeltConnection::Structure(*i, idx), con_pos);
                    }
                }
            } else {
                for (idx, con_pos) in structure.output_pos().enumerate() {
                    let dist2 = (con_pos - pos).length2();
                    if dist2 < search_radius.powi(2) {
                        return (BeltConnection::Structure(*i, idx), con_pos);
                    }
                }
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

    pub fn find_by_id(&self, id: StructureId) -> Option<&Structure> {
        self.structures
            .iter()
            .find_map(|(scan_id, st)| if *scan_id == id { Some(st) } else { None })
    }

    pub fn add_structure(&mut self, st: Structure) -> StructureId {
        let ret = self.structure_id_gen;
        self.structures.insert(self.structure_id_gen, st);
        self.structure_id_gen += 1;
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

    pub fn update(&mut self, credits: &mut u32, train: &mut Train) {
        // Update structures
        let st_ids = self.structures.keys().copied().collect::<Vec<_>>();
        for id in st_ids {
            let Some(st) = self.structures.get_mut(&id) else {
                continue;
            };
            let result = st.update(credits);

            let mut moved_iron_ores = 0;
            let mut moved_ingots = 0;
            let mut record_moved = |item, delta: i32| match item {
                Item::IronOre => moved_iron_ores += delta,
                Item::Ingot => moved_ingots += delta,
            };

            // Insert items by structures
            for (dest_id, item) in result.insert_items {
                match dest_id {
                    EntityId::Belt(belt_id) => {
                        if let Some(belt) = self.belts.get_mut(&belt_id) {
                            if belt.try_insert(item) {
                                record_moved(item, 1);
                            }
                        }
                    }
                    EntityId::Structure(st_id) => {
                        if let Some(st) = self.structures.get_mut(&st_id) {
                            if st.try_insert(item) {
                                record_moved(item, 1);
                            }
                        }
                    }
                    EntityId::Station(st_id, car_idx) => {
                        if let TrainTask::Wait(_, wait_station) = train.train_task {
                            let car_idx_u = normalize_car_idx(train, car_idx);
                            if 0 <= car_idx_u
                                && wait_station == st_id
                                && train.try_insert(car_idx_u as usize, item)
                            {
                                record_moved(item, 1);
                            }
                        }
                    }
                }
            }

            // Remove items by structures
            for (dest_id, item) in result.remove_items {
                match dest_id {
                    EntityId::Belt(belt_id) => {
                        if let Some(belt) = self.belts.get_mut(&belt_id) {
                            belt.post_update(1);
                        }
                    }
                    EntityId::Structure(st_id) => {
                        if let Some(st) = self.structures.get_mut(&st_id) {
                            match item {
                                Item::IronOre => st.post_update(1, 0),
                                Item::Ingot => st.post_update(0, 1),
                            }
                        }
                    }
                    EntityId::Station(station_id, car_idx) => {
                        if let TrainTask::Wait(_, wait_station) = train.train_task {
                            let car_idx_u = normalize_car_idx(train, car_idx);
                            if 0 <= car_idx_u
                                && wait_station == station_id
                                && train.try_remove(car_idx_u as usize, item)
                            {
                                record_moved(item, -1);
                            }
                        }
                    }
                }
            }

            // Re-borrow the original structure
            let Some(st) = self.structures.get_mut(&id) else {
                continue;
            };
            st.post_update(moved_iron_ores, moved_ingots);
        }

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

    pub(crate) fn preview_delete(&self, pos: Vec2, search_radius: f64) -> Option<StructureOrBelt> {
        let search_radius2 = search_radius.powi(2);
        if let Some(id) = self.structures.iter().find_map(|(id, st)| {
            if (st.pos - pos).length2() < search_radius2 {
                Some(*id)
            } else {
                None
            }
        }) {
            return Some(StructureOrBelt::Structure(id));
        }

        if let Some(id) = self.belts.iter().find_map(|(id, belt)| {
            let delta = belt.end - belt.start;
            let length = delta.length();
            let parallel = delta / length;
            let perp = parallel.left90();
            // s and t coordinates are along the belt and its perpendicular axis, respectively.
            let s = parallel.dot(pos - belt.start);
            let t = perp.dot(pos - belt.start);
            if 0. < s && s < length && t.abs() < search_radius {
                Some(*id)
            } else {
                None
            }
        }) {
            return Some(StructureOrBelt::Belt(id));
        }

        None
    }

    pub(crate) fn delete(&mut self, pos: Vec2, search_radius: f64) {
        let Some(found) = self.preview_delete(pos, search_radius) else {
            return;
        };
        match found {
            StructureOrBelt::Structure(id) => {
                self.structures.remove(&id);
            }
            StructureOrBelt::Belt(id) => {
                self.belts.remove(&id);
            }
        }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Item {
    IronOre,
    Ingot,
}

fn normalize_car_idx(train: &Train, car_idx: i32) -> i32 {
    let direction = train
        .cars
        .get(0)
        .map_or(SegmentDirection::Forward, |car| car.direction);
    match direction {
        SegmentDirection::Forward => -car_idx,
        SegmentDirection::Backward => car_idx,
    }
}
