mod belts;
mod pipes;
mod power_network;
mod structure_type;

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::{
    train::Train,
    train_tracks::{SegmentDirection, StationId, TrainTask},
    vec2::Vec2,
};

use self::{
    pipes::{PIPE_FLOW_RATE, WATER_PUMP_RATE},
    power_network::{PowerNetwork, build_power_networks},
};

pub(crate) use self::{
    belts::{
        BELT_MAX_SLOPE, BELT_SPEED, Belt, BeltConnection, BeltId, ITEM_INTERVAL, MAX_BELT_LENGTH,
    },
    pipes::{FluidBox, FluidType, MAX_FLUID_AMOUNT, Pipe, PipeConnection, PipeId},
    power_network::{MAX_WIRE_REACH, PowerWire},
    structure_type::StructureType,
};

pub(crate) const ORE_MINE_CAPACITY: i32 = 10;
const ORE_MINE_FREQUENCY: usize = 120;
pub(crate) const INGOT_CAPACITY: i32 = 20;
/// Amount of steam generated by a unit of coal and water.
const STEAM_GEN: f64 = 50.;
const STEAM_PUMP_RATE: f64 = 1.;
/// Amount of energy (kJ) per unit volume of steam.
const POWER_PER_STEAM: f64 = 10.;

pub(crate) type StructureId = usize;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct Structure {
    pub pos: Vec2<f64>,
    pub ty: StructureType,
    pub process: StructureProcess,
    pub inventory: Inventory,
    cooldown: usize,
    pub output_belts: [Option<BeltId>; 3],
    pub connected_pipes: Option<PipeId>,
    pub orientation: f64,
    pub connected_station: Option<(StationId, i32)>,
    pub next_output: u32,
    pub ore_type: Option<OreType>,
    pub input_fluid: Option<FluidBox>,
    pub output_fluid: Option<FluidBox>,
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

    /// Returns the electricity that would be consumed. Positive means demand and negative means supply.
    pub fn power_demand_supply(&self) -> f64 {
        use StructureType::*;
        match self.ty {
            OreMine | Smelter => {
                if matches!(self.process, StructureProcess::Produce(_, _)) {
                    1.
                } else {
                    0.
                }
            }
            Sink | Loader | Unloader | Splitter | Merger | Boiler | ElectricPole => 0.,
            WaterPump => 0.5,
            SteamEngine => {
                if let Some(input) = &self.input_fluid
                    && input.ty == FluidType::Steam
                {
                    let after = (input.amount - 1.).max(0.);
                    (after - input.amount) * POWER_PER_STEAM
                } else {
                    0.
                }
            }
            // Atomic batteries provides perpetual power source without consumption. We may add half-life decay,
            // but it is thought to have very long half-life that we can assume it's a constant.
            AtomicBattery => 1.,
        }
    }

    /// Whether this structure can demand power. If true, it can be connected to power poles and be a part of
    /// a power network.
    fn power_sink(&self) -> bool {
        self.ty.power_sink()
    }

    /// Whether this structure can supply power. If true, it can be connected to power poles and be a part of
    /// a power network.
    fn power_source(&self) -> bool {
        self.ty.power_source()
    }

    /// Update this entity for one tick. It may try to send items to another by returning such a result.
    /// It takes `power_stats` argument, which is the statistics of local power network connected to this entity.
    pub fn update(
        &mut self,
        score: &mut u32,
        power_stats: Option<&PowerStats>,
    ) -> StructureUpdateResult {
        let mut ret = StructureUpdateResult {
            insert_items: vec![],
            remove_items: vec![],
            fluids: vec![],
            gen_power: 0.,
        };
        match self.ty {
            StructureType::OreMine => {
                if matches!(self.process, StructureProcess::None)
                    && self.inventory.sum() < ORE_MINE_CAPACITY
                {
                    let item = self.ore_type.map(|ty| match ty {
                        OreType::Iron => Item::IronOre,
                        OreType::Coal => Item::Coal,
                    });
                    self.process = match item {
                        Some(item) => StructureProcess::Produce(item, ORE_MINE_FREQUENCY as f64),
                        None => StructureProcess::None,
                    };
                } else if let StructureProcess::Produce(item, ref mut time) = self.process
                    && let Some(power_stats) = power_stats
                {
                    if *time < power_stats.sufficiency {
                        match item {
                            Item::IronOre => self.inventory.iron += 1,
                            Item::Coal => self.inventory.coal += 1,
                            _ => {}
                        }
                        self.process = StructureProcess::None;
                    } else {
                        *time -= power_stats.sufficiency;
                    }
                }

                if let Some(belt_id) = self.output_belts.iter().find_map(|belt| belt.as_ref()) {
                    match self.ore_type {
                        Some(OreType::Iron) => {
                            if 0 < self.inventory.iron {
                                ret.insert_items
                                    .push((EntityId::Belt(*belt_id), Item::IronOre));
                            }
                        }
                        Some(OreType::Coal) => {
                            if 0 < self.inventory.coal {
                                ret.insert_items
                                    .push((EntityId::Belt(*belt_id), Item::Coal));
                            }
                        }
                        _ => {}
                    }
                }
            }
            StructureType::Smelter => {
                if matches!(self.process, StructureProcess::None) && 0 < self.inventory.iron {
                    self.inventory.iron -= 1;
                    self.process =
                        StructureProcess::Produce(Item::Ingot, ORE_MINE_FREQUENCY as f64);
                } else if let StructureProcess::Produce(_, ref mut work) = self.process
                    && let Some(power_stats) = power_stats
                {
                    if *work < power_stats.sufficiency && self.inventory.sum() < ORE_MINE_CAPACITY {
                        self.inventory.ingot += 1;
                        self.process = StructureProcess::None;
                    } else {
                        *work -= power_stats.sufficiency;
                    }
                }

                if 0 < self.inventory.ingot {
                    if let Some(belt_id) = self.output_belts.iter().find_map(|belt| belt.as_ref()) {
                        ret.insert_items
                            .push((EntityId::Belt(*belt_id), Item::Ingot));
                    }
                }
            }
            StructureType::Sink => {
                if self.cooldown == 0 && 0 < self.inventory.ingot {
                    self.inventory.ingot -= 1;
                    *score += 1;
                    self.cooldown = 1;
                } else {
                    self.cooldown = self.cooldown.saturating_sub(1);
                }
            }
            StructureType::Loader => {
                if let Some((station, car_idx)) = self.connected_station {
                    if 0 < self.inventory.iron {
                        ret.insert_items
                            .push((EntityId::Station(station, car_idx), Item::IronOre));
                    } else if 0 < self.inventory.ingot {
                        ret.insert_items
                            .push((EntityId::Station(station, car_idx), Item::Ingot));
                    }
                }
            }
            StructureType::Unloader => {
                if let Some((station, car_idx)) = self.connected_station {
                    // Attempt to unload both item types.
                    // TODO: avoid heap allocations
                    if self.inventory.iron < ORE_MINE_CAPACITY {
                        ret.remove_items
                            .push((EntityId::Station(station, car_idx), Item::IronOre));
                    }
                    if self.inventory.ingot < INGOT_CAPACITY {
                        ret.remove_items
                            .push((EntityId::Station(station, car_idx), Item::Ingot));
                    }
                }

                if 0 < self.inventory.iron {
                    if let Some(belt_id) = self.output_belts.iter().find_map(|belt| belt.as_ref()) {
                        ret.insert_items
                            .push((EntityId::Belt(*belt_id), Item::IronOre));
                    }
                }

                if 0 < self.inventory.ingot {
                    if let Some(belt_id) = self.output_belts.iter().find_map(|belt| belt.as_ref()) {
                        ret.insert_items
                            .push((EntityId::Belt(*belt_id), Item::Ingot));
                    }
                }
            }
            StructureType::Splitter => {
                if let Some(belt_id) = self.output_belts[self.next_output as usize] {
                    if 0 < self.inventory.iron {
                        ret.insert_items
                            .push((EntityId::Belt(belt_id), Item::IronOre));
                    } else if 0 < self.inventory.ingot {
                        ret.insert_items
                            .push((EntityId::Belt(belt_id), Item::Ingot));
                    } else if 0 < self.inventory.coal {
                        ret.insert_items.push((EntityId::Belt(belt_id), Item::Coal));
                    }
                } else {
                    // Rotate when the belt is not connected
                    self.next_output = (self.next_output + 1) % self.output_belts.len() as u32;
                }
            }
            StructureType::Merger => {
                if let Some(belt_id) = self.output_belts[0] {
                    if 0 < self.inventory.iron {
                        ret.insert_items
                            .push((EntityId::Belt(belt_id), Item::IronOre));
                    } else if 0 < self.inventory.ingot {
                        ret.insert_items
                            .push((EntityId::Belt(belt_id), Item::Ingot));
                    }
                }
            }
            StructureType::WaterPump => {
                if let Some(output) = &mut self.output_fluid {
                    if matches!(output.ty, FluidType::Water) {
                        output.amount = (output.amount + WATER_PUMP_RATE).min(MAX_FLUID_AMOUNT);
                    }
                } else {
                    self.output_fluid = Some(FluidBox {
                        ty: FluidType::Water,
                        amount: WATER_PUMP_RATE,
                    })
                }

                if let Some((fluid_box, pipe_id)) = self.output_fluid.zip(self.connected_pipes) {
                    ret.fluids.push((
                        EntityId::Pipe(pipe_id),
                        FluidBox {
                            ty: fluid_box.ty,
                            amount: fluid_box.amount.min(PIPE_FLOW_RATE),
                        },
                    ))
                }
            }
            StructureType::Boiler => {
                if 0 < self.inventory.coal
                    && let Some(input) = &mut self.input_fluid
                    && 1. <= input.amount
                    && input.ty == FluidType::Water
                    && self.output_fluid.is_none_or(|f| {
                        f.ty == FluidType::Steam && f.amount <= MAX_FLUID_AMOUNT - STEAM_GEN
                    })
                {
                    self.inventory.coal -= 1;
                    input.amount -= 1.;
                    if let Some(output) = &mut self.output_fluid {
                        output.amount += STEAM_GEN;
                    } else {
                        self.output_fluid = Some(FluidBox {
                            ty: FluidType::Steam,
                            amount: STEAM_GEN,
                        });
                    }
                }

                if let Some((fluid_box, pipe_id)) = self.output_fluid.zip(self.connected_pipes) {
                    ret.fluids.push((
                        EntityId::Pipe(pipe_id),
                        FluidBox {
                            ty: fluid_box.ty,
                            amount: fluid_box.amount.min(PIPE_FLOW_RATE),
                        },
                    ))
                }
            }
            StructureType::SteamEngine => {
                if let Some(input) = &mut self.input_fluid
                    && input.ty == FluidType::Steam
                    && let Some(power_stats) = power_stats
                {
                    let after = (input.amount - power_stats.load).max(0.);
                    ret.gen_power += (after - input.amount) * POWER_PER_STEAM;
                    input.amount = after;
                }
            }
            StructureType::ElectricPole => {}
            StructureType::AtomicBattery => {
                ret.gen_power += 1.;
            }
        }
        ret
    }

    pub fn try_insert(&mut self, item: Item) -> bool {
        match item {
            Item::IronOre => {
                // Don't fill the entire buffer, since some structures needs output
                if self.inventory.sum() < ORE_MINE_CAPACITY - 2 {
                    self.inventory.iron += 1;
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
                ) && self.inventory.sum() < ORE_MINE_CAPACITY
                {
                    self.inventory.ingot += 1;
                    return true;
                }
            }
            Item::Coal => {
                if self.inventory.sum() < ORE_MINE_CAPACITY {
                    self.inventory.coal += 1;
                    return true;
                }
            }
        }
        false
    }

    /// Try to insert some amount of fluid. It can be rejected if the fluid type is incompatible.
    /// Pressure is ignored in the current implementation.
    /// Returns the amount of fluid actually moved.
    pub fn try_insert_fluid(
        &mut self,
        incoming_fluid: FluidBox,
        _pressure: f64,
    ) -> Option<FluidBox> {
        if let Some(input) = &mut self.input_fluid {
            if input.ty == incoming_fluid.ty && input.amount < MAX_FLUID_AMOUNT {
                let flow =
                    (incoming_fluid.amount * STEAM_PUMP_RATE).min(MAX_FLUID_AMOUNT - input.amount);
                input.amount += flow;
                return Some(FluidBox {
                    ty: incoming_fluid.ty,
                    amount: flow,
                });
            }
        } else if matches!(self.ty, StructureType::Boiler) && incoming_fluid.ty == FluidType::Water
        {
            // Accepts only water
            self.input_fluid = Some(incoming_fluid);
            return Some(incoming_fluid);
        } else if matches!(self.ty, StructureType::SteamEngine)
            && incoming_fluid.ty == FluidType::Steam
        {
            // Accepts only steam
            self.input_fluid = Some(incoming_fluid);
            return Some(incoming_fluid);
        }
        None
    }

    /// Attempt to remove items that were successfully deleted, after the receiver confirmed.
    pub fn post_update(&mut self, remove_inventory: Inventory, remove_fluids: Vec<FluidBox>) {
        self.inventory.iron = self.inventory.iron.saturating_sub(remove_inventory.iron);
        self.inventory.ingot = self.inventory.ingot.saturating_sub(remove_inventory.ingot);
        self.inventory.coal = self.inventory.coal.saturating_sub(remove_inventory.coal);

        if matches!(self.ty, StructureType::Splitter) {
            // Only rotate the output belt when it sends in attempt to make it as even as possible
            for _ in 0..remove_inventory.sum() {
                self.next_output = (self.next_output + 1) % self.output_belts.len() as u32;
            }
        }

        for fluid in remove_fluids {
            if let Some(output) = &mut self.output_fluid
                && output.ty == fluid.ty
            {
                // It is a logical error that the amount of removed fluid exceeds the existing, but we make sure to
                // not have negative amount.
                output.amount = (output.amount - fluid.amount).max(0.);
                if output.amount == 0. {
                    self.output_fluid = None;
                }
            }
        }
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

    pub fn pipe_pos(&self) -> impl Iterator<Item = Vec2> {
        self.ty
            .pipes()
            .iter()
            .map(|(pos, _)| self.relative_pos(pos))
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
    pub fluids: Vec<(EntityId, FluidBox)>,
    pub gen_power: f64,
}

impl StructureUpdateResult {
    pub fn new() -> Self {
        Self {
            insert_items: vec![],
            remove_items: vec![],
            fluids: vec![],
            gen_power: 0.,
        }
    }
}

/// A reference type that can indicate either a structure, a belt or a train car parked on a station.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub(crate) enum EntityId {
    Structure(StructureId),
    Belt(BeltId),
    Pipe(PipeId),
    Station(StationId, i32),
    Wire(StructureId, StructureId),
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
    pub pipes: HashMap<PipeId, Pipe>,
    pub pipe_id_gen: PipeId,
    pub power_wires: Vec<PowerWire>,
    power_networks: Vec<PowerNetwork>,
}

impl Structures {
    pub fn new(structures: HashMap<usize, Structure>) -> Self {
        let structure_id_gen = structures.keys().max().map_or(0, |id| id + 1);
        Self {
            structures,
            structure_id_gen,
            belts: HashMap::new(),
            belt_id_gen: 0,
            pipes: HashMap::new(),
            pipe_id_gen: 0,
            power_wires: vec![],
            power_networks: vec![],
        }
    }

    pub fn update_power_network(&mut self) {
        self.power_networks = build_power_networks(self, &self.power_wires);
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

    pub(crate) fn find_pipe_con(
        &self,
        pos: Vec2<f64>,
        search_radius: f64,
    ) -> (PipeConnection, Vec2<f64>) {
        for (i, structure) in &self.structures {
            for (idx, con_pos) in structure.pipe_pos().enumerate() {
                let dist2 = (con_pos - pos).length2();
                if dist2 < search_radius.powi(2) {
                    return (PipeConnection::Structure(*i, idx), con_pos);
                }
            }
        }
        for (i, pipe) in &self.pipes {
            let dist2 = (pipe.start - pos).length2();
            if dist2 < search_radius.powi(2) {
                return (PipeConnection::PipeStart(*i), pipe.start);
            }
            let dist2 = (pipe.end - pos).length2();
            if dist2 < search_radius.powi(2) {
                return (PipeConnection::PipeEnd(*i), pipe.end);
            }
        }
        (PipeConnection::Pos, pos)
    }

    pub fn find_structure(&self, pos: Vec2, search_radius: f64) -> Option<StructureId> {
        self.structures
            .iter()
            .find(|(_, s)| (s.pos - pos).length2() < search_radius.powi(2))
            .map(|(id, _)| *id)
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
        self.update_power_network();
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

    pub fn add_pipe(
        &mut self,
        start_pos: Vec2,
        start_con: PipeConnection,
        end_pos: Vec2,
        end_con: PipeConnection,
    ) -> PipeId {
        let ret = self.pipe_id_gen;
        self.pipes.insert(
            self.pipe_id_gen,
            Pipe::new(start_pos, start_con, end_pos, end_con),
        );
        self.pipe_id_gen += 1;
        ret
    }

    pub fn add_wire(&mut self, start: StructureId, end: StructureId) {
        if !self
            .power_wires
            .iter()
            .any(|wire| wire.0 == start && wire.1 == end)
        {
            self.power_wires.push(PowerWire(start, end));
            self.update_power_network();
        }
    }

    /// Give opportunities to all entities for updates.
    /// Entities include structures, belts and pipes.
    /// Returns power sufficiency stats for printing.
    pub fn update(&mut self, credits: &mut u32, train: &mut Train) -> PowerStats {
        let power_network_stats: Vec<_> = self
            .power_networks
            .iter()
            .map(|pn| {
                let (power_demand, power_supply) = self
                    .structures
                    .iter()
                    // For debugging, supply with a little power by default
                    .fold((0., 1.), |(mut demand, mut supply), cur| {
                        if !pn.sources.contains(cur.0) && pn.sinks.contains(cur.0) {
                            return (demand, supply);
                        }
                        let val = cur.1.power_demand_supply();
                        if val < 0. {
                            supply += -val;
                        } else {
                            demand += val;
                        }
                        (demand, supply)
                    });
                PowerStats::new(power_demand, power_supply)
            })
            .collect();

        // Update structures
        let st_ids = self.structures.keys().copied().collect::<Vec<_>>();
        for id in st_ids {
            let Some(st) = self.structures.get_mut(&id) else {
                continue;
            };
            let power_stats = self
                .power_networks
                .iter()
                .enumerate()
                .find(|(_, pn)| pn.sinks.contains(&id) || pn.sources.contains(&id))
                .and_then(|(i, _)| power_network_stats.get(i));
            let result = st.update(credits, power_stats);

            let mut moved_iron_ores = 0;
            let mut moved_ingots = 0;
            let mut moved_coal = 0;
            let mut record_moved = |item, delta: i32| match item {
                Item::IronOre => moved_iron_ores += delta,
                Item::Ingot => moved_ingots += delta,
                Item::Coal => moved_coal += delta,
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
                    EntityId::Pipe(_) => {
                        // Pipes cannot receive items.
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
                    EntityId::Wire(_, _) => {
                        // Items cannot be added to wires
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
                    EntityId::Pipe(_) => {
                        // Pipes cannot remove items.
                    }
                    EntityId::Structure(st_id) => {
                        if let Some(st) = self.structures.get_mut(&st_id) {
                            match item {
                                Item::IronOre => {
                                    st.post_update(Inventory::default().with_iron(1), vec![])
                                }
                                Item::Ingot => {
                                    st.post_update(Inventory::default().with_ingot(1), vec![])
                                }
                                Item::Coal => {
                                    st.post_update(Inventory::default().with_coal(1), vec![])
                                }
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
                    EntityId::Wire(_, _) => {
                        // Items cannot be removed from wires
                    }
                }
            }

            let mut moved_fluids = vec![];
            for (dest_id, fluid) in result.fluids {
                match dest_id {
                    EntityId::Pipe(pipe_id) => {
                        if let Some(pipe) = self.pipes.get_mut(&pipe_id) {
                            // Structure outputs have the highest pressure so that it will never flow back.
                            // Imagine a pump is actively pushing it out.
                            moved_fluids.extend(pipe.try_insert(fluid, 1.));
                        }
                    }
                    _ => {
                        // Structures can only output fluids to a pipe.
                    }
                }
            }

            // Re-borrow the original structure
            let Some(st) = self.structures.get_mut(&id) else {
                continue;
            };
            st.post_update(
                Inventory {
                    iron: moved_iron_ores,
                    ingot: moved_ingots,
                    coal: moved_coal,
                },
                moved_fluids,
            );
        }
        self.update_belts();
        self.update_pipes();
        power_network_stats
            .iter()
            .fold(PowerStats::default(), |acc, cur| {
                PowerStats::new(acc.demand + cur.demand, acc.supply + cur.supply)
            })
    }

    pub(crate) fn preview_delete(&self, pos: Vec2, search_radius: f64) -> Option<EntityId> {
        let search_radius2 = search_radius.powi(2);
        if let Some(id) = self.structures.iter().find_map(|(id, st)| {
            if (st.pos - pos).length2() < search_radius2 {
                Some(*id)
            } else {
                None
            }
        }) {
            return Some(EntityId::Structure(id));
        }

        let find_close_edge = |start: Vec2, end: Vec2| {
            let delta = end - start;
            let length = delta.length();
            let parallel = delta / length;
            let perp = parallel.left90();
            // s and t coordinates are along the belt and its perpendicular axis, respectively.
            let s = parallel.dot(pos - start);
            let t = perp.dot(pos - start);
            0. < s && s < length && t.abs() < search_radius
        };

        if let Some((id, _)) = self
            .belts
            .iter()
            .find(|(_, belt)| find_close_edge(belt.start, belt.end))
        {
            return Some(EntityId::Belt(*id));
        }

        if let Some((id, _)) = self
            .pipes
            .iter()
            .find(|(_, pipe)| find_close_edge(pipe.start, pipe.end))
        {
            return Some(EntityId::Pipe(*id));
        }

        if let Some(wire) = self.power_wires.iter().find(|wire| {
            let Some(start) = self.structures.get(&wire.0) else {
                return false;
            };
            let Some(end) = self.structures.get(&wire.1) else {
                return false;
            };
            find_close_edge(start.pos, end.pos)
        }) {
            return Some(EntityId::Wire(wire.0, wire.1));
        }

        None
    }

    pub(crate) fn delete(&mut self, pos: Vec2, search_radius: f64) {
        let Some(found) = self.preview_delete(pos, search_radius) else {
            return;
        };
        match found {
            EntityId::Structure(id) => {
                self.structures.remove(&id);
                self.update_power_network();
            }
            EntityId::Belt(id) => {
                self.belts.remove(&id);
            }
            EntityId::Pipe(id) => {
                self.pipes.remove(&id);
            }
            EntityId::Wire(from, to) => {
                let from_and_to = [from, to];
                self.power_wires.retain(|wire| {
                    !from_and_to.contains(&wire.0) || !from_and_to.contains(&wire.1)
                });
                self.update_power_network();
            }
            _ => {}
        }
    }
}

#[derive(Copy, Clone, Default)]
pub(crate) struct PowerStats {
    pub demand: f64,
    pub supply: f64,
    pub sufficiency: f64,
    pub load: f64,
}

impl PowerStats {
    fn new(demand: f64, supply: f64) -> Self {
        Self {
            demand,
            supply,
            sufficiency: if demand != 0. {
                (supply / demand).min(1.)
            } else {
                0.
            },
            load: if supply != 0. {
                (demand / supply).min(1.)
            } else {
                0.
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum Item {
    IronOre,
    Ingot,
    Coal,
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

#[derive(Clone, Copy, Debug)]
pub(crate) struct OreVein {
    pub pos: Vec2,
    pub ty: OreType,
    pub occupied_miner: Option<StructureId>,
}

impl OreVein {
    pub fn new(pos: Vec2, ty: OreType) -> Self {
        Self {
            pos,
            ty,
            occupied_miner: None,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) enum OreType {
    Iron,
    Coal,
}

/// Inventory amount is signed because it is also used to represent deltas.
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub(crate) struct Inventory {
    pub iron: i32,
    pub ingot: i32,
    pub coal: i32,
}

impl Inventory {
    pub fn sum(&self) -> i32 {
        self.iron + self.ingot + self.coal
    }

    pub fn with_iron(mut self, count: i32) -> Self {
        self.iron += count;
        self
    }

    pub fn with_ingot(mut self, count: i32) -> Self {
        self.ingot += count;
        self
    }

    pub fn with_coal(mut self, count: i32) -> Self {
        self.coal += count;
        self
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) enum StructureProcess {
    #[default]
    None,
    Produce(Item, f64),
}
