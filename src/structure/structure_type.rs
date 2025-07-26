use serde::{Deserialize, Serialize};

use crate::vec2::Vec2;

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
    WaterPump,
    Boiler,
    SteamEngine,
    ElectricPole,
    AtomicBattery,
}

impl StructureType {
    pub fn input_ports(&self) -> &'static [(Vec2, f64)] {
        use StructureType::*;
        const SINK: &[(Vec2, f64)] = &[
            (Vec2::new(0., 4.), 0.),
            (Vec2::new(-1., 4.), 0.),
            (Vec2::new(-2., 4.), 0.),
            (Vec2::new(1., 4.), 0.),
            (Vec2::new(2., 4.), 0.),
        ];
        const BOILER: &[(Vec2, f64)] = &[(Vec2::new(0., 1.), 0.)];
        match self {
            OreMine | Smelter | Loader | Splitter => &STRUCTURE_INPUT_POS[0..1],
            Merger => &STRUCTURE_INPUT_POS[..],
            Sink => SINK,
            Unloader | WaterPump | SteamEngine | ElectricPole | AtomicBattery => &[],
            Boiler => BOILER,
        }
    }

    pub fn output_ports(&self) -> &'static [(Vec2, f64)] {
        use StructureType::*;
        match self {
            OreMine | Smelter | Unloader | Merger => &STRUCTURE_OUTPUT_POS[0..1],
            Splitter => &STRUCTURE_OUTPUT_POS[..],
            Loader | Sink | WaterPump | Boiler | SteamEngine | ElectricPole | AtomicBattery => &[],
        }
    }

    pub fn pipes(&self) -> &'static [(Vec2, f64)] {
        const WATER_PUMP: &[(Vec2, f64)] = &[(Vec2::new(0., -1.), PI)];
        const BOILER: &[(Vec2, f64)] = &[
            (Vec2::new(1., 0.), -PI / 2.),
            (Vec2::new(-1., 0.), PI / 2.),
            (Vec2::new(0., -1.), PI),
        ];
        const STEAM_ENGINE: &[(Vec2, f64)] = &[(Vec2::new(0., 2.), 0.)];
        match self {
            Self::WaterPump => WATER_PUMP,
            Self::Boiler => BOILER,
            Self::SteamEngine => STEAM_ENGINE,
            _ => &[],
        }
    }

    /// Whether this structure can demand power. If true, it can be connected to power poles and be a part of
    /// a power network.
    pub(crate) fn power_sink(&self) -> bool {
        use StructureType::*;
        match self {
            OreMine | Smelter => true,
            Sink | Loader | Unloader | Splitter | Merger | Boiler | WaterPump | SteamEngine
            | ElectricPole | AtomicBattery => false,
        }
    }

    /// Whether this structure can supply power. If true, it can be connected to power poles and be a part of
    /// a power network.
    pub(crate) fn power_source(&self) -> bool {
        use StructureType::*;
        match self {
            SteamEngine => true,
            OreMine | Smelter | Sink | Loader | Unloader | Splitter | Merger | Boiler
            | WaterPump | ElectricPole => false,
            AtomicBattery => true,
        }
    }
}
