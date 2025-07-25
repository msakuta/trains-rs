use super::{
    // PowerWire,
    StructureId,
    Structures,
};
use std::collections::HashSet;

use serde::{Deserialize, Serialize};

pub(crate) const MAX_WIRE_REACH: f64 = 50.;

#[derive(Eq, PartialEq, Hash, Copy, Clone, Serialize, Deserialize, Debug)]
pub(crate) struct PowerWire(pub StructureId, pub StructureId);

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(crate) struct PowerNetwork {
    pub wires: Vec<PowerWire>,
    pub sources: HashSet<StructureId>,
    pub sinks: HashSet<StructureId>,
}

pub(crate) fn build_power_networks(
    structures: &Structures,
    power_wires: &[PowerWire],
) -> Vec<PowerNetwork> {
    let mut left_wires = power_wires.iter().collect::<HashSet<_>>();
    let mut ret = vec![];

    for (&id, s) in structures.structures.iter() {
        if !s.power_sink() && !s.power_source() {
            continue;
        }
        let mut expand_list = HashSet::<StructureId>::new();
        let mut wires = vec![];
        let mut sources = HashSet::new();
        let mut sinks = HashSet::new();
        if s.power_source() {
            sources.insert(id);
        }
        if s.power_sink() {
            sinks.insert(id);
        }

        expand_list.insert(id);

        // Simple Dijkstra
        while !expand_list.is_empty() {
            let mut next_expand = HashSet::<StructureId>::new();
            for id in expand_list {
                if let Some(s) = structures.structures.get(&id) {
                    if s.power_source() {
                        sources.insert(id);
                    }
                    if s.power_sink() {
                        sinks.insert(id);
                    }
                }
                while let Some(wire) = left_wires.iter().find(|w| w.0 == id || w.1 == id).copied() {
                    next_expand.insert(if wire.0 == id { wire.1 } else { wire.0 });
                    left_wires.remove(&wire);
                    wires.push(*wire);
                }
            }
            expand_list = next_expand;
        }

        if !sources.is_empty() && !sinks.is_empty() {
            ret.push(PowerNetwork {
                wires,
                sources,
                sinks,
            });
        }
    }
    ret
}
