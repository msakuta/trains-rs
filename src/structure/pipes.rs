use serde::{Deserialize, Serialize};

use crate::vec2::Vec2;

use super::{BeltConnection, EntityId, StructureId};

pub(crate) const MAX_FLUID_AMOUNT: f64 = 100.;
pub(super) const WATER_PUMP_RATE: f64 = 0.1;
pub(super) const PIPE_FLOW_RATE: f64 = 0.2;

pub(crate) type PipeId = usize;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct Pipe {
    pub(crate) start: Vec2<f64>,
    pub(crate) start_con: PipeConnection,
    pub(crate) end: Vec2<f64>,
    pub(crate) end_con: PipeConnection,
    pub(crate) fluid: Option<FluidBox>,
}

impl Pipe {
    pub fn new(
        start: Vec2<f64>,
        start_con: PipeConnection,
        end: Vec2<f64>,
        end_con: PipeConnection,
    ) -> Self {
        Self {
            start,
            start_con,
            end,
            end_con,
            fluid: None,
        }
    }

    pub(super) fn update(&mut self) -> PipeUpdateResult {
        let mut ret = PipeUpdateResult::default();
        if let Some(fluid) = &mut self.fluid {
            let fluid_box = FluidBox {
                amount: (fluid.amount * PIPE_FLOW_RATE)
                    .max(PIPE_FLOW_RATE)
                    .min(fluid.amount),
                ty: fluid.ty,
            };
            let fullness = fluid.amount / MAX_FLUID_AMOUNT;
            for con in [self.start_con, self.end_con] {
                match con {
                    PipeConnection::PipeStart(pipe_id) | PipeConnection::PipeEnd(pipe_id) => {
                        ret.moved_fluids
                            .push((EntityId::Pipe(pipe_id), fluid_box, fullness));
                    }
                    PipeConnection::Structure(st_id, _) => {
                        ret.moved_fluids
                            .push((EntityId::Structure(st_id), fluid_box, fullness));
                    }
                    _ => {}
                }
            }
        }
        ret
    }

    /// Try to insert some amount of fluid. The rate of flow depends on the pressure and amount already in the pipe.
    /// If the pipe has another type of fluid, it will be rejected.
    /// Returns the amount of fluid actually moved.
    pub(super) fn try_insert(
        &mut self,
        incoming_fluid: FluidBox,
        pressure: f64,
    ) -> Option<FluidBox> {
        let Some(fluid_box) = &mut self.fluid else {
            self.fluid = Some(incoming_fluid);
            return Some(incoming_fluid);
        };
        if fluid_box.ty == incoming_fluid.ty && fluid_box.amount < MAX_FLUID_AMOUNT {
            let my_pressure = fluid_box.amount / MAX_FLUID_AMOUNT;
            let delta = pressure - my_pressure;
            if delta < 0. {
                return None;
            }
            let flow = delta * PIPE_FLOW_RATE;
            let after = (fluid_box.amount + flow).min(MAX_FLUID_AMOUNT);
            let ret = FluidBox {
                amount: after - fluid_box.amount,
                ty: fluid_box.ty,
            };
            fluid_box.amount = after;
            return Some(ret);
        }
        None
    }

    /// Attempt to remove items that were successfully deleted.
    pub fn post_update(&mut self, fluid: FluidBox) {
        if let Some(fluid_box) = &mut self.fluid {
            fluid_box.amount = (fluid_box.amount - fluid.amount).max(0.0);
            if fluid_box.amount == 0. {
                self.fluid = None;
            }
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct PipeUpdateResult {
    pub moved_fluids: Vec<(EntityId, FluidBox, f64)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum FluidType {
    Water,
    Steam,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub(crate) struct FluidBox {
    pub amount: f64,
    pub ty: FluidType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) enum PipeConnection {
    /// An end of a pipe that connects to nothing.
    Pos,
    Structure(StructureId, usize),
    /// Pipe start can only connect to end and vice versa
    PipeStart(PipeId),
    PipeEnd(PipeId),
}
