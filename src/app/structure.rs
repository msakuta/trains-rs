use eframe::{
    egui::{self, Color32, Painter, Pos2, Rect, pos2, vec2},
    epaint::PathShape,
};
use ordered_float::NotNan;

use crate::{
    structure::{
        BELT_MAX_SLOPE, BeltConnection, INGOT_CAPACITY, ITEM_INTERVAL, Item, MAX_BELT_LENGTH,
        ORE_MINE_CAPACITY, Structure, StructureOrBelt, StructureType,
    },
    transform::PaintTransform,
    vec2::Vec2,
};

use super::{HeightMap, TrainsApp};

pub(super) const STRUCTURE_ICON_SIZE: f32 = 10.;
const SELECT_THRESHOLD: f64 = 10.;
/// The distance to determine if the mouse cursor is close enough to an ore vein
const ORE_MINE_DISTANCE_THRESHOLD: f64 = 30.;

impl TrainsApp {
    pub(super) fn render_structures(&self, painter: &Painter, paint_transform: &PaintTransform) {
        for (_, structure) in &self.structures.structures {
            let pos = paint_transform.to_pos2(structure.pos);
            Self::render_structure(
                pos,
                structure.orientation,
                false,
                structure.ty,
                painter,
                paint_transform,
            );
        }

        let paint_bars = |st: &Structure| {
            let base_pos = paint_transform.to_pos2(st.pos).to_vec2();
            const BAR_WIDTH: f32 = 50.;
            const BAR_HEIGHT: f32 = 5.;
            const BAR_STRIDE: f32 = 8.; // We want some gaps to differentiate them
            const BAR_OFFSET: f32 = 30.;
            for (y, fullness, color) in [
                (
                    0,
                    st.iron as f32 / ORE_MINE_CAPACITY as f32,
                    Color32::from_rgb(255, 255, 0),
                ),
                (1, st.ingot as f32 / INGOT_CAPACITY as f32, Color32::GREEN),
            ] {
                let y_pos = base_pos.y + BAR_OFFSET + y as f32 * BAR_STRIDE;
                painter.rect_filled(
                    Rect::from_center_size(pos2(base_pos.x, y_pos), vec2(BAR_WIDTH, BAR_HEIGHT)),
                    0.,
                    Color32::BLACK,
                );

                painter.rect_filled(
                    Rect::from_min_size(
                        pos2(base_pos.x - BAR_WIDTH / 2., y_pos - BAR_HEIGHT / 2.),
                        vec2(fullness * BAR_WIDTH, BAR_HEIGHT),
                    ),
                    0.,
                    color,
                );
            }

            Some(())
        };

        if 2. < self.transform.scale() {
            for (_, structure) in &self.structures.structures {
                paint_bars(structure);
            }
        }
    }

    pub(super) fn render_belts(&self, painter: &Painter, paint_transform: &PaintTransform) {
        let scale = self.transform.scale();
        for belt in self.structures.belts.values() {
            let start = paint_transform.to_pos2(belt.start);
            let end = paint_transform.to_pos2(belt.end);
            painter.arrow(start, end - start, (2., Color32::BLUE));

            // Render items only when they are likely more than 1 pixels
            if ITEM_INTERVAL < scale as f64 {
                let length = (belt.end - belt.start).length();

                for (item, dist) in &mut belt.items.iter() {
                    let f = *dist / length;
                    let pos = belt.start * (1. - f) + belt.end * f;
                    match item {
                        Item::IronOre => {
                            painter.circle_filled(
                                paint_transform.to_pos2(pos),
                                (ITEM_INTERVAL * 0.5) as f32 * scale,
                                Color32::BLUE,
                            );
                        }
                        Item::Ingot => {
                            painter.rect_filled(
                                Rect::from_center_size(
                                    paint_transform.to_pos2(pos),
                                    egui::Vec2::splat((ITEM_INTERVAL * 0.75) as f32 * scale),
                                ),
                                0.,
                                Color32::BLUE,
                            );
                        }
                    }
                }
            }
        }
    }

    /// A common rendering logic for the built structures and preview ghosts.
    pub(super) fn render_structure(
        pos: Pos2,
        orient: f64,
        preview: bool,
        ty: StructureType,
        painter: &Painter,
        paint_transform: &PaintTransform,
    ) {
        let color = if preview {
            Color32::from_rgba_premultiplied(255, 127, 191, 191)
        } else {
            match ty {
                StructureType::OreMine => Color32::from_rgb(0, 127, 191),
                StructureType::Smelter => Color32::from_rgb(0, 191, 127),
                StructureType::Sink => Color32::from_rgb(127, 0, 127),
                StructureType::Loader => Color32::from_rgb(127, 127, 0),
                StructureType::Unloader => Color32::from_rgb(0, 63, 127),
            }
        };
        let line_color = Color32::from_rgb(0, 63, 31);

        // Render the real size with enough zoom
        if 2. < paint_transform.scale() {
            let s = orient.sin();
            let c = orient.cos();
            let rotate = |ofs: [f64; 2]| {
                let x = ofs[0] * c - ofs[1] * s;
                let y = ofs[0] * s + ofs[1] * c;
                pos + paint_transform.to_vec2(Vec2::new(x, y))
            };

            painter.add(PathShape::convex_polygon(
                [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]
                    .into_iter()
                    .map(rotate)
                    .collect(),
                color,
                (1., line_color),
            ));

            painter.add(PathShape::convex_polygon(
                [[0., -1.5], [-0.3, -1.2], [0.3, -1.2]]
                    .into_iter()
                    .map(rotate)
                    .collect(),
                color,
                (1., line_color),
            ));

            painter.add(PathShape::convex_polygon(
                [[0., 1.2], [-0.3, 1.5], [0.3, 1.5]]
                    .into_iter()
                    .map(rotate)
                    .collect(),
                color,
                (1., line_color),
            ));
        } else {
            // render icon that does not shink if zoomed out
            painter.rect_filled(
                Rect::from_center_size(pos, vec2(STRUCTURE_ICON_SIZE, STRUCTURE_ICON_SIZE)),
                0.,
                color,
            );
        }
    }

    pub(super) fn find_belt_con(&self, pos: Vec2, input: bool) -> (BeltConnection, Vec2) {
        self.structures
            .find_belt_con(pos, SELECT_THRESHOLD / self.transform.scale() as f64, input)
    }

    pub(super) fn preview_delete_structure(
        &self,
        pointer: Pos2,
        painter: &Painter,
        paint_transform: &PaintTransform,
    ) {
        let pos = paint_transform.from_pos2(pointer);
        let search_radius = SELECT_THRESHOLD / self.transform.scale() as f64;
        if let Some(id) = self.structures.preview_delete(pos, search_radius) {
            match id {
                StructureOrBelt::Structure(id) => {
                    if let Some(st) = self.structures.structures.get(&id) {
                        let pos = paint_transform.to_pos2(st.pos);
                        Self::render_structure(
                            pos,
                            st.orientation,
                            true,
                            st.ty,
                            painter,
                            paint_transform,
                        );
                    }
                }
                StructureOrBelt::Belt(id) => {
                    if let Some(belt) = self.structures.belts.get(&id) {
                        let start = paint_transform.to_pos2(belt.start);
                        let end = paint_transform.to_pos2(belt.end);
                        painter.arrow(start, end - start, (4., Color32::from_rgb(255, 0, 255)));
                    }
                }
            }
        }
    }

    pub(super) fn add_ore_mine(&mut self, pointer_pos: Vec2) -> Result<(), String> {
        let Some(pos) = self.building_structure else {
            if self.heightmap.is_water(&pointer_pos) {
                return Err("Cannot build in water".to_string());
            }
            self.building_structure = Some(pointer_pos);
            return Ok(());
        };
        let delta = pos - pointer_pos;
        let orient = delta.y.atan2(delta.x) - std::f64::consts::PI * 0.5;
        let Some(ore_vein) = self.ore_veins.iter_mut().find(|ov| ov.pos == pos) else {
            return Err("ore vein expected".to_string());
        };
        if ore_vein
            .occupied_miner
            .is_some_and(|id| self.structures.find_by_id(dbg!(id)).is_some())
        {
            return Err("ore vein already occupied".to_string());
        }
        let st_id = self
            .structures
            .add_structure(Structure::new_ore_mine(pos, orient));
        ore_vein.occupied_miner = Some(st_id);
        self.building_structure = None;
        Ok(())
    }

    pub(super) fn preview_ore_mine(
        &mut self,
        pointer: Pos2,
        painter: &Painter,
        paint_transform: &PaintTransform,
    ) {
        let pos = paint_transform.from_pos2(pointer);
        let scan_range2 =
            NotNan::new((ORE_MINE_DISTANCE_THRESHOLD / paint_transform.scale() as f64).powi(2))
                .unwrap();
        if let Some((ore_vein, _)) = self
            .ore_veins
            .iter()
            .map(|ov| (ov, NotNan::new((ov.pos - pos).length2()).unwrap()))
            .filter(|(_, dist2)| *dist2 < scan_range2)
            .min_by_key(|(_, dist2)| *dist2)
        {
            let delta = ore_vein.pos - paint_transform.from_pos2(pointer);
            let orient = delta.y.atan2(delta.x) - std::f64::consts::PI * 0.5;
            Self::render_structure(
                paint_transform.to_pos2(ore_vein.pos),
                orient,
                true,
                StructureType::OreMine,
                &painter,
                &paint_transform,
            );
            self.building_structure = Some(ore_vein.pos);
        } else {
            self.building_structure = None;
        }
    }

    pub(super) fn add_belt(&mut self, pos: Vec2) -> Result<(), String> {
        if let Some((start_con, start_pos)) = &self.belt_connection {
            let (end_con, end_pos) = self.find_belt_con(pos, true);
            // Disallow connection to itself and end-to-end
            if matches!(end_con, BeltConnection::BeltEnd(_)) {
                return Err("You cannot connect the end of a belt to another end".to_string());
            }
            if matches!((end_con, start_con), (BeltConnection::Structure(eid), BeltConnection::Structure(sid)) if eid == *sid)
            {
                return Err("You cannot connect a belt itself".to_string());
            }
            if MAX_BELT_LENGTH.powi(2) <= (end_pos - *start_pos).length2() {
                return Err("Belt is too long".to_string());
            }
            if intersects_water(*start_pos, end_pos, &self.heightmap) {
                return Err("Belt is intersecting water".to_string());
            }
            if exceeds_slope(*start_pos, end_pos, &self.heightmap) {
                return Err("Belt is exceeding maximum slope".to_string());
            }
            let belt_id = self
                .structures
                .add_belt(*start_pos, *start_con, end_pos, end_con);
            match start_con {
                BeltConnection::Structure(start_st) => {
                    if let Some(st) = self.structures.structures.get_mut(start_st) {
                        st.output_belts.insert(belt_id);
                        println!("Added belt {belt_id} to output_belts");
                    }
                }
                // If we connect to a belt, we add a reference to the upstream belt.
                BeltConnection::BeltEnd(upstream_belt) => {
                    if let Some(upstream_belt) = self.structures.belts.get_mut(upstream_belt) {
                        upstream_belt.end_con = BeltConnection::BeltStart(belt_id);
                        println!("Added belt {belt_id} to upstream belt");
                    }
                }
                _ => {}
            }
            self.belt_connection = None;
        } else {
            self.belt_connection = Some(self.find_belt_con(pos, false));
        }
        Ok(())
    }

    pub(super) fn preview_belt(
        &mut self,
        pos: Vec2,
        painter: &Painter,
        paint_transform: &PaintTransform,
    ) {
        if let Some((start_con, start_pos)) = &self.belt_connection {
            let (end_con, end_pos) = self.find_belt_con(pos, true);
            if matches!(end_con, BeltConnection::Structure(_)) && end_con != *start_con {
                painter.rect_filled(
                    Rect::from_center_size(
                        paint_transform.to_pos2(end_pos),
                        vec2(STRUCTURE_ICON_SIZE, STRUCTURE_ICON_SIZE),
                    ),
                    0.,
                    Color32::from_rgb(255, 127, 191),
                );
            }
            let color = if (end_pos - *start_pos).length2() < MAX_BELT_LENGTH.powi(2)
                && !intersects_water(*start_pos, end_pos, &self.heightmap)
                && !exceeds_slope(*start_pos, end_pos, &self.heightmap)
            {
                Color32::from_rgba_premultiplied(127, 0, 255, 191)
            } else {
                Color32::RED
            };
            painter.line_segment(
                [
                    paint_transform.to_pos2(end_pos),
                    paint_transform.to_pos2(*start_pos),
                ],
                (2., color),
            );
        } else if let (BeltConnection::Structure(_), end_pos) = self.find_belt_con(pos, false) {
            painter.rect_filled(
                Rect::from_center_size(
                    paint_transform.to_pos2(end_pos),
                    vec2(STRUCTURE_ICON_SIZE, STRUCTURE_ICON_SIZE),
                ),
                0.,
                Color32::from_rgb(255, 127, 191),
            );
        }
    }
}

fn intersects_water(start_pos: Vec2, end_pos: Vec2, heightmap: &HeightMap) -> bool {
    let interpolations = (start_pos.x - end_pos.x)
        .abs()
        .ceil()
        .max((start_pos.y - end_pos.y).abs().ceil()) as usize;
    (0..interpolations).any(|i| {
        let pos = start_pos.lerp(end_pos, i as f64 / interpolations as f64);
        heightmap.is_water(&pos)
    })
}

/// Checks if the line segment between start_pos and end_pos has a point that exceeds maximum slope.
fn exceeds_slope(start_pos: Vec2, end_pos: Vec2, heightmap: &HeightMap) -> bool {
    let interpolations = (start_pos.x - end_pos.x)
        .abs()
        .ceil()
        .max((start_pos.y - end_pos.y).abs().ceil()) as usize;
    (0..interpolations).any(|i| {
        let pos = start_pos.lerp(end_pos, i as f64 / interpolations as f64);
        BELT_MAX_SLOPE.powi(2) < heightmap.gradient(&pos).length2()
    })
}
