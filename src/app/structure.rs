use eframe::{
    egui::{self, Color32, Painter, Pos2, Rect, pos2, vec2},
    epaint::PathShape,
};

use crate::{
    structure::{
        BeltConnection, INGOT_CAPACITY, ITEM_INTERVAL, Item, ORE_MINE_CAPACITY, Structure,
        StructureOrBelt, StructureType,
    },
    transform::PaintTransform,
    vec2::Vec2,
};

use super::TrainsApp;

pub(super) const STRUCTURE_ICON_SIZE: f32 = 10.;
const SELECT_THRESHOLD: f64 = 10.;

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

        let paint_bar = |st: &Structure| {
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
                paint_bar(structure);
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
            Color32::from_rgb(255, 127, 191)
        } else {
            match ty {
                StructureType::OreMine => Color32::from_rgb(0, 127, 191),
                StructureType::Smelter => Color32::from_rgb(0, 191, 127),
                StructureType::Sink => Color32::from_rgb(127, 0, 127),
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
}
