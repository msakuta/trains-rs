use eframe::egui::{Color32, Painter, Pos2, Rect, pos2, vec2};

use crate::{
    structure::{INGOT_CAPACITY, ORE_MINE_CAPACITY, STRUCTURE_SIZE, Structure, StructureType},
    transform::PaintTransform,
};

use super::TrainsApp;

pub(super) const STRUCTURE_ICON_SIZE: f32 = 10.;

impl TrainsApp {
    pub(super) fn render_structures(&self, painter: &Painter, paint_transform: &PaintTransform) {
        for (_, structure) in &self.structures.structures {
            let pos = paint_transform.to_pos2(structure.pos);
            Self::render_structure(pos, false, structure.ty, painter, paint_transform);
        }

        let paint_bar = |st: &Structure| {
            let base_pos = paint_transform.to_pos2(st.pos).to_vec2();
            const BAR_WIDTH: f32 = 50.;
            const BAR_HEIGHT: f32 = 10.;
            const BAR_OFFSET: f32 = 30.;
            painter.rect_filled(
                Rect::from_center_size(
                    pos2(base_pos.x, base_pos.y + BAR_OFFSET),
                    vec2(BAR_WIDTH, BAR_HEIGHT),
                ),
                0.,
                Color32::BLACK,
            );

            for (y, fullness) in [
                (0, st.iron as f32 / ORE_MINE_CAPACITY as f32),
                (1, st.ingot as f32 / INGOT_CAPACITY as f32),
            ] {
                painter.rect_filled(
                    Rect::from_min_size(
                        pos2(
                            base_pos.x - BAR_WIDTH / 2.,
                            base_pos.y + BAR_OFFSET - BAR_HEIGHT / 2. + y as f32 * BAR_HEIGHT,
                        ),
                        vec2(fullness * BAR_WIDTH, BAR_HEIGHT),
                    ),
                    0.,
                    Color32::GREEN,
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

    /// A common rendering logic for the built structures and preview ghosts.
    pub(super) fn render_structure(
        pos: Pos2,
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

        // Render the real size with enough zoom
        if 2. < paint_transform.scale() {
            let size = STRUCTURE_SIZE as f32 * paint_transform.scale();
            painter.rect_filled(Rect::from_center_size(pos, vec2(size, size)), 0., color);
        } else {
            // render icon that does not shink if zoomed out
            painter.rect_filled(
                Rect::from_center_size(pos, vec2(STRUCTURE_ICON_SIZE, STRUCTURE_ICON_SIZE)),
                0.,
                color,
            );
        }
    }
}
