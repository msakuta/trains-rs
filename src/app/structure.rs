use cgmath::{Matrix3, Rad, Vector2};
use eframe::{
    egui::{
        self, Color32, Painter, Pos2, Rect, SizeHint, Stroke, TextureOptions, load::TexturePoll,
        pos2, vec2,
    },
    epaint::PathShape,
};
use ordered_float::NotNan;

use crate::{
    app::{COAL_URL, INGOT_URL, ORE_URL},
    structure::{
        BELT_MAX_SLOPE, BELT_SPEED, Belt, BeltConnection, EntityId, ITEM_INTERVAL, Item,
        MAX_BELT_LENGTH, MAX_FLUID_AMOUNT, ORE_MINE_CAPACITY, Pipe, PipeConnection, Structure,
        StructureType,
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
            const BAR_OFFSET: f32 = 30.;
            let (fullness, color, y_pos);
            match st.ty {
                StructureType::OreMine
                | StructureType::Smelter
                | StructureType::Sink
                | StructureType::Loader
                | StructureType::Splitter
                | StructureType::Merger
                | StructureType::Unloader => {
                    fullness = st.inventory.sum() as f32 / ORE_MINE_CAPACITY as f32;
                    color = Color32::from_rgb(255, 255, 0);
                    y_pos = base_pos.y + BAR_OFFSET;
                }
                StructureType::WaterPump => {
                    fullness =
                        st.output_fluid.map_or(0., |fb| fb.amount) as f32 / MAX_FLUID_AMOUNT as f32;
                    color = Color32::from_rgb(0, 255, 255);
                    y_pos = base_pos.y + BAR_OFFSET;
                }
            }

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

            Some(())
        };

        if 2. < self.transform.scale() {
            for (_, structure) in &self.structures.structures {
                paint_bars(structure);
            }
        }
    }

    fn render_belt(
        &self,
        belt: &Belt,
        painter: &Painter,
        paint_transform: &PaintTransform,
        color: Color32,
    ) {
        let start = paint_transform.to_pos2(belt.start);
        let end = paint_transform.to_pos2(belt.end);

        if self.transform.scale() < 2. {
            painter.arrow(start, end - start, (2., color));
        } else {
            let delta = belt.end - belt.start;
            let length = delta.length();
            let tangent = delta / length;
            let normal = tangent.left90();
            let width = 0.5;
            let fill_color = Color32::from_rgba_premultiplied(
                color.r() / 2 + 127,
                color.g() / 2 + 127,
                color.b() / 2 + 127,
                color.a(),
            );
            painter.add(PathShape::convex_polygon(
                [
                    belt.start + normal * width,
                    belt.end + normal * width,
                    belt.end - normal * width,
                    belt.start - normal * width,
                ]
                .into_iter()
                .map(|p| paint_transform.to_pos2(p))
                .collect(),
                fill_color,
                (1., color),
            ));

            let interval = 3.;
            for t in 0..(length / interval).ceil() as u32 {
                let s = interval * t as f64 + (self.time as f64 * BELT_SPEED) % interval;
                if length < s {
                    break;
                }
                let pos = belt.start + tangent * s;
                let center = paint_transform.to_pos2(pos);
                let left = paint_transform.to_pos2(pos + normal * width - tangent * 0.5);
                let right = paint_transform.to_pos2(pos - normal * width - tangent * 0.5);
                painter.line(vec![left, center, right], (1., color));
            }
        }
    }

    pub(super) fn render_belts(&self, painter: &Painter, paint_transform: &PaintTransform) {
        let scale = self.transform.scale();
        for belt in self.structures.belts.values() {
            self.render_belt(belt, painter, paint_transform, Color32::BLUE);

            // Render items only when they are likely more than 1 pixels
            if ITEM_INTERVAL < scale as f64 {
                let length = (belt.end - belt.start).length();

                for (item, dist) in &mut belt.items.iter() {
                    let f = *dist / length;
                    let pos = belt.start * (1. - f) + belt.end * f;
                    if let Err(e) = self.render_item(pos, item, painter, paint_transform) {
                        println!("{e}");
                    }
                }
            }
        }
    }

    fn render_pipe(
        &self,
        pipe: &Pipe,
        painter: &Painter,
        paint_transform: &PaintTransform,
        color: Color32,
    ) {
        let start = paint_transform.to_pos2(pipe.start);
        let end = paint_transform.to_pos2(pipe.end);

        if self.transform.scale() < 2. {
            painter.line_segment([start, end], (2., color));
        } else {
            let delta = pipe.end - pipe.start;
            let length = delta.length();
            let tangent = delta / length;
            let normal = tangent.left90();
            let width = 0.5;
            let fill_color = Color32::from_rgba_premultiplied(
                color.r() / 2 + 127,
                color.g() / 2 + 127,
                color.b() / 2 + 127,
                color.a(),
            );
            painter.add(PathShape::convex_polygon(
                [
                    pipe.start + normal * width,
                    pipe.end + normal * width,
                    pipe.end - normal * width,
                    pipe.start - normal * width,
                ]
                .into_iter()
                .map(|p| paint_transform.to_pos2(p))
                .collect(),
                fill_color,
                (1., color),
            ));

            if let Some(fluid) = &pipe.fluid {
                let width = fluid.amount / MAX_FLUID_AMOUNT / 2.;
                painter.add(PathShape::convex_polygon(
                    [
                        pipe.start + normal * width,
                        pipe.end + normal * width,
                        pipe.end - normal * width,
                        pipe.start - normal * width,
                    ]
                    .into_iter()
                    .map(|p| paint_transform.to_pos2(p))
                    .collect(),
                    Color32::from_rgb(0, 0, 191),
                    Stroke::NONE,
                ));
            }
        }
    }

    pub(super) fn render_pipes(&self, painter: &Painter, paint_transform: &PaintTransform) {
        for belt in self.structures.pipes.values() {
            self.render_pipe(
                belt,
                painter,
                paint_transform,
                Color32::from_rgb(0, 127, 191),
            );
        }
    }

    fn render_item(
        &self,
        pos: Vec2,
        item: &Item,
        painter: &Painter,
        paint_transform: &PaintTransform,
    ) -> Result<(), String> {
        match item {
            Item::IronOre => self.render_img(pos, ORE_URL, painter, paint_transform),
            Item::Ingot => self.render_img(pos, INGOT_URL, painter, paint_transform),
            Item::Coal => self.render_img(pos, COAL_URL, painter, paint_transform),
        }
    }

    fn render_img(
        &self,
        pos: Vec2,
        url: &str,
        painter: &Painter,
        paint_transform: &PaintTransform,
    ) -> Result<(), String> {
        let tex = painter.ctx().try_load_texture(
            url,
            TextureOptions {
                magnification: egui::TextureFilter::Nearest,
                minification: egui::TextureFilter::Linear,
                mipmap_mode: Some(egui::TextureFilter::Linear),
                wrap_mode: egui::TextureWrapMode::Repeat,
            },
            SizeHint::default(),
        );

        let tex = tex.map_err(|e| format!("Failed to load texture {url}: {e}"))?;

        if let TexturePoll::Ready { texture } = tex {
            let rect = Rect::from_center_size(
                paint_transform.to_pos2(pos),
                egui::Vec2::splat(self.transform.scale() * ITEM_INTERVAL as f32),
            );
            const UV: Rect = Rect::from_min_max(pos2(0., 0.), Pos2::new(1.0, 1.0));
            painter.image(texture.id, rect, UV, Color32::WHITE);
        } else {
            return Err(format!("Pending to load texture {url}"));
        }

        Ok(())
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
                StructureType::Splitter => Color32::from_rgb(95, 191, 0),
                StructureType::Merger => Color32::from_rgb(191, 95, 0),
                StructureType::WaterPump => Color32::from_rgb(0, 95, 191),
            }
        };
        let line_color = Color32::from_rgb(0, 63, 31);

        let render_triangle = |transform: Matrix3<f64>| {
            painter.add(PathShape::convex_polygon(
                [[0., -0.3], [-0.3, 0.2], [0.3, 0.2]]
                    .into_iter()
                    .map(|v| -> Pos2 {
                        let local_pos: [f64; 2] =
                            (transform * Vector2::from(v).extend(1.)).truncate().into();
                        pos + paint_transform.to_vec2(local_pos.into())
                    })
                    .collect(),
                color,
                (1., line_color),
            ));
        };

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
                match ty {
                    StructureType::Loader | StructureType::Unloader => {
                        [[-4., -1.], [4., -1.], [4., 1.], [-4., 1.]]
                    }
                    StructureType::Sink => [[-4., -4.], [4., -4.], [4., 4.], [-4., 4.]],
                    _ => [[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]],
                }
                .into_iter()
                .map(rotate)
                .collect(),
                color,
                (1., line_color),
            ));

            for (pos, local_orient) in ty.input_ports() {
                render_triangle(
                    Matrix3::from_angle_z(Rad(orient))
                        * Matrix3::from_translation(pos.to_vector2())
                        * Matrix3::from_angle_z(Rad(*local_orient))
                        * Matrix3::from_translation(0.5 * Vector2::unit_y()),
                );
            }

            for (pos, local_orient) in ty.output_ports() {
                render_triangle(
                    Matrix3::from_angle_z(Rad(orient))
                        * Matrix3::from_translation(pos.to_vector2())
                        * Matrix3::from_angle_z(Rad(*local_orient))
                        * Matrix3::from_translation(-0.5 * Vector2::unit_y()),
                );
            }
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

    pub(super) fn find_pipe_con(&self, pos: Vec2) -> (PipeConnection, Vec2) {
        self.structures
            .find_pipe_con(pos, SELECT_THRESHOLD / self.transform.scale() as f64)
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
                EntityId::Structure(id) => {
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
                EntityId::Belt(id) => {
                    if let Some(belt) = self.structures.belts.get(&id) {
                        self.render_belt(
                            belt,
                            painter,
                            paint_transform,
                            Color32::from_rgb(255, 0, 255),
                        );
                    }
                }
                EntityId::Pipe(id) => {
                    if let Some(pipe) = self.structures.pipes.get(&id) {
                        self.render_pipe(
                            pipe,
                            painter,
                            paint_transform,
                            Color32::from_rgb(255, 0, 255),
                        );
                    }
                }
                _ => {}
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
        let mut ore_mine = Structure::new_ore_mine(pos, orient);
        ore_mine.ore_type = Some(ore_vein.ty);
        let st_id = self.structures.add_structure(ore_mine);
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

    pub(super) fn add_water_pump(&mut self, pointer_pos: Vec2) -> Result<(), String> {
        let Some(pos) = self.building_structure else {
            if !self.heightmap.is_water(&pointer_pos) {
                return Err("Cannot build outside water".to_string());
            }
            self.building_structure = Some(pointer_pos);
            return Ok(());
        };
        let delta = pos - pointer_pos;
        let orient = delta.y.atan2(delta.x) - std::f64::consts::PI * 0.5;
        let water_pump = Structure::new_structure(StructureType::WaterPump, pos, orient);
        let _ = self.structures.add_structure(water_pump);
        self.building_structure = None;
        Ok(())
    }

    pub(super) fn add_belt(&mut self, pos: Vec2) -> Result<(), String> {
        if let Some((start_con, start_pos)) = &self.belt_connection {
            let (end_con, end_pos) = self.find_belt_con(pos, true);
            // Disallow connection to itself and end-to-end
            if matches!(end_con, BeltConnection::BeltEnd(_)) {
                return Err("You cannot connect the end of a belt to another end".to_string());
            }
            if matches!((end_con, start_con), (BeltConnection::Structure(eid, eidx), BeltConnection::Structure(sid, sidx)) if eid == *sid && eidx == *sidx)
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
                BeltConnection::Structure(start_st, con_idx) => {
                    if let Some(st) = self.structures.structures.get_mut(start_st) {
                        st.output_belts[*con_idx] = Some(belt_id);
                        println!("Added belt {belt_id} to output_belts[{con_idx}]");
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
            if matches!(end_con, BeltConnection::Structure(_, _)) && end_con != *start_con {
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
            // Fake belt object to preview
            let belt = Belt::new(*start_pos, *start_con, end_pos, end_con);
            self.render_belt(&belt, painter, paint_transform, color);
        } else if let (BeltConnection::Structure(_, _), end_pos) = self.find_belt_con(pos, false) {
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

    pub(super) fn add_pipe(&mut self, pos: Vec2) -> Result<(), String> {
        if let Some((start_con, start_pos)) = &self.pipe_connection {
            let (end_con, end_pos) = self.find_pipe_con(pos);
            if matches!((end_con, start_con), (PipeConnection::Structure(eid, eidx), PipeConnection::Structure(sid, sidx)) if eid == *sid && eidx == *sidx)
            {
                return Err("You cannot connect a pipe itself".to_string());
            }
            if MAX_BELT_LENGTH.powi(2) <= (end_pos - *start_pos).length2() {
                return Err("Pipe is too long".to_string());
            }
            if exceeds_slope(*start_pos, end_pos, &self.heightmap) {
                return Err("Pipe is exceeding maximum slope".to_string());
            }
            let pipe_id = self
                .structures
                .add_pipe(*start_pos, *start_con, end_pos, end_con);
            match start_con {
                PipeConnection::Structure(start_st, con_idx) => {
                    if let Some(st) = self.structures.structures.get_mut(start_st) {
                        st.connected_pipes = Some(pipe_id);
                        println!("Added belt {pipe_id} to output_belts[{con_idx}]");
                    }
                }
                // If we connect to a belt, we add a reference to the upstream belt.
                PipeConnection::PipeEnd(connecting_pipe) => {
                    if let Some(connecting_pipe) = self.structures.pipes.get_mut(connecting_pipe) {
                        connecting_pipe.end_con = PipeConnection::PipeStart(pipe_id);
                        println!("Added pipe {pipe_id} to connecting pipe");
                    }
                }
                _ => {}
            }
            self.pipe_connection = None;
        } else {
            self.pipe_connection = Some(self.find_pipe_con(pos));
        }
        Ok(())
    }

    pub(super) fn preview_pipe(
        &mut self,
        pos: Vec2,
        painter: &Painter,
        paint_transform: &PaintTransform,
    ) {
        if let Some((start_con, start_pos)) = &self.pipe_connection {
            let (end_con, end_pos) = self.find_pipe_con(pos);
            if matches!(end_con, PipeConnection::Structure(_, _)) && end_con != *start_con {
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
            // Fake pipe object to preview
            let pipe = Pipe::new(*start_pos, *start_con, end_pos, end_con);
            self.render_pipe(&pipe, painter, paint_transform, color);
        } else if let (PipeConnection::Structure(_, _), end_pos) = self.find_pipe_con(pos) {
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
    let delta = start_pos - end_pos;
    let interpolations = delta.x.abs().ceil().max(delta.y.abs().ceil()) as usize;
    (0..interpolations).any(|i| {
        let pos = start_pos.lerp(end_pos, i as f64 / interpolations as f64);
        heightmap.is_water(&pos)
    })
}

/// Checks if the line segment between start_pos and end_pos has a point that exceeds maximum slope.
fn exceeds_slope(start_pos: Vec2, end_pos: Vec2, heightmap: &HeightMap) -> bool {
    let delta = start_pos - end_pos;
    let interpolations = delta.x.abs().ceil().max(delta.y.abs().ceil()) as usize;
    (0..interpolations).any(|i| {
        let pos = start_pos.lerp(end_pos, i as f64 / interpolations as f64);
        BELT_MAX_SLOPE.powi(2) < heightmap.gradient(&pos).length2()
    })
}
