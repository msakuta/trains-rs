//! A widget for background image.

use eframe::egui::{self, Color32, Painter, Pos2, Rect, TextureOptions, Vec2, pos2};

use crate::transform::PaintTransform;

/// An abstraction over a cached texture handle.
/// It constructs the image resource in egui on the first call.
pub(crate) struct BgImage {
    texture: Option<egui::TextureHandle>,
}

impl BgImage {
    pub fn new() -> Self {
        Self { texture: None }
    }

    pub fn _clear(&mut self) {
        self.texture.take();
    }

    pub fn paint<T, E>(
        &mut self,
        painter: &Painter,
        app_data: T,
        img_getter: impl Fn(T) -> Result<egui::ColorImage, E>,
        paint_transform: &PaintTransform,
    ) -> Result<(), E> {
        let texture: &egui::TextureHandle = if let Some(texture) = &self.texture {
            texture
        } else {
            let image = img_getter(app_data)?;
            self.texture.get_or_insert_with(|| {
                // Load the texture only once.
                painter.ctx().load_texture(
                    "my-image",
                    image,
                    TextureOptions {
                        magnification: egui::TextureFilter::Nearest,
                        minification: egui::TextureFilter::Linear,
                        mipmap_mode: Some(egui::TextureFilter::Linear),
                        wrap_mode: egui::TextureWrapMode::Repeat,
                    },
                )
            })
        };

        let origin = paint_transform.transform_pos2(pos2(0., texture.size()[1] as f32));
        let scale = paint_transform.scale();
        let size = texture.size_vec2() * scale;
        let min = Vec2::new(origin.x as f32, origin.y as f32);
        let max = min + size;
        let rect = Rect {
            min: min.to_pos2(),
            max: max.to_pos2(),
        };
        const UV: Rect = Rect::from_min_max(pos2(0., 1.), Pos2::new(1.0, 0.0));
        painter.image(texture.id(), rect, UV, Color32::WHITE);
        Ok(())
    }
}
