use std::{
    fmt::Display,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub, SubAssign},
};

use cgmath::{Vector2, Zero};
use serde::{Deserialize, Serialize};

/// A generic custom 2D vector type that can take f64 or TapeTerm as a type argument.
/// It can be a bit confusing with `egui::Vec2` so I may rename it.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Vec2<T = f64> {
    pub x: T,
    pub y: T,
}

impl<T> Vec2<T> {
    pub const fn new(x: T, y: T) -> Self {
        Self { x, y }
    }

    pub const fn splat(x: T) -> Self
    where
        T: Copy,
    {
        Self { x, y: x }
    }
}

impl<T: Copy> Vec2<T> {
    pub fn map<U>(&self, f: impl Fn(T) -> U) -> Vec2<U> {
        Vec2 {
            x: f(self.x),
            y: f(self.y),
        }
    }
}

impl<T: Display> Display for Vec2<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}, {})", self.x, self.y)
    }
}

impl<T> PartialEq for Vec2<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.x.eq(&other.x) && self.y.eq(&other.y)
    }
}

impl<T: Default> Default for Vec2<T> {
    fn default() -> Self {
        Self {
            x: T::default(),
            y: T::default(),
        }
    }
}

impl<T> Vec2<T>
where
    T: Add<T, Output = T> + Mul<T, Output = T> + Clone + Copy,
{
    pub fn length2(&self) -> T {
        self.x * self.x + self.y * self.y
    }

    #[allow(dead_code)]
    pub fn dot(&self, rhs: Self) -> T {
        self.x * rhs.x + self.y * rhs.y
    }
}

impl<T> Vec2<T>
where
    T: Neg<Output = T> + Clone + Copy,
{
    /// Return a copy of the vector with 90 degrees rotated to the left.
    /// It only makes sense in 2D.
    pub fn left90(&self) -> Self {
        Self {
            x: -self.y,
            y: self.x,
        }
    }
}

impl Vec2<f64> {
    pub fn length(&self) -> f64 {
        self.length2().sqrt()
    }

    #[allow(dead_code)]
    pub fn normalized(&self) -> Self {
        let len = self.length();
        if len.is_subnormal() {
            Self {
                x: self.x,
                y: self.y,
            }
        } else {
            Self {
                x: self.x / len,
                y: self.y / len,
            }
        }
    }

    pub fn zero() -> Self {
        Self { x: 0., y: 0. }
    }

    pub fn lerp(&self, rhs: Self, f: f64) -> Self {
        *self * (1. - f) + rhs * f
    }
}

/// Bridges to cgmath types
impl<T: Clone + Copy + Zero> Vec2<T> {
    pub fn to_vector2(&self) -> cgmath::Vector2<T> {
        cgmath::Vector2::new(self.x, self.y)
    }

    pub fn to_vector3(&self) -> cgmath::Vector3<T> {
        cgmath::Vector3::new(self.x, self.y, T::zero())
    }
}

impl<T> From<[T; 2]> for Vec2<T> {
    fn from([x, y]: [T; 2]) -> Self {
        Vec2 { x, y }
    }
}

impl<T> From<Vector2<T>> for Vec2<T> {
    fn from(v: Vector2<T>) -> Self {
        Vec2 { x: v.x, y: v.y }
    }
}

impl<T: Add<Output = T>> Add for Vec2<T> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: AddAssign> AddAssign for Vec2<T> {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl<T: Sub<Output = T>> Sub for Vec2<T> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T: SubAssign> SubAssign for Vec2<T> {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl<T: Mul<Output = T> + Copy> Mul<T> for Vec2<T> {
    type Output = Self;
    fn mul(self, rhs: T) -> Self::Output {
        Self {
            x: self.x * rhs,
            y: self.y * rhs,
        }
    }
}

impl<T: Div<Output = T> + Copy> Div<T> for Vec2<T> {
    type Output = Self;
    fn div(self, rhs: T) -> Self::Output {
        Self {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl<T: Neg<Output = T>> Neg for Vec2<T> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            x: -self.x,
            y: -self.y,
        }
    }
}
