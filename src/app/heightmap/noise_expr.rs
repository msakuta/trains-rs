use crate::{
    perlin_noise::{Xorshift64Star, gen_seeds, perlin_noise_pixel, white_fractal_noise},
    vec2::Vec2,
};

use super::{softabs, softclamp, softmax};

#[derive(Clone)]
pub(super) enum Expr {
    Literal(f64),
    Variable(String),
    FnInvoke(String, Vec<Expr>, FnContext),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

pub(super) enum Value {
    Scalar(f64),
    Vec2(Vec2<f64>),
}

#[derive(Clone)]
pub(super) struct FnContext {
    seeds: Vec<u64>,
}

impl FnContext {
    pub fn new() -> Self {
        Self { seeds: vec![] }
    }
}

/// Precompute uniform constants in the AST and store them in the nodes.
/// The most important kind of constants is the perlin noise seed values, which are initialized using the given rng,
/// which means the values depend on the order of appearance in the traversal of the AST.
///
/// Theoretically, we could fold constants like general purpose languages.
pub(super) fn precompute(expr: &mut Expr, rng: &mut Xorshift64Star) -> Result<(), String> {
    match expr {
        Expr::FnInvoke(fname, args, ctx) if fname == "perlin_noise" || fname == "fractal_noise" => {
            let Some([x, octaves]) = args.get_mut(..2) else {
                return Err(format!("{fname} requires at least 2 arguments"));
            };
            precompute(x, rng)?;
            let Expr::Literal(octaves) = octaves else {
                return Err(format!("octaves for {fname} must be a constant"));
            };
            ctx.seeds = gen_seeds(rng, *octaves as u32);
            for arg in &mut args[2..] {
                precompute(arg, rng)?;
            }
        }
        Expr::FnInvoke(_, args, _) => {
            for arg in args {
                precompute(arg, rng)?;
            }
        }
        Expr::Add(lhs, rhs) | Expr::Sub(lhs, rhs) | Expr::Mul(lhs, rhs) | Expr::Div(lhs, rhs) => {
            precompute(lhs, rng)?;
            precompute(rhs, rng)?;
        }
        _ => {}
    }
    Ok(())
}

pub(super) fn eval(expr: &Expr, x: &Vec2<f64>) -> Result<Value, String> {
    use Value::{Scalar, Vec2 as Vector};
    Ok(match expr {
        Expr::Literal(val) => Value::Scalar(*val),
        Expr::Variable(name) => {
            if name == "x" {
                Value::Vec2(*x)
            } else {
                return Err(format!("Variable {name} was not supported yet"));
            }
        }
        Expr::FnInvoke(name, args, fn_ctx) => {
            let val = args
                .iter()
                .map(|arg| eval(arg, x))
                .collect::<Result<Vec<_>, _>>()?;
            match name as &str {
                "softclamp" => {
                    if let Some([Value::Scalar(val), Value::Scalar(max)]) = val.get(..2) {
                        Value::Scalar(softclamp(*val, *max))
                    } else {
                        return Err("softclamp only supports 2 scalar arguments".to_string());
                    }
                }
                "softabs" => {
                    if let Some([Value::Scalar(val), Value::Scalar(rounding)]) = val.get(..2) {
                        Value::Scalar(softabs(*val, *rounding))
                    } else {
                        return Err("softabs only supports 2 scalar arguments".to_string());
                    }
                }
                "softmax" => {
                    if let Some([Value::Scalar(lhs), Value::Scalar(rhs)]) = val.get(..2) {
                        Value::Scalar(softmax(*lhs, *rhs))
                    } else {
                        return Err("softmax only supports 2 scalar arguments".to_string());
                    }
                }
                "perlin_noise" | "fractal_noise" => {
                    if let Some(
                        [
                            Value::Vec2(val),
                            Value::Scalar(octaves),
                            Value::Scalar(persistence),
                        ],
                    ) = val.get(..3)
                    {
                        Value::Scalar(if name == "perlin_noise" {
                            perlin_noise_pixel(
                                val.x,
                                val.y,
                                *octaves as u32,
                                &fn_ctx.seeds,
                                *persistence,
                            )
                        } else {
                            white_fractal_noise(val.x, val.y, &fn_ctx.seeds, *persistence)
                        })
                    } else {
                        return Err(
                            "perlin_nosie only supports (vector, scalar, scalar) arguments"
                                .to_string(),
                        );
                    }
                }
                _ => return Err(format!("Function {name} is not defined")),
            }
        }
        Expr::Add(lhs, rhs) => bin_op(
            eval(lhs, x)?,
            eval(rhs, x)?,
            |lhs, rhs| lhs + rhs,
            |lhs, rhs| lhs + rhs,
        ),
        Expr::Sub(lhs, rhs) => bin_op(
            eval(lhs, x)?,
            eval(rhs, x)?,
            |lhs, rhs| lhs - rhs,
            |lhs, rhs| lhs - rhs,
        ),
        Expr::Mul(lhs, rhs) => match (eval(lhs, x)?, eval(rhs, x)?) {
            (Scalar(lhs), Scalar(rhs)) => Scalar(lhs + rhs),
            (Scalar(lhs), Vector(rhs)) => Vector(rhs * lhs),
            (Vector(lhs), Scalar(rhs)) => Vector(lhs * rhs),
            (Vector(_), Vector(_)) => {
                return Err("Multiplying vectors are not supported".to_string());
            }
        },
        Expr::Div(lhs, rhs) => match (eval(lhs, x)?, eval(rhs, x)?) {
            (Scalar(lhs), Scalar(rhs)) => Scalar(lhs / rhs),
            (Vector(lhs), Scalar(rhs)) => Vector(lhs / rhs),
            (_, Vector(_)) => return Err("Division by a vector is not supported".to_string()),
        },
    })
}

fn bin_op(
    lhs: Value,
    rhs: Value,
    scalar_op: impl Fn(f64, f64) -> f64,
    vector_op: impl Fn(Vec2<f64>, Vec2<f64>) -> Vec2<f64>,
) -> Value {
    use Value::{Scalar, Vec2 as Vector};
    match (lhs, rhs) {
        (Scalar(lhs), Scalar(rhs)) => Scalar(scalar_op(lhs, rhs)),
        (Scalar(lhs), Vector(rhs)) => Vector(vector_op(Vec2::splat(lhs), rhs)),
        (Vector(lhs), Scalar(rhs)) => Vector(vector_op(lhs, Vec2::splat(rhs))),
        (Vector(lhs), Vector(rhs)) => Vector(vector_op(lhs, rhs)),
    }
}
