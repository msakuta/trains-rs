use crate::{
    perlin_noise::{Xorshift64Star, gen_seeds, perlin_noise_pixel},
    vec2::Vec2,
};

use super::softclamp;

pub(super) enum Expr {
    Literal(f64),
    Variable(String),
    FnInvoke(String, Vec<Expr>, FnContext),
}

pub(super) enum Value {
    Scalar(f64),
    Vec2(Vec2<f64>),
}

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
        Expr::FnInvoke(fname, args, ctx) if fname == "perlin_noise" => {
            let Some([x, octaves]) = args.get_mut(..2) else {
                return Err("perlin_noise requires at least 2 arguments".to_string());
            };
            precompute(x, rng)?;
            let Expr::Literal(octaves) = octaves else {
                return Err("octaves for perlin_noise must be a constant".to_string());
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
        _ => {}
    }
    Ok(())
}

pub(super) fn eval(expr: &Expr, x: &Vec2<f64>) -> Result<Value, String> {
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
                        Value::Scalar(softclamp(*val, *rounding))
                    } else {
                        return Err("softabs only supports 2 scalar arguments".to_string());
                    }
                }
                "perlin_noise" => {
                    if let Some(
                        [
                            Value::Vec2(val),
                            Value::Scalar(octaves),
                            Value::Scalar(persistence),
                        ],
                    ) = val.get(..3)
                    {
                        Value::Scalar(perlin_noise_pixel(
                            val.x,
                            val.y,
                            *octaves as u32,
                            &fn_ctx.seeds,
                            *persistence,
                        ))
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
    })
}
