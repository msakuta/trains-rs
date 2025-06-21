use crate::{perlin_noise::perlin_noise_pixel, vec2::Vec2};

use super::softclamp;

pub(super) enum Expr {
    Literal(f64),
    Variable(String),
    FnInvoke(String, Vec<Expr>),
}

pub(super) enum Value {
    Scalar(f64),
    Vec2(Vec2<f64>),
}

pub(super) struct EvalContext {
    pub octaves: u32,
    pub seeds: Vec<u64>,
    pub persistence: f64,
}

pub(super) fn eval(expr: &Expr, x: &Vec2<f64>, context: &EvalContext) -> Result<Value, String> {
    Ok(match expr {
        Expr::Literal(val) => Value::Scalar(*val),
        Expr::Variable(name) => {
            if name == "x" {
                Value::Vec2(*x)
            } else {
                return Err(format!("Variable {name} was not supported yet"));
            }
        }
        Expr::FnInvoke(name, args) => {
            let val = args
                .iter()
                .map(|arg| eval(arg, x, context))
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
                    if let Some(Value::Vec2(val)) = val.get(0) {
                        Value::Scalar(perlin_noise_pixel(
                            val.x,
                            val.y,
                            context.octaves,
                            &context.seeds,
                            context.persistence,
                        ))
                    } else {
                        return Err("perlin_nosie only supports vector argument".to_string());
                    }
                }
                _ => return Err(format!("Function {name} is not defined")),
            }
        }
    })
}
