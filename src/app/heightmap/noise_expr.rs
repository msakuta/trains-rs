use std::collections::HashMap;

use crate::{
    perlin_noise::{Xorshift64Star, gen_seeds, perlin_noise_pixel, white_fractal_noise},
    vec2::Vec2,
};

use super::{softabs, softclamp, softmax};

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Expr {
    Literal(f64),
    Variable(String),
    FnInvoke(String, Vec<Expr>, FnContext),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Div(Box<Expr>, Box<Expr>),
}

#[derive(Clone, Debug, PartialEq)]
pub(super) enum Stmt {
    VarDef(String, Expr),
    Expr(Expr),
}

pub(super) type Ast = Vec<Stmt>;

#[derive(Clone)]
pub(super) enum Value {
    Scalar(f64),
    Vec2(Vec2<f64>),
}

#[derive(Clone, Debug, PartialEq)]
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
pub(super) fn precompute(ast: &mut Ast, rng: &mut Xorshift64Star) -> Result<(), String> {
    let mut constants = HashMap::new();
    for stmt in ast {
        match stmt {
            Stmt::VarDef(name, ex) => {
                if let Some(val) = precompute_expr(ex, &constants, rng)? {
                    constants.insert(name.clone(), val);
                }
            }
            Stmt::Expr(ex) => {
                precompute_expr(ex, &constants, rng)?;
            }
        }
    }
    Ok(())
}

/// Returns Ok(Some(value)) if it was a constant expression, meaning the sub-expression does not contain a reference
/// to "x", thus allows us to fold it to a constant.
fn precompute_expr(
    expr: &mut Expr,
    constants: &HashMap<String, Value>,
    rng: &mut Xorshift64Star,
) -> Result<Option<Value>, String> {
    use Value::{Scalar, Vec2 as Vector};
    Ok(match expr {
        Expr::FnInvoke(fname, args, ctx) if fname == "perlin_noise" || fname == "fractal_noise" => {
            let Some([x, octaves]) = args.get_mut(..2) else {
                return Err(format!("{fname} requires at least 2 arguments"));
            };
            precompute_expr(x, constants, rng)?;
            let Some(Value::Scalar(octaves)) = precompute_expr(octaves, constants, rng)? else {
                return Err(format!(
                    "octaves for {fname} must be a constant scalar, but got {octaves:?}"
                ));
            };
            ctx.seeds = gen_seeds(rng, octaves as u32);
            for arg in &mut args[2..] {
                precompute_expr(arg, constants, rng)?;
            }
            None
        }
        Expr::Literal(val) => Some(Scalar(*val)),
        Expr::Variable(name) => constants.get(name).cloned(),
        Expr::FnInvoke(name, args, _) => {
            let args = args
                .iter_mut()
                .map(|arg| precompute_expr(arg, constants, rng))
                .collect::<Result<Vec<Option<_>>, _>>()?;
            let args = args.into_iter().collect::<Option<Vec<_>>>();

            // Evaluate the function only if all the arguments are constants, otherwise return None
            if let Some(args) = args {
                Some(eval_fn(name, &args, None)?)
            } else {
                None
            }
        }
        Expr::Add(lhs, rhs) => precompute_bin_op(lhs, rhs, constants, rng, |lhs, rhs| {
            Ok(bin_op(lhs, rhs, |lhs, rhs| lhs + rhs, |lhs, rhs| lhs + rhs))
        })?,
        Expr::Sub(lhs, rhs) => precompute_bin_op(lhs, rhs, constants, rng, |lhs, rhs| {
            Ok(bin_op(lhs, rhs, |lhs, rhs| lhs - rhs, |lhs, rhs| lhs - rhs))
        })?,
        Expr::Mul(lhs, rhs) => precompute_bin_op(lhs, rhs, constants, rng, |lhs, rhs| {
            Ok(match (lhs, rhs) {
                (Scalar(lhs), Scalar(rhs)) => Scalar(lhs + rhs),
                (Scalar(lhs), Vector(rhs)) => Vector(rhs * lhs),
                (Vector(lhs), Scalar(rhs)) => Vector(lhs * rhs),
                (Vector(_), Vector(_)) => {
                    return Err("Multiplying vectors are not supported".to_string());
                }
            })
        })?,
        Expr::Div(lhs, rhs) => precompute_bin_op(lhs, rhs, constants, rng, |lhs, rhs| {
            Ok(match (lhs, rhs) {
                (Scalar(lhs), Scalar(rhs)) => Scalar(lhs / rhs),
                (Vector(lhs), Scalar(rhs)) => Vector(lhs / rhs),
                (_, Vector(_)) => return Err("Division by a vector is not supported".to_string()),
            })
        })?,
    })
}

fn precompute_bin_op(
    lhs: &mut Expr,
    rhs: &mut Expr,
    constants: &HashMap<String, Value>,
    rng: &mut Xorshift64Star,
    process: impl Fn(Value, Value) -> Result<Value, String>,
) -> Result<Option<Value>, String> {
    let Some(lhs) = precompute_expr(lhs, constants, rng)? else {
        return Ok(None);
    };
    let Some(rhs) = precompute_expr(rhs, constants, rng)? else {
        return Ok(None);
    };
    Ok(Some(process(lhs, rhs)?))
}

pub(super) fn run(ast: &Ast, x: &Vec2<f64>) -> Result<Value, String> {
    let mut variables = HashMap::new();
    let mut res = None;
    for stmt in ast {
        match stmt {
            Stmt::VarDef(name, ex) => {
                variables.insert(name.clone(), eval(ex, x, &variables)?);
            }
            Stmt::Expr(ex) => {
                res = Some(eval(ex, x, &variables)?);
            }
        }
    }
    res.ok_or_else(|| "evaluated result does not contain an expression".to_string())
}

fn eval(expr: &Expr, x: &Vec2<f64>, variables: &HashMap<String, Value>) -> Result<Value, String> {
    use Value::{Scalar, Vec2 as Vector};
    Ok(match expr {
        Expr::Literal(val) => Value::Scalar(*val),
        Expr::Variable(name) => {
            if name == "x" {
                Value::Vec2(*x)
            } else if let Some(var) = variables.get(name) {
                var.clone()
            } else {
                return Err(format!("Variable {name} was not defined"));
            }
        }
        Expr::FnInvoke(name, args, fn_ctx) => {
            let args = args
                .iter()
                .map(|arg| eval(arg, x, variables))
                .collect::<Result<Vec<_>, _>>()?;
            eval_fn(name, &args, Some(fn_ctx))?
        }
        Expr::Add(lhs, rhs) => bin_op(
            eval(lhs, x, variables)?,
            eval(rhs, x, variables)?,
            |lhs, rhs| lhs + rhs,
            |lhs, rhs| lhs + rhs,
        ),
        Expr::Sub(lhs, rhs) => bin_op(
            eval(lhs, x, variables)?,
            eval(rhs, x, variables)?,
            |lhs, rhs| lhs - rhs,
            |lhs, rhs| lhs - rhs,
        ),
        Expr::Mul(lhs, rhs) => match (eval(lhs, x, variables)?, eval(rhs, x, variables)?) {
            (Scalar(lhs), Scalar(rhs)) => Scalar(lhs * rhs),
            (Scalar(lhs), Vector(rhs)) => Vector(rhs * lhs),
            (Vector(lhs), Scalar(rhs)) => Vector(lhs * rhs),
            (Vector(_), Vector(_)) => {
                return Err("Multiplying vectors are not supported".to_string());
            }
        },
        Expr::Div(lhs, rhs) => match (eval(lhs, x, variables)?, eval(rhs, x, variables)?) {
            (Scalar(lhs), Scalar(rhs)) => Scalar(lhs / rhs),
            (Vector(lhs), Scalar(rhs)) => Vector(lhs / rhs),
            (_, Vector(_)) => return Err("Division by a vector is not supported".to_string()),
        },
    })
}

fn eval_fn(name: &str, args: &[Value], fn_ctx: Option<&FnContext>) -> Result<Value, String> {
    Ok(match name as &str {
        "vec2" => {
            if let Some([Value::Scalar(x), Value::Scalar(y)]) = args.get(..2) {
                Value::Vec2(Vec2::new(*x, *y))
            } else {
                return Err("vec2 only supports 2 scalar arguments".to_string());
            }
        }
        "x" => {
            if let Some(Value::Vec2(vec)) = args.get(0) {
                Value::Scalar(vec.x)
            } else {
                return Err("x only supports 1 vector argument".to_string());
            }
        }
        "y" => {
            if let Some(Value::Vec2(vec)) = args.get(0) {
                Value::Scalar(vec.y)
            } else {
                return Err("y only supports 1 vector argument".to_string());
            }
        }
        "softclamp" => {
            if let Some([Value::Scalar(val), Value::Scalar(max)]) = args.get(..2) {
                Value::Scalar(softclamp(*val, *max))
            } else {
                return Err("softclamp only supports 2 scalar arguments".to_string());
            }
        }
        "softabs" => {
            if let Some([Value::Scalar(val), Value::Scalar(rounding)]) = args.get(..2) {
                Value::Scalar(softabs(*val, *rounding))
            } else {
                return Err("softabs only supports 2 scalar arguments".to_string());
            }
        }
        "softmax" => {
            if let Some([Value::Scalar(lhs), Value::Scalar(rhs)]) = args.get(..2) {
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
            ) = args.get(..3)
            {
                let Some(fn_ctx) = fn_ctx else {
                    return Err(format!("{name} is only allowed in non-const expression"));
                };
                Value::Scalar(if name == "perlin_noise" {
                    perlin_noise_pixel(val.x, val.y, *octaves as u32, &fn_ctx.seeds, *persistence)
                } else {
                    white_fractal_noise(val.x, val.y, &fn_ctx.seeds, *persistence)
                })
            } else {
                return Err(
                    "perlin_nosie only supports (vector, scalar, scalar) arguments".to_string(),
                );
            }
        }
        _ => return Err(format!("Function {name} is not defined")),
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
