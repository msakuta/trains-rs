use nom::{
    Finish, IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, multispace0},
    combinator::{opt, recognize},
    error::ParseError,
    multi::{fold_many0, fold_many1, many0, separated_list0},
    number::complete::float,
    sequence::{delimited, pair, terminated},
};

use crate::app::heightmap::noise_expr::FnContext;

use super::noise_expr::{Ast, Expr, Stmt};

type Span<'a> = &'a str;

/// A combinator that takes a parser `inner` and produces a parser that also consumes both leading and
/// trailing whitespace, returning the output of `inner`.
fn ws<'a, F: 'a, O, E: ParseError<Span<'a>>>(
    inner: F,
) -> impl Parser<&'a str, Output = O, Error = E>
where
    F: Parser<Span<'a>, Output = O, Error = E>,
{
    delimited(multispace0, inner, multispace0)
}

fn identifier(input: Span) -> IResult<Span, Span> {
    recognize(pair(
        alt((alpha1, tag("_"))),
        many0(alt((alphanumeric1, tag("_")))),
    ))
    .parse(input)
}

fn ident_space(input: Span) -> IResult<Span, Span> {
    ws(identifier).parse(input)
}

fn numeric_literal_expression(input: Span) -> IResult<Span, Expr> {
    delimited(multispace0, float_value, multispace0).parse(input)
}

fn float_value(i: Span) -> IResult<Span, Expr> {
    let (r, v) = float(i)?;
    Ok((r, Expr::Literal(v as f64)))
}

fn var_ref(input: Span) -> IResult<Span, Expr> {
    let (r, res) = ident_space(input)?;
    Ok((r, Expr::Variable(res.to_string())))
}

// We parse any expr surrounded by parens, ignoring all whitespaces around those
fn parens(i: Span) -> IResult<Span, Expr> {
    let (r0, _) = multispace0(i)?;
    let (r, res) = delimited(tag("("), expr, tag(")")).parse(r0)?;
    let (r, _) = multispace0(r)?;
    Ok((r, res))
}

fn func_invoke(i: Span) -> IResult<Span, Expr> {
    let (r, ident) = ws(identifier).parse(i)?;
    func_args(ident)(r)
}

fn func_args<'a>(name: Span<'a>) -> impl Fn(Span<'a>) -> IResult<Span<'a>, Expr> {
    |i| {
        // println!("func_invoke ident: {}", ident);
        let (r, args) = delimited(
            multispace0,
            delimited(
                char('('),
                terminated(separated_list0(ws(char(',')), expr), opt(ws(char(',')))),
                char(')'),
            ),
            multispace0,
        )
        .parse(i)?;
        Ok((r, Expr::FnInvoke(name.to_string(), args, FnContext::new())))
    }
}

fn primary_expression(i: Span) -> IResult<Span, Expr> {
    alt((func_invoke, numeric_literal_expression, var_ref, parens)).parse(i)
}

// We read an initial factor and for each time we find
// a * or / operator followed by another factor, we do
// the math by folding everything
fn term(i: Span) -> IResult<Span, Expr> {
    let (r, init) = primary_expression(i)?;

    fold_many0(
        pair(alt((char('*'), char('/'))), primary_expression),
        move || init.clone(),
        move |acc, (op, val): (char, Expr)| {
            if op == '*' {
                Expr::Mul(Box::new(acc), Box::new(val))
            } else {
                Expr::Div(Box::new(acc), Box::new(val))
            }
        },
    )
    .parse(r)
}

fn expr(i: Span) -> IResult<Span, Expr> {
    let (r, init) = term(i)?;
    expr_rest(r, init)
}

fn expr_rest(i: Span, init: Expr) -> IResult<Span, Expr> {
    fold_many0(
        pair(alt((char('+'), char('-'))), term),
        move || init.clone(),
        move |acc, (op, val): (char, Expr)| {
            if op == '+' {
                Expr::Add(Box::new(acc), Box::new(val))
            } else {
                Expr::Sub(Box::new(acc), Box::new(val))
            }
        },
    )
    .parse(i)
}

/// Parses an assignment statement, e.g. "a = 1;". Note that the semicolon is required.
fn assign_stmt<'a>(name: Span<'a>) -> impl Fn(Span<'a>) -> IResult<Span<'a>, Stmt> {
    move |i| {
        let (r, _) = ws(tag("=")).parse(i)?;
        let (r, init) = expr(r)?;
        let (r, _) = ws(tag(";")).parse(r)?;
        Ok((r, Stmt::VarDef(name.to_string(), init)))
    }
}

fn ident_stmt(i: Span) -> IResult<Span, Stmt> {
    let (r, name) = ident_space(i)?;

    // We use partial application (curried functions) of parsers to avoid backtracking duplicate parsing
    // for ambiguous syntax. e.g. "a = 1;" and "a(1)" are a statement and a function call expression, respectively,
    // but it is not decided yet at the point just after the identifier "a" is parsed.
    alt((
        assign_stmt(name),
        // expr_to_stmt(func_args(name)),
        // expr_to_stmt(|i| expr_rest(i, Expr::Variable(name.to_string()))),
    ))
    .parse(r)
}

fn stmt(i: Span) -> IResult<Span, Stmt> {
    alt((ident_stmt, expr_to_stmt(expr))).parse(i)
}

/// A combinator that takes a parser `inner` and produces a parser that produces an expression statement instead.
fn expr_to_stmt<'a, F: 'a, E: ParseError<Span<'a>>>(
    mut inner: F,
) -> impl Parser<&'a str, Output = Stmt, Error = E>
where
    F: Parser<Span<'a>, Output = Expr, Error = E>,
{
    move |i| {
        let (r, ex) = inner.parse(i)?;
        Ok((r, Stmt::Expr(ex)))
    }
}

pub(crate) fn parse(i: Span) -> Result<Ast, String> {
    Ok(fold_many1(
        stmt,
        || vec![],
        |mut acc, cur| {
            acc.push(cur);
            acc
        },
    )
    .parse(i)
    .finish()
    .map_err(|e| e.to_string())?
    .1)
}

#[test]
fn test_primary() {
    let src = "a(1)";
    let ast = parse(src).unwrap();
    assert_eq!(
        ast,
        vec![Stmt::Expr(Expr::FnInvoke(
            "a".to_string(),
            vec![Expr::Literal(1.)],
            FnContext::new()
        ))]
    );

    let src = "a(1) * 10";
    let ast = parse(src).unwrap();
    assert_eq!(
        ast,
        vec![Stmt::Expr(Expr::Mul(
            Box::new(Expr::FnInvoke(
                "a".to_string(),
                vec![Expr::Literal(1.)],
                FnContext::new()
            )),
            Box::new(Expr::Literal(10.))
        ),)]
    );

    let src = "a(1) / 10";
    let ast = parse(src).unwrap();
    assert_eq!(
        ast,
        vec![Stmt::Expr(Expr::Div(
            Box::new(Expr::FnInvoke(
                "a".to_string(),
                vec![Expr::Literal(1.)],
                FnContext::new()
            )),
            Box::new(Expr::Literal(10.))
        ),)]
    );
}
