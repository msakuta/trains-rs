use nom::{
    Finish, IResult, Parser,
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, alphanumeric1, char, multispace0},
    combinator::{opt, recognize},
    error::ParseError,
    multi::{many0, separated_list0},
    number::complete::float,
    sequence::{delimited, pair, terminated},
};

use crate::app::heightmap::noise_expr::FnContext;

use super::noise_expr::Expr;

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
    let (r, res) = delimited(tag("("), primary_expression, tag(")")).parse(r0)?;
    let (r, _) = multispace0(r)?;
    Ok((r, res))
}

fn func_invoke(i: Span) -> IResult<Span, Expr> {
    let (r, ident) = ws(identifier).parse(i)?;
    // println!("func_invoke ident: {}", ident);
    let (r, args) = delimited(
        multispace0,
        delimited(
            char('('),
            terminated(
                separated_list0(ws(char(',')), primary_expression),
                opt(ws(char(','))),
            ),
            char(')'),
        ),
        multispace0,
    )
    .parse(r)?;
    Ok((r, Expr::FnInvoke(ident.to_string(), args, FnContext::new())))
}

fn primary_expression(i: Span) -> IResult<Span, Expr> {
    alt((func_invoke, numeric_literal_expression, var_ref, parens)).parse(i)
}

pub(crate) fn parse(i: Span) -> Result<Expr, String> {
    Ok(primary_expression(i).finish().map_err(|e| e.to_string())?.1)
}
