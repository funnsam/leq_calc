use std::fmt::{Display, Formatter};
use logos::{Lexer, Logos, Span};
use fraction::GenericFraction;
pub type Fraction = GenericFraction::<u32>;

#[derive(Debug, Clone, PartialEq)]
pub enum Number {
    Rational(Fraction),
    Constant(Constant),
}

#[derive(Debug, Clone, PartialEq)]
enum Constant {
    Pi, Tau
}

#[derive(Debug, Clone, PartialEq)]
pub enum ResultExpr {
    Number(Number),
    Variable(char),
    Operation(Operator, Vec<ResultExpr>),
    Function(Function, Vec<ResultExpr>),
}

impl ResultExpr {
    pub fn print(&self) {
        match self {
            Self::Number(Number::Rational(frac)) => print!("\x1b[97m{frac}\x1b[0m"),
            Self::Number(Number::Constant(c)) => print!("\x1b[1;97m{c:?}\x1b[0m"),
            Self::Variable(c) => print!("\x1b[1;97m{c}\x1b[0m"),
            Self::Operation(op, ops) => {
                print!("(");
                if ops.len() >= 2 {
                    ops[0].print();
                    for i in ops.iter().skip(1) {
                        print!(" {op} ");
                        i.print();
                    }
                } else {
                    print!("{op}");
                    ops[0].print();
                }
                print!(")");
            },
            Self::Function(f, ops) => {
                print!("{f:?}(");
                for i in ops.iter() {
                    i.print();
                    print!(", ");
                }
                print!(")");
            },
        }
    }

    pub fn evaluate(&self) -> Fraction {
        match self {
            Self::Number(Number::Rational(n)) => n.clone(),
            Self::Operation(Operator::UnaryAdd, ops) => ops[0].evaluate(),
            Self::Operation(Operator::UnarySubtract, ops) => -ops[0].evaluate(),
            Self::Operation(Operator::Add, ops) => ops.iter().fold(Fraction::from(0), |acc, expr| acc + expr.evaluate()),
            Self::Operation(Operator::Subtract, ops) => ops.iter().fold(Fraction::from(0), |acc, expr| acc - expr.evaluate()),
            Self::Operation(Operator::Multiply, ops) => ops.iter().fold(Fraction::from(1), |acc, expr| acc * expr.evaluate()),
            Self::Operation(Operator::Divide, ops) => {
                let mut acc = ops[0].evaluate();
                for i in ops.iter().skip(1) {
                    acc /= i.evaluate()
                }

                acc
            },
            _ => todo!("{self:?}"),
        }
    }

    pub fn get_unknown_multiple(&self, sign: bool) -> Fraction {
        (match self {
            Self::Variable(_) => Fraction::from(1),
            Self::Number(Number::Rational(f)) => f.clone(),
            Self::Operation(Operator::UnaryAdd, ops) => ops[0].get_unknown_multiple(true),
            Self::Operation(Operator::UnarySubtract, ops) => ops[0].get_unknown_multiple(false),
            Self::Operation(Operator::Add, ops) => ops.iter().fold(Fraction::from(0), |acc, this| acc + this.get_unknown_multiple(true)),
            Self::Operation(Operator::Subtract, ops) => ops.iter().fold(Fraction::from(0), |acc, this| acc - this.get_unknown_multiple(true)),
            Self::Operation(Operator::Multiply, ops) => ops.iter().fold(Fraction::from(1), |acc, this| acc * this.get_unknown_multiple(true)),
            Self::Operation(Operator::Divide, ops) => {
                let mut acc = ops[0].get_unknown_multiple(sign);
                for i in ops.iter().skip(1) {
                    acc /= i.get_unknown_multiple(sign);
                }

                acc
            },
            _ => todo!()
        }) * sign.then_some(1).unwrap_or(-1)
    }

    pub fn contains_division(&self) -> bool {
        match self {
            Self::Operation(Operator::Divide, _) => true,
            Self::Operation(_, ops) => ops.iter().fold(false, |acc, el| acc | el.contains_division()),
            _ => false
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Operator {
    Add, Subtract, Multiply, Divide, Power,
    UnaryAdd, UnarySubtract,

    Equals,
}

impl Display for Operator {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        use Operator::*;
        match self {
            Add | UnaryAdd
                => write!(f, "+"),
            Subtract | UnarySubtract
                => write!(f, "-"),
            Multiply => write!(f, "×"),
            Divide   => write!(f, "÷"),
            Power    => write!(f, "^"),
            Equals   => write!(f, "="),
        }
    }
}

#[derive(Debug, Clone, Logos)]
#[logos(skip r"\s")]
enum Tokens {
    #[token("+")]
    Add,
    #[token("-")]
    Subtract,
    #[token("*")]
    #[token("×")]
    Multiply,
    #[token("/")]
    #[token("÷")]
    Divide,
    #[token("^")]
    Power,

    #[token("=")]
    EqualSign,

    #[token("(")]
    #[token("[")]
    #[token("{")]
    BracketStart,
    #[token(")")]
    #[token("]")]
    #[token("}")]
    BracketEnd,

    #[regex(r"(π|pi)" , callback = |_| Some(Number::Constant(Constant::Pi)))]
    #[regex(r"(τ|tau)", callback = |_| Some(Number::Constant(Constant::Tau)))]
    #[regex(r"[\d]+(\.[\d]+)?", callback = |lex| Some(Number::Rational(Fraction::from(lex.slice().parse::<f64>().ok()?))))]
    Number(Option<Number>),

    #[regex(r"[a-zA-Z]", callback = |lex| lex.slice().chars().nth(0).unwrap())]
    Variable(char),

    #[regex(r"(sin|cos|tan|rad|deg|sqrt|root|min|max)", callback = |lex| Function::from_str(lex.slice()))]
    Function(Function),
    #[token(",")]
    Comma,

    UnaryAdd,
    UnarySubtract,
    None
}

#[derive(Debug, Clone, PartialEq)]
pub enum Function {
    Sin, Cos, Tan,
    Rad, Deg,
    Sqrt, Root,
    Min, Max,
}

impl Function {
    fn from_str(s: &str) -> Self {
        use Function::*;
        match s.to_lowercase().as_str() {
            "sin" => Sin,
            "cos" => Cos,
            "tan" => Tan,

            "rad" => Rad,
            "deg" => Deg,

            "sqrt" => Sqrt,
            "root" => Root,

            "min" => Min,
            "max" => Max,
            _ => unreachable!()
        }
    }
    fn op_count(&self) -> usize {
        use Function::*;
        match self {
            Root | Min | Max => 2,
            _ => 1,
        }
    }
    fn execute(&self, ops: &mut Vec<f64>) -> Result<f64, String> {
        use Function::*;
        macro_rules! function_execute {
            ($oc:expr, $exec:block) => {
                if $oc > ops.len() {
                    return Err(format!("Not enough arguments to call {self:?}"));
                } else {
                    let r = $exec;
                    ops.truncate(ops.len()-$oc);
                    return Ok(r);
                }
            };
        }
        
        macro_rules! last_n {
            ($a:expr, $b:expr) => { $a[$a.len()-$b-1] };
        }

        match self {
            Sin => function_execute!(1, { last_n!(ops, 0).sin() }),
            Cos => function_execute!(1, { last_n!(ops, 0).cos() }),
            Tan => function_execute!(1, { last_n!(ops, 0).tan() }),

            Rad => function_execute!(1, { last_n!(ops, 0).to_radians() }),
            Deg => function_execute!(1, { last_n!(ops, 0).to_degrees() }),

            Sqrt => function_execute!(1, { last_n!(ops, 0).sqrt() }),
            Root => function_execute!(2, { last_n!(ops, 0).powf(1.0 / last_n!(ops, 1)) }),

            Min => function_execute!(2, { last_n!(ops, 0).min(last_n!(ops, 1)) }),
            Max => function_execute!(2, { last_n!(ops, 0).max(last_n!(ops, 1)) }),
        }
    }
}

impl Tokens {
    fn precedence(&self) -> usize {
        match self {
            Self::UnaryAdd | Self::UnarySubtract
                => 4,
            Self::Power
                => 3,
            Self::Multiply | Self::Divide
                => 2,
            Self::Add | Self::Subtract
                => 1,
            Self::EqualSign
                => 0,
            _   => panic!("{self:?}")
        }
    }

    fn is_left_ass(&self) -> bool {
        match self {
            Self::Add | Self::Subtract | Self::Multiply | Self::Divide | Self::EqualSign => true,
            Self::Power => false,
            _ => panic!("{self:?}")
        }
    }

    fn infer_mult(&self) -> bool {
        matches!(self, Self::Number(_) | Self::Variable(_) | Self::BracketEnd)
    }

    fn unary_form(&self) -> Result<Self, String> {
        match self {
            Self::Add => Ok(Self::UnaryAdd),
            Self::Subtract => Ok(Self::UnarySubtract),
            _ => Err(format!("Operation {self:?} cannot be an unary operation"))
        }
    }
}

#[derive(Debug)]
struct Parser {
    buf: Vec<(Result<Tokens, ()>, Span)>,
    idx: usize
}

impl Parser {
    fn new(lex: &mut Lexer<Tokens>) -> Self {
        let mut buf = Vec::new();
        while let Some(tok) = lex.next() {
            buf.push((tok, lex.span()));
        }
        Self {
            buf, idx: 0
        }
    }

    fn next<'a>(&'a mut self) -> Option<&'a (Result<Tokens, ()>, Span)> {
        self.idx += 1;
        self.buf.get(self.idx-1)
    }
}

pub fn to_expr(eval: &str) -> Result<ResultExpr, String> {
    let mut parser = Parser::new(&mut Tokens::lexer(eval));

    let mut out_queue: Vec<(Result<Tokens, ()>, Span)> = Vec::new();
    let mut op_stack : Vec<(Result<Tokens, ()>, Span)> = Vec::new();

    let mut last_tok = Tokens::None;

    while let Some(tok) = parser.next() {
        if tok.0.is_err() {
            return Err(format!("Unexpected `{}` (in {}..{})", &eval[tok.1.start..tok.1.end], tok.1.start, tok.1.end))
        }
        
        let t = tok.0.as_ref().unwrap().clone();
        match t {
            Tokens::Number(_) | Tokens::Variable(_) => {
                if last_tok.infer_mult() {
                    add_op(&mut out_queue, &mut op_stack, Tokens::Multiply, tok.1.clone());
                }
                out_queue.push(tok.clone())
            },
            Tokens::Function(_) => {
                if last_tok.infer_mult() {
                    add_op(&mut out_queue, &mut op_stack, Tokens::Multiply, tok.1.clone());
                }
                op_stack.push(tok.clone());
            },
            Tokens::Comma => {
                while let Some(op) = op_stack.last() {
                    if matches!(op.0, Ok(Tokens::BracketStart)) {
                        break;
                    }

                    out_queue.push(op_stack.pop().unwrap());
                }
            },
            Tokens::Add | Tokens::Subtract | Tokens::Multiply | Tokens::Divide | Tokens::Power | Tokens::EqualSign => {
                let o = if last_tok.infer_mult() { t.clone() } else { t.unary_form()? };
                add_op(&mut out_queue, &mut op_stack, o, tok.1.clone());
            },
            Tokens::BracketStart => {
                if last_tok.infer_mult() {
                    add_op(&mut out_queue, &mut op_stack, Tokens::Multiply, tok.1.clone());
                }
                op_stack.push(tok.clone());
            },
            Tokens::BracketEnd => {
                let mut got_start = false;
                while let Some(op) = op_stack.pop() {
                    if matches!(op.0.as_ref().unwrap(), Tokens::BracketStart) {
                        got_start = true;
                        break;
                    }
                    out_queue.push(op);
                }
                if !got_start {
                    return Err(format!("Mismatched brackets (can't find starting brackets that matches the ending bracket at char {})", tok.1.start))
                }
                if let Some(f) = op_stack.last() {
                    if matches!(f.0, Ok(Tokens::Function(_))) {
                        out_queue.push(op_stack.pop().unwrap());
                    }
                }
            },
            _ => unreachable!(),
        }
        last_tok = t;
    }

    while let Some(op) = op_stack.pop() {
        out_queue.push(op);
    }

    let mut tree = Vec::new();

    macro_rules! push_operator {
        ($name: ident, $n: expr) => {{
            let mut ops = Vec::with_capacity($n);
            for _ in 0..$n {
                ops.push(results_pop(&mut tree)?);
            }

            ops.reverse();

            tree.push(ResultExpr::Operation(Operator::$name, ops));
        }};
    }

    for i in out_queue.into_iter() {
        match i.0.unwrap() {
            Tokens::Number(n) => tree.push(ResultExpr::Number(n.unwrap())),
            Tokens::Variable(n) => tree.push(ResultExpr::Variable(n)),
            Tokens::Add           => push_operator!(Add, 2),
            Tokens::Subtract      => push_operator!(Subtract, 2),
            Tokens::Multiply      => push_operator!(Multiply, 2),
            Tokens::Divide        => push_operator!(Divide, 2),
            Tokens::Power         => push_operator!(Power, 2),
            Tokens::EqualSign     => push_operator!(Equals, 2),
            Tokens::UnaryAdd      => push_operator!(UnaryAdd, 1),
            Tokens::UnarySubtract => push_operator!(UnarySubtract, 1),
            Tokens::Function(f) => {
                let n = f.op_count();
                let mut ops = Vec::with_capacity(n);
                for _ in 0..n {
                    ops.push(results_pop(&mut tree)?);
                }

                ops.reverse();

                tree.push(ResultExpr::Function(f, ops));
            },
            Tokens::BracketStart => return Err(format!("Mismatched brackets (can't find ending brackets that matches the starting bracket at char {})", i.1.start)),
            _ => unreachable!()
        }
    }

    if tree.len() != 1 {
        return Err(format!("Result stack length doesn't match (expected 1, found {})", tree.len()))
    }

    Ok(tree[0].clone())
}

macro_rules! sign_to_op {
    ($sign: expr) => {
        if $sign {
            Operator::UnaryAdd
        } else {
            Operator::UnarySubtract
        }
    };
}

pub fn seperate_terms(tree: &ResultExpr, terms: &mut Vec<ResultExpr>, sign: bool) {
    match tree {
        ResultExpr::Operation(op, ops) => {
            match op {
                Operator::Add => {
                    seperate_terms(&ops[0], terms, sign);
                    seperate_terms(&ops[1], terms, sign);
                },
                Operator::UnaryAdd => {
                    seperate_terms(&ops[0], terms, true);
                },
                Operator::Subtract => {
                    seperate_terms(&ops[0], terms, sign);
                    seperate_terms(&ops[1], terms, !sign);
                },
                Operator::UnarySubtract => {
                    seperate_terms(&ops[0], terms, false);
                },
                _ => {
                    terms.append(&mut distributive_law(tree.clone(), sign).clone())
                },
            }
        },
        _ => terms.push(ResultExpr::Operation(sign_to_op!(sign), vec![tree.clone()])),
    }
}

fn distributive_law(term: ResultExpr, sign: bool) -> Vec<ResultExpr> {
    match term {
        ResultExpr::Operation(ref op, ref ops) => {
            if matches!(op, Operator::Multiply | Operator::Divide) {
                let is_rhs = super::equation::is_const_term(&ops[1]).is_some();
                match &ops[is_rhs as usize] {
                    ResultExpr::Operation(_op, _ops) => {
                        if matches!(_op, Operator::Add | Operator::Subtract) {
                            let mut lhs = Vec::new();
                            let mut rhs = Vec::new();
                            seperate_terms(&_ops[0], &mut lhs, true);
                            seperate_terms(&_ops[1], &mut rhs, matches!(_op, Operator::Add));
                            for i in lhs.iter_mut() {
                                let mut iops = vec![ResultExpr::Operation(sign_to_op!(sign), vec![ops[(!is_rhs) as usize].clone()]), i.clone()];
                                if !is_rhs { iops.reverse() }
                                *i = ResultExpr::Operation(op.clone(), iops)
                            }
                            for i in rhs.iter_mut() {
                                let mut iops = vec![ResultExpr::Operation(sign_to_op!(sign), vec![ops[(!is_rhs) as usize].clone()]), i.clone()];
                                if !is_rhs { iops.reverse() }
                                *i = ResultExpr::Operation(op.clone(), iops)
                            }

                            lhs.append(&mut rhs);
                            lhs
                        } else if op == _op {
                            let mut terms = Vec::new();
                            seperate_terms(&_ops[1], &mut terms, true);
                            for i in terms.iter_mut() {
                                *i = ResultExpr::Operation(op.clone(), vec![ResultExpr::Operation(sign_to_op!(sign), vec![ops[(!is_rhs) as usize].clone()]), _ops[0].clone(), i.clone()])
                            }

                            terms
                        } else {
                            vec![ResultExpr::Operation(sign_to_op!(sign), vec![term])]
                        }
                    },
                    _ => vec![ResultExpr::Operation(sign_to_op!(sign), vec![term])],
                }
            } else {
                vec![ResultExpr::Operation(sign_to_op!(sign), vec![term])]
            }
        },
        _ => vec![ResultExpr::Operation(sign_to_op!(sign), vec![term])],
    }
}

fn add_op(out_queue: &mut Vec<(Result<Tokens, ()>, Span)>,op_stack: &mut Vec<(Result<Tokens, ()>, Span)>, o1: Tokens, s: Span) {
    while let Some(op) = op_stack.last() {
        let o2 = op.0.as_ref().unwrap();
        if !matches!(o2, Tokens::BracketStart) && (o2.precedence() > o1.precedence() || (o2.precedence() == o1.precedence() && o1.is_left_ass())) {
            out_queue.push(op_stack.pop().unwrap());
        } else {
            break;
        }
    }
    op_stack.push((Ok(o1), s));
}

fn results_pop<A>(results: &mut Vec<A>) -> Result<A, String> {
    match results.pop() {
        Some(n) => Ok(n),
        None => Err(format!("Ran out of items while evaluating result stack"))
    }
}
