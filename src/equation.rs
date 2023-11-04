use crate::calculator::*;

pub struct Solver {
    pub lhs: Vec<ResultExpr>,
    pub rhs: Vec<ResultExpr>,

    variable: Option<char>
}

macro_rules! negate {
    ($tar: expr) => {
        ResultExpr::Operation(Operator::UnarySubtract, vec![$tar])
    };
}

impl Solver {
    pub fn new(lhs: Vec<ResultExpr>, rhs: Vec<ResultExpr>) -> Self {
        Self {
            lhs, rhs,
            variable: None,
        }
    }

    pub fn print(&self) {
        for (i, el) in self.lhs.iter().enumerate() {
            if i != 0 { print!(" + ") }
            el.print();
        }
        print!(" = ");
        for (i, el) in self.rhs.iter().enumerate() {
            if i != 0 { print!(" + ") }
            el.print();
        }
        println!();
    }

    pub fn solve_step(&mut self) -> bool {
        if self.variable.is_none() {
            let mut unknowns = Vec::new();
            let mut constant_terms = Vec::new();
            let mut variable = None;
            for i in self.lhs.iter() {
                let c = is_const_term(i);
                if c.is_some() {
                    unknowns.push(i.clone());

                    if variable.is_some() && c.is_some() {
                        if variable.unwrap() != c.unwrap() {
                            todo!("2 variables")
                        }
                    }
                    variable = c;
                }
            }
            for i in self.rhs.iter() {
                let c = is_const_term(i);
                if c.is_some() {
                    unknowns.push(negate!(i.clone()));

                    if variable.is_some() && c.is_some() {
                        if variable.unwrap() != c.unwrap() {
                            todo!("2 variables")
                        }
                    }
                    variable = c;
                }
            }
            for i in self.rhs.iter() {
                if is_const_term(i).is_none() {
                    constant_terms.push(i.clone())
                }
            }
            for i in self.lhs.iter() {
                if is_const_term(i).is_none() {
                    constant_terms.push(negate!(i.clone()))
                }
            }
            self.lhs = unknowns;
            self.rhs = constant_terms;
            self.variable = variable;
            return true
        }

        if let Some(rhs) = evaluate(&self.rhs) {
            self.rhs = vec![rhs];
            return true;
        }

        if let Some(lhs) = simplify(&self.lhs, self.variable.unwrap()) {
            self.lhs = vec![lhs];
            return true;
        }

        match &self.lhs[0] {
            ResultExpr::Operation(Operator::Multiply, ops) => {
                self.rhs[0] = ResultExpr::Operation(Operator::Divide, vec![self.rhs[0].clone(), ops[1].clone()]);
                self.lhs[0] = ops[0].clone();
                return true;
            },
            _ => ()
        }

        false
    }
}

pub fn is_const_term(i: &ResultExpr) -> Option<char> {
    match i {
        ResultExpr::Number(_)   => None,
        ResultExpr::Variable(c) => Some(*c),
        ResultExpr::Operation(_, ops) => ops.iter().fold(None, |acc, el| {
            let c = is_const_term(el);
            if acc.is_none() && c.is_none() {
                None
            } else if acc.is_none() && c.is_some() {
                c
            } else if acc.is_some() && c.is_none() {
                acc
            } else {
                acc
            }
        }),
        _ => todo!()
    }
}

fn evaluate(terms: &Vec<ResultExpr>) -> Option<ResultExpr> {
    let mut acc = Fraction::from(0);

    for i in terms.iter() {
        acc += i.evaluate();
    }

    let res = ResultExpr::Number(Number::Rational(acc));

    if terms.len() == 1 {
        if terms[0] == res {
            return None
        }
    }

    Some(res)
}

fn simplify(terms: &Vec<ResultExpr>, var: char) -> Option<ResultExpr> {
    let mut acc = Fraction::from(0);

    for i in terms.iter() {
        acc += i.get_unknown_multiple(true);
    }

    let res = if acc != Fraction::from(1) { ResultExpr::Operation(
        Operator::Multiply,
        vec![
            ResultExpr::Variable(var),
            ResultExpr::Number(Number::Rational(acc))
        ]
    )} else { ResultExpr::Variable(var) };

    if terms.len() == 1 {
        if terms[0] == res {
            return None
        }
    }

    Some(res)
}
