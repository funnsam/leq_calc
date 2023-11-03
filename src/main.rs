mod calculator;
mod equation;
//mod c;
fn main() {
    if std::env::args().count() != 2 {
        println!("\x1b[1;31mError:\x1b[0m 1 argument is required\n\x1b[1mUsage:\x1b[0m calc \"\x1b[1m[Expression]\x1b[0m\"");
        std::process::exit(-1);
    }
    let expr = match calculator::to_expr(&std::env::args().nth(1).unwrap()) {
        Ok(v) => v,
        Err(e) => {
            println!("\x1b[1;31mError:\x1b[0m {e}");
            std::process::exit(-1);
        },
    };

    let (lhs, rhs) = match expr {
        calculator::ResultExpr::Operation(calculator::Operator::Equals, ops) => (ops[0].clone(), ops[1].clone()),
        _ => {
            println!("\x1b[1;31mError:\x1b[0m expected an equation");
            std::process::exit(-1);
        },
    };

    let mut lterms = Vec::new();
    let mut rterms = Vec::new();
    calculator::seperate_terms(&lhs, &mut lterms, true);
    calculator::seperate_terms(&rhs, &mut rterms, true);
    let mut solver = equation::Solver::new(lterms, rterms);
    solver.print();

    while solver.solve_step() {
        solver.print();
    }
}
