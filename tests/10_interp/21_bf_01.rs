
fn __assert(b: bool){ 1/(b as u8);}

const TAPESIZE: usize = 2048;
struct Tape {
    tape: [usize; TAPESIZE],
    position: usize,
}

fn get(tape: &Tape) -> usize {
    tape.tape[tape.position]
}

fn set(tape: &mut Tape, val: usize) {
    tape.tape[tape.position] = val
}

fn inc(tape: &mut Tape) {
    tape.tape[tape.position] += 1;
}

fn dec(tape: &mut Tape) {
    tape.tape[tape.position] -= 1;
}

fn advance(tape: &mut Tape) {
    tape.position += 1;
}

fn devance(tape: &mut Tape) {
    tape.position -= 1;
}


const EXIT: usize = 0;
const GET: usize = 1;
const SET: usize = 2;
const INC: usize = 3;
const DEC: usize = 4;
const ADV: usize = 5;
const DEV: usize = 6;
const PRINT: usize = 7;
const OUT: usize = 8;
const JFF: usize = 9;
const JTB: usize = 10;

fn main() {
    let program = [INC, INC, INC, INC, INC, INC, INC, INC, INC, INC, JFF + 208, PRINT, DEC, JTB + 160, PRINT, EXIT];


    let mut tape = Tape { tape: [0; TAPESIZE], position: 0 };
    let mut pc = 0;

    let mut check = 10;

    loop {
        let code = program[pc];

        if code == EXIT {
            break;
        } else if code == SET {

        } else if code == INC {
            inc(&mut tape);
        } else if code == DEC {
            dec(&mut tape);
        } else if code == ADV {
            advance(&mut tape);
        } else if code == DEV {
            devance(&mut tape);
        } else if code == PRINT {
            __assert(check == get(&tape));
            if check > 0 {
                check -= 1;
            }
       } else if code == OUT {
            // __out(get(&mut tape));
        } else if (code & 0xF) == JFF { //[
            if get(&tape) == 0 {
                pc = code >> 4usize;
            }
        } else if (code & 0xF) == JTB { //]
            if get(&tape) != 0 {
                pc = code >> 4usize;
            }
        }
        pc += 1;
    }

    __assert(check == 0);
}