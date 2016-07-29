
fn __assert(b: bool){ 1/(b as u8);}


fn test_eq() {
    __assert(true);
    // TODO
    __assert(!false);
    __assert(!false == true);
    __assert(!true == false);
    __assert(!!true);
    __assert(true == true);
    __assert(false == false);
    __assert(true != false);
    __assert(false != true);
}

fn test_var() {
    let a = 1;
    let b = 2;

    __assert(a == 1);
    __assert(b == 2);
    __assert(a == a);
    __assert(b == b);
    __assert(a != b);
    __assert(b != a);
}

fn test_simple_binop() {
    let a = 1;
    let b = 2;

    __assert(a + b == 3);
    __assert(a * b == 2);
    __assert(0 * a == 0);
}

fn test_shift() {
    __assert( 1 >> 2 == 0);
    __assert( 1 << 2 == 4);
}

fn test_bin_add() {
    __assert( 1 & 2 == 0);
    __assert( 1 & 3 == 1);
}

fn test_block() {
    let x = {
        let a = 1;
        let b = 2;
        a + b
    };

    __assert(x == 3);
}

fn main() {
    test_eq();
    test_var();
    test_simple_binop();
    test_shift();
    test_bin_add();
    test_block();
}