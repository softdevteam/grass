
fn __assert(b: bool){ 1/(b as u8);}

fn test_if() {
    let x = 42;

    if x == 42 {
        __assert(true);
    } else {
        __assert(false);
    }
}

//
fn test_if2() {
    let x = 42;
    let mut inner = 0;
    if x == 42 {
        __assert(true);
        inner = 99;
        __assert(true);
    }

    __assert(inner == 99);
}

fn test_loop() {
    let mut i = 0;
    while i < 10 {
        i += 1;
    }
    __assert(i == 10);
}

fn test_continue() {
    let mut i = 0;
    let mut sum = 0;
    while i < 100 {
        if i % 2 == 1 {
            i += 1;
            continue;
        }
        sum += i;
        i += 1;
    }

    __assert(sum == 2450)
}

fn test_break() {
    let mut i = 0;
    while i < 10 {
        if i == 6 {
            break;
        }
        i += 1;
    }
    __assert(i == 6);
}


fn main() {
    test_if();
    test_if2();
    test_loop();
    test_continue();
    test_break();
}