
fn __assert(_: bool){}

fn test_if() {
    let x = 42;

    if x == 42 {
        __assert(true);
    } else {
        __assert(false);
    }
}

fn test_loop() {
    let mut i = 0;
    while i < 10 {
        i += 1;
    }
    __assert(i == 10);
}


fn main() {
    test_if();
    test_loop();
}