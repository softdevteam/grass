
fn __assert(b: bool){ 1/(b as u8);}

fn identity(x: u32) -> u32 {
    x
}

fn fib(n: u32) -> u32  {
    if n == 0 || n == 1 {
        n
    } else {
        fib(n -1) + fib(n -2)
    }
}



fn test_func() {
    __assert(identity(1) == 1);
    __assert(identity(identity(2)) == identity(2));
}

fn test_fib() {
    __assert(fib(0) == 0);
    __assert(fib(1) == 1);
    __assert(fib(2) == 1);
    __assert(fib(3) == 2);
    __assert(fib(4) == 3);
    __assert(fib(5) == 5);
    __assert(fib(10) == 55);
}



fn main() {
    test_func();
    test_fib();
}