

fn __assert(b: bool){ 1/(b as u8);}


fn second(list: &[u32]) -> u32 {
    list[1]
}

fn main() {
    let x = [4, 5, 6];
    let z = second(&x);

    __assert(z == 5);
}