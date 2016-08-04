

# grass


To build one needs to install the `nightly` toolchain:

    grass$ rustup override set nightly

To build the project just do

    grass$ cargo build

    # or for much faster execution
    grass$ cargo build --release

To run `grass` use

    grass$ target/debug/grassc --sysroot ~/.multirust/toolchains/<your nightly> <target.rs>

