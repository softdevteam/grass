

# grass


To build one needs to install the `nightly` toolchain:

    grass$ rustup override set nightly

To run `grass` use

    grass$ target/debug/grassc --sysroot ~/.multirust/toolchains/<your nightly> <target.rs>

