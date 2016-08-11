

debug:
	RUSTFLAGS="-C rpath" cargo build

release:
	RUSTFLAGS="-C rpath" cargo build --release
