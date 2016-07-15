#![feature(rustc_private)]

#![feature(plugin)]

#![plugin(clippy)]

#[macro_use]
extern crate log;

extern crate serialize as rustc_serialize;

extern crate rustc;
extern crate syntax;
extern crate rustc_const_math;
extern crate rustc_data_structures;

pub mod bc;
pub mod core;
pub mod interp;

