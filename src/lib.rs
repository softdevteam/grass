#![feature(rustc_private)]

#![feature(plugin)]

 
//#![plugin(clippy)]
#![allow(doc_markdown, unneeded_field_pattern)]

#![allow(unused_imports, unused_variables, dead_code)]

#[macro_use]
extern crate log;

extern crate serialize as rustc_serialize;

extern crate rustc;
extern crate syntax;
extern crate rustc_const_math;
extern crate rustc_data_structures;

// this has to be first for macros to be available in other modules
#[macro_use]
mod util;

pub mod bc;
pub mod core;
pub mod interp;
pub mod trace;
