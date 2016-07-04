#![feature(rustc_private)]
#![feature(box_syntax)]

extern crate grass;
extern crate rustc;
extern crate rustc_driver;
extern crate getopts;

use grass::bc::{translate, Context};

use rustc::session::Session;
use rustc_driver::{driver, CompilerCalls, Compilation};


struct GrassCompilerCalls;

impl<'a> CompilerCalls<'a> for GrassCompilerCalls {
    fn build_controller(
        &mut self,
        _: &Session,
        _: &getopts::Matches
    ) -> driver::CompileController<'a> {
        let mut control = driver::CompileController::basic();

        control.after_analysis.callback = Box::new(|state| {
            state.session.abort_if_errors();
            let map = state.mir_map.unwrap();
            let tcx = state.tcx.unwrap();
            let context = Context{tcx: tcx, map: &map};

            let (mut program, main) = translate::generate_bytecode(&context);
            // interpret::interpret(&mut program, main, tcx, map);
        });

        control.after_analysis.stop = Compilation::Stop;

        control
    }
}


fn main() {
    let args: Vec<String> = std::env::args().collect();
    rustc_driver::run_compiler(&args, &mut GrassCompilerCalls);
}
