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

            let node_id = state.session.entry_fn.borrow().unwrap().0;
            let def_id = tcx.map.local_def_id(node_id);
            let mut program = translate::generate_bytecode(&context, def_id);
        });

        control.after_analysis.stop = Compilation::Stop;

        control
    }
}


fn main() {
    let args: Vec<String> = std::env::args().collect();
    rustc_driver::run_compiler(&args, &mut GrassCompilerCalls);
}
