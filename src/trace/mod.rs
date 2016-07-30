
use std::collections::BTreeMap;

use core::objects::{R_BoxedValue, InstructionPointer};
use bc::bytecode::{OpCode, Guard};
use interp::Interpreter;


const HOT_LOOP_COUNT: usize = 100;

#[derive(Default)]
pub struct Tracer {
    traces: BTreeMap<usize, Trace>,

    active: Option<Vec<OpCode>>,
    loop_start: usize,

    // in_pc to count
    counter: BTreeMap<usize, usize>,

}


impl Tracer {

    pub fn new() -> Self {
        Tracer::default()
    }


    // there are three different cases what to do when encountering a merge point
    // in normal execution
    //
    // #1 There is a trace for the particular merge point available
    //      => execute trace
    //
    // #2 Tracing is active
    //      => check if we can finish the trace at this point (loop is closed)
    //         if so, finalise trace and go #1
    //
    // #3 Vanilla execution mode
    //     => increase counter for this merge point
    //        check if counter reached threshhold
    //        if so, mark merge point as start of a hot loop and start tracing mode

    pub fn handle_mergepoint(&mut self, interp: &mut Interpreter, in_pc: usize) {

        // #1
        // thanks borrow checker
        if self.traces.contains_key(&in_pc) {
            let mut trace = self.traces.get_mut(&in_pc).unwrap();
            trace.exec_for(interp);
        }
        // #2
        else if self.active.is_some() && in_pc == self.loop_start {
            let mut active = self.active.take().unwrap();
            self.traces.insert(in_pc, Trace { trace: active });
        }

        // #3
        else {
            let count = {
                let count = self.counter.entry(in_pc).or_insert(0);
                *count += 1;
                *count
            };

            if count > HOT_LOOP_COUNT {
                self.active = Some(Vec::new());
                self.counter.clear();
                self.loop_start = in_pc;
            }
        }
    }

    // recovery: InstructionPointer)
    pub fn trace_opcode(&mut self, interp: &mut Interpreter, opcode: &OpCode) {
        if self.active.is_none() {
            return;
        }

        match *opcode {
            // we can entirely ignore simple jumps
            OpCode::Skip(_) | OpCode::JumpBack(_) => {},

            // we still can ignore the jump part of conditional jumps
            // but we have to replace the if part with a guard
            OpCode::SkipIf(target) | OpCode::JumpBackIf(target) => {
                let r_cond = interp.stack.last().unwrap().clone().unwrap_value();
                if let R_BoxedValue::Bool(boolean) = r_cond {
                    // self.trace.unwrap().add(OpCode::Guard(Guard {
                    //     expected: boolean,
                    //     recovery: recovery
                    // }));
                }
            },

            ref oc => {
                self.active.as_mut().unwrap().push(oc.clone());
            },
        }
    }
}

#[derive(Default)]
pub struct Trace {
    pub trace: Vec<OpCode>,
}

impl Trace {

    fn exec_for(&mut self, interp: &mut Interpreter) {
        // the trace is always exited through a failed guard check
        let guard = self.eval_trace();
        self.deoptimise(interp, guard);
    }

    fn deoptimise(&mut self, interp: &mut Interpreter, guard: Guard) {

    }

    fn eval_trace(&mut self) -> Guard {
        unimplemented!();

        loop {
            for opcode in &self.trace {
                match *opcode {
                    _ => unimplemented!(),
                }
            }
        }
    }
}
