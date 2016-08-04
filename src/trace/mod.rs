
pub mod optimisation;

use std::io;
use std::io::Write;
use std::rc::Rc;
use std::collections::BTreeMap;

use core::objects::{R_BoxedValue, InstructionPointer, R_Function, CallFrame};
use bc::bytecode::{OpCode, Guard, InternalFunc};

use interp::{Interpreter, StackVal};

use self::optimisation as opt;

const HOT_LOOP_COUNT: usize = 5;

#[derive(Default)]
pub struct Tracer {
    traces: BTreeMap<usize, Vec<OpCode>>,

    active: Option<Vec<OpCode>>,
    loop_start: usize,

    // in_pc to count
    counter: BTreeMap<usize, usize>,

}


impl Tracer {

    pub fn new() -> Self {
        Tracer::default()
    }

    fn veryify_trace(&self, trace: &Vec<OpCode>) {
        for (i, opcode) in trace.iter().enumerate() {
            if let OpCode::Guard(..) = *opcode {
                match trace[i-1] {
                    OpCode::InternalFunc(InternalFunc::MergePoint) => {
                        panic!("got merge point, {:?}", self.traces.len());
                    },
                    _ => {}
                }
            }
        }
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
    //        check if counter reached threshold
    //        if so, mark merge point as start of a hot loop and start tracing mode

    pub fn handle_mergepoint(&mut self, interp: &mut Interpreter, in_pc: usize) -> Option<InstructionPointer> {

        // #1
        if self.traces.contains_key(&in_pc) {
            //trace: &'a mut Trace

            let guard = self.run_trace(interp, in_pc);
            interp.stack.push(StackVal::Owned(R_BoxedValue::Bool(!guard.expected)));

            // we just executed a subtrace whilst tracing
            if self.active.is_some() {
                let mut active = &mut self.active.as_mut().unwrap();
                active.push(OpCode::RunTrace(in_pc));
                active.push(OpCode::ConstValue(R_BoxedValue::Bool(!guard.expected)));
            }

            Some(guard.recovery)
        }

        // #2
        else if self.active.is_some() && in_pc == self.loop_start {
            let mut active = self.active.take().unwrap();
            self.veryify_trace(&active);

            for opcode in &active {
                debug!("#TR {:?}", opcode);
            }

            self.traces.insert(in_pc, active);
            assert!(self.active.is_none());

            None
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

            None
        }
    }

    pub fn run_trace(&self, interp: &mut Interpreter, in_pc: usize) -> Guard {
        let trace = self.traces.get(&in_pc).unwrap();
        let runner = TraceRunner { trace: trace, tracer: self };
        runner.exec(interp)

    }

    pub fn trace_opcode(&mut self, interp: &mut Interpreter, opcode: &OpCode, ip: InstructionPointer) {
        if self.active.is_none() {
            return;
        }

        match *opcode {
            // we can entirely ignore simple jumps
            OpCode::Skip(_) | OpCode::JumpBack(_) => {},

            // we still can ignore the jump part of conditional jumps
            // but we have to replace the if part with a guard
            OpCode::SkipIf(target) | OpCode::JumpBackIf(target) => {
                let r_cond = interp.stack.last().unwrap().clone().into_owned().unwrap_value();
                if let R_BoxedValue::Bool(boolean) = r_cond {
                    let mut active = self.active.as_mut().unwrap();
                    active.push(OpCode::Guard(Guard {
                        expected: boolean,
                        recovery: ip.clone(),
                    }));
                } else {
                    panic!("expected bool, got {:?}", r_cond);
                }
            },

            OpCode::Call => {
                // we need to remove the function we would have called
                let mut active = self.active.as_mut().unwrap();
                let val = active.pop().unwrap().clone();

                if let OpCode::ConstValue(R_BoxedValue::Func(target)) = val {
                    let func = interp.program.get_func(target);
                    active.push(OpCode::FlatCall(target, ip, func));
                } else {
                    panic!("expected func, got {:?}", val);
                }
            },

            ref oc => {
                let mut active = self.active.as_mut().unwrap();
                active.push(oc.clone());
            },
        }
    }
}



pub struct TraceRunner<'a> {
    pub trace: &'a Vec<OpCode>,
    pub tracer: &'a Tracer,
}

impl<'a> TraceRunner<'a> {

    fn exec(&self, interp: &mut Interpreter) -> Guard {
        // the trace is always exited through a failed guard check
        let guard = self.eval_trace(interp);
        // self.deoptimise(guard);
        guard
    }

    fn deoptimise(&self, guard: Guard) {

    }

    fn eval_trace(&self, interp: &mut Interpreter) -> Guard {

        loop { for opcode in self.trace {
            match *opcode {
                OpCode::FlatCall(ref def_id, _, _) => debug!("#TE FlatCall({:?})", def_id),
                _ => debug!("#TE {:?}", opcode),
            }

            match *opcode {
                OpCode::Guard(ref guard) => {
                    let failed = self.o_guard(interp, guard);
                    if failed {
                        return guard.clone();
                    }
                },

                OpCode::RunTrace(in_pc) => {
                    self.tracer.run_trace(interp, in_pc);
                },

                OpCode::Panic => panic!("assertion failed"),

                OpCode::ConstValue(ref val) => {
                    interp.stack.push(StackVal::Owned(val.clone()));
                },

                OpCode::Tuple(size) => interp.o_tuple(size),
                OpCode::TupleInit(size) => interp.o_tuple_init(size),
                OpCode::TupleGet(idx) => interp.o_tuple_get(idx),
                OpCode::TupleSet(idx) => interp.o_tuple_set(idx),

                // XXX: proper implementation of unsize
                OpCode::Unsize
                | OpCode::Use => {
                    let val = interp.stack.pop().unwrap().into_owned();
                    interp.stack.push(val);
                },

                OpCode::Ref(..) => interp.o_ref(),

                OpCode::Deref => interp.o_deref(),

                OpCode::Load(local_index) => interp.o_load(local_index),

                OpCode::Store(local_index) => interp.o_store(local_index),

                OpCode::InternalFunc(ref kind) => {
                    match *kind {
                        InternalFunc::Out => {
                            let val = interp.pop_value();
                            println!("#OUT {:?}", val);
                        },

                        InternalFunc::MergePoint => {
                            interp.stack.pop().unwrap();
                        },

                        InternalFunc::Print => {
                            let val = interp.pop_value();
                            if let R_BoxedValue::Usize(n) = val {
                                print!("{}", n as u8 as char);
                            } else {
                                panic!("expected number, got {:?}", val);
                            }
                        },
                        InternalFunc::Assert => {
                            let val = interp.pop_value();
                            if let R_BoxedValue::Bool(test_success) = val {
                                if test_success {
                                    print!(".");
                                } else {
                                    print!("E");
                                }
                                io::stdout().flush().ok().expect("Could not flush stdout");
                            }
                        },
                    }
                },

                OpCode::FlatCall(_, ref ip, ref func) => {
                    self.o_flat_call(interp, ip.clone(), func.clone());
                },

                OpCode::Return => {
                    interp.stack_frames.pop();
                },

                OpCode::GetIndex => interp.o_get_index(),
                OpCode::AssignIndex => interp.o_assign_index(),

                OpCode::Array(size) => interp.o_array(size),

                OpCode::Repeat(size) => interp.o_repeat(size),

                OpCode::Len => interp.o_len(),

                OpCode::BinOp(kind) => interp.o_binop(kind),
                OpCode::CheckedBinOp(kind) => interp.o_checked_binop(kind),

                OpCode::Not => interp.o_not(),
                OpCode::Neg => unimplemented!(),

                _ => unimplemented!(),
            }
        } }
    }
}

// seprate opcode callbacks
impl<'a> TraceRunner<'a> {
    fn o_guard(&self, interp: &mut Interpreter, guard: &Guard) -> bool {
        let val = interp.pop_value();
        if let R_BoxedValue::Bool(b) = val {
            b != guard.expected
        } else {
            panic!("expected bool, got {:?}", val);
        }
    }

    fn o_flat_call(&self, interp: &mut Interpreter, return_addr: InstructionPointer, func: Rc<R_Function>) {
        let mut frame = CallFrame::new(Some(return_addr), func.locals_cnt);
        for idx in (0..func.args_cnt).rev() {
            frame.locals[idx] = interp.stack.pop().unwrap().into_cell().unwrap_cell();
        }
        interp.stack_frames.push(frame);
    }
}

