

use bc::bytecode::OpCode;



pub type Trace = Vec<OpCode>;


fn next_with_pattern<F>(trace: &Trace, start: usize, pattern: F) -> Option<usize>
where F: Fn(&OpCode) -> bool {
    for (offset, opcode) in trace[start..].iter().enumerate() {
        if pattern(opcode) {
            return Some(start + offset)
        }
    }
    None
}


pub fn detect_unused_returns(trace: &mut Trace) {
    let len = trace.len();
    let ignore_filter = |opcode: &OpCode| match *opcode {
        OpCode::Return => false,
        _ => true
    };

    for (this_index, next_index) in (0..len-1).zip((1..len)) {
        let this = trace[this_index].clone();
        let next = trace[next_index].clone();

        if let OpCode::Return = next {
            if let Some(idx) = next_with_pattern(trace, next_index, &ignore_filter) {
                if let OpCode::Pop = trace[idx] {
                    if let OpCode::Tuple(0) = this {
                        trace[this_index] = OpCode::Noop;
                        trace[idx] = OpCode::Noop;
                    }
                }
            }
        }
    }
}

pub fn detect_unused_variables(trace: &Trace) -> Trace {
    // in the first pass all variables which are used (loaded) are marked
    // then all local variables get a new index assigned (e.g. LoadLocal(7) -> LoadLocal(5))
    // in the second pass these opcodes get written to the trace

    let mut used_vars: Vec<bool> = Vec::new();
    let mut new_positions: Vec<usize> = Vec::new();

    // detect used
    for opcode in trace {
        match *opcode {
            OpCode::StackFrame(size, argcount) => {
                used_vars.reserve(size);
                for idx in 0..size {
                    //can't optimize arguments away
                    used_vars.push(idx <= argcount);
                    new_positions.push(0);
                }
            },
            OpCode::LoadLocal(n) => {
                used_vars[n] = true;
            },
            _ => {}
        }
    }

    //remap LoadLocal index
    let mut unused_count = 0;
    for (idx, is_used) in used_vars.iter().enumerate() {
        if *is_used {
            new_positions[idx] = unused_count;
        } else {
            unused_count += 1;
        }
    }

    let mut new_trace = Vec::new();
    // remove unused
    for opcode in trace {
        let new_opcode = match *opcode {
            OpCode::StackFrame(size, argcount) => {
                OpCode::StackFrame(size-unused_count, argcount)
            },
            OpCode::StoreLocal(idx) => {
                let is_used = used_vars[idx];
                if is_used {
                    let offset = new_positions[idx];
                    OpCode::StoreLocal(idx - offset)
                } else {
                    OpCode::Pop
                }
            },
            OpCode::QuickStoreLocal(idx) => {
                let is_used = used_vars[idx];
                if is_used {
                    let offset = new_positions[idx];
                    OpCode::QuickStoreLocal(idx - offset)
                } else {
                    // QuickStoreLocal leaves value on stack
                    OpCode::None
                }

            },

            OpCode::LoadLocal(idx) => {
                let offset = new_positions[idx];
                OpCode::LoadLocal(idx - offset)
            },
            _ => {
                opcode.clone()
            }
        };
        new_trace.push(new_opcode);
    }

    new_trace
}