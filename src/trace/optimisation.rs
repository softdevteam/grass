

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
