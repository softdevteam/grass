
use std::rc::Rc;
use std::cell::RefCell;

use rustc::mir::mir_map::MirMap;
use rustc::mir::repr::BinOp;
use rustc::ty::TyCtxt;
use rustc::hir::def_id::DefId;

use bc::translate::Program;
use bc::bytecode::OpCode;
use core::objects::{Cell, R_BoxedValue, CallFrame,
    R_Pointer, R_Function, R_Struct,
    InstructionPointer};



pub struct Interpreter<'a, 'cx: 'a> {
    pub program: &'a mut Program<'a, 'cx>,
    pub stack: Vec<Cell>,
    pub stack_frames: Vec<CallFrame>,
}

/// Baseline interpreter.
impl<'a, 'cx> Interpreter<'a, 'cx> {
    fn new(program: &'a mut Program<'a, 'cx>) -> Self {
        Interpreter {
            program: program,
            stack: Vec::new(),
            stack_frames: Vec::new(),
        }
    }

    fn run(&mut self, main: DefId) {
        let main_func = self.program.get_func(main);

        self.stack_frames.push(CallFrame::new(None, main_func.locals_cnt));

        let mut pc: usize = 0;
        let mut func = main_func;

        loop {
            let opcode = func.opcodes[pc].clone();
            debug!("#EX {:?}", opcode);

            match opcode {
                OpCode::ConstValue(val) => {
                    self.stack.push(Cell::Owned(val));
                },

                OpCode::Tuple(size) => self.o_tuple(size),

                OpCode::TupleGet(idx) => self.o_tuple_get(idx),
                OpCode::TupleSet(idx) => self.o_tuple_set(idx),

                OpCode::Use => {
                    let val = self.stack.pop().unwrap().as_value();
                    self.stack.push(val);
                    // let r_val = match self.stack.pop().unwrap() {
                    //     R_BoxedValue::Ptr(r_ptr) => {
                    //         r_ptr.deref(self).clone()
                    //     },

                    //     r_val => r_val,
                    // };

                    // self.stack.push(r_val);
                },

                OpCode::Ref(..) => self.o_ref(),

                OpCode::Deref => self.o_deref(),

                OpCode::Load(local_index) => self.o_load(local_index),

                OpCode::Store(local_index) => self.o_store(local_index),

                OpCode::Call => {
                    // load and activate func
                    func = self.o_call(func.clone(), pc);
                    // jump to first instruction of function
                    // continue is necessary because else pc += 1 would be executed
                    pc = 0;
                    continue;
                },

                OpCode::Return => {
                    if let Some(ret) = self.o_return() {
                        func = ret.func;
                        pc = ret.pc;
                    } else {
                        break;
                    }
                },

                OpCode::Skip(n) => { pc += n; continue },

                OpCode::SkipIf(n) => {
                    let val = self.pop_value();
                    if let R_BoxedValue::Bool(b) = val {
                        if b {
                            pc += n;
                            continue;
                        }
                    } else {
                        panic!("expected bool, git {:?}", val);
                    }
                },

                OpCode::BinOp(kind) => self.o_binop(kind),
                OpCode::CheckedBinOp(kind) => self.o_checked_binop(kind),

                OpCode::Noop => (),

                _ => {
                    println!("XXX: {:?}", opcode);
                    unimplemented!()
                }
            }

            pc += 1;
        }

    }

    fn stack_ptr(&self) -> usize {
        self.stack_frames.len() - 1
    }

    // fn pop_address(&mut self) -> R_Pointer {
        // let r_val = self.stack.pop().unwrap();
        // if let R_BoxedValue::Ptr(ptr) = r_val {
        //     ptr
        // } else {
        //     panic!("expected pointer, got {:?}", r_val);
        // }
    // }

    // fn locals(&mut self) -> &mut Vec<R_BoxedValue> {
        // &mut self.stack_frames.last_mut().unwrap().locals
    // }

    fn active_frame(&self) -> &CallFrame {
        self.stack_frames.last().unwrap()
    }

    fn o_load(&mut self, local_idx: usize) {
        let cell_ptr = self.active_frame().locals[local_idx].clone();
        self.stack.push(Cell::Ref(cell_ptr))
    }

    fn o_store(&mut self, local_idx: usize) {
        let val = self.stack.pop().unwrap();
        let mut cell = self.active_frame().locals[local_idx].borrow_mut();
        *cell = val.unwrap_value();
    }

    fn o_ref(&mut self) {
        let addr = self.stack.pop().unwrap().as_ref();
        self.stack.push(addr);
    }

    fn o_deref(&mut self) {
        let address = self.stack.pop().unwrap().deref();
        self.stack.push(address);
    }

    fn o_call(&mut self, cur_func: Rc<R_Function>, cur_pc: usize) -> Rc<R_Function> {
        if let R_BoxedValue::Func(def_id) = self.stack.pop().unwrap().as_value().unwrap_value() {
            let func = self.program.get_func(def_id);
            let return_addr = InstructionPointer{ func: cur_func, pc: cur_pc };
            let mut frame = CallFrame::new(Some(return_addr), func.locals_cnt);
            for idx in (0..func.args_cnt).rev() {
                frame.locals[idx] = self.stack.pop().unwrap().to_ref();
            }

            self.stack_frames.push(frame);
            func
        } else {
            panic!("expected func");
        }
    }

    fn o_return(&mut self) -> Option<InstructionPointer> {
        match self.stack_frames.pop() {
            Some(frame) => frame.return_addr,
            None => None,
        }
    }

    fn o_tuple(&mut self, size: usize) {
        let mut tuple = R_Struct::tuple(size);
        for idx in (0..size).rev() {
            unimplemented!();
            // tuple.data[idx] = self.stack.pop().unwrap().as_value(self).unwrap_value();
        }
        self.stack.push(Cell::Owned(R_BoxedValue::Struct(tuple)));
    }

    fn o_tuple_set(&mut self, idx: usize) {
        unimplemented!()
        // let tuple_address = self.stack.pop().unwrap().unwrap_address();
        // let value = self.pop_stack_value();

        // match tuple_address {
        //     Address::StackLocal(addr) => {
        //         if let WrappedValue::Tuple(ref mut tuple) = self.w_stack[addr] {
        //             tuple.data[idx] = value;
        //         }
        //     },
        //     _ => panic!("can not load tuple at {:?}", tuple_address),
        // }
    }

    fn o_tuple_get(&mut self, idx: usize) {
        // self.stack.pop().unwrap().as_value()
        // match self.stack.pop().unwrap().unwrap_address() {
        if let R_BoxedValue::Struct(r_struct) = self.pop_value() {
            let ptr = r_struct.data[idx].clone();
            self.stack.push(ptr);
        } else {
            unimplemented!()
        }
    }

    fn pop_value(&mut self) -> R_BoxedValue {
        self.stack.pop().unwrap().as_value().unwrap_value()
    }

    fn o_binop(&mut self, kind: BinOp) {
        let val = self._do_binop(kind);
        self.stack.push(val);
    }

    fn o_checked_binop(&mut self, kind: BinOp) {
        //TODO: actually check binops
        let mut tuple = R_Struct::tuple(2);
        tuple.data[0] = self._do_binop(kind);
        tuple.data[1] = Cell::Owned(R_BoxedValue::Bool(true));
        self.stack.push(Cell::Owned(R_BoxedValue::Struct(tuple)));
    }

    fn _do_binop(&mut self, kind: BinOp) -> Cell {

        use core::objects::R_BoxedValue::*;
        use rustc::mir::repr::BinOp::*;

        let right = self.pop_value();
        let left = self.pop_value();

        // copied from miri
        macro_rules! int_binops {
            ($v:ident, $l:ident, $r:ident) => ({
                match kind {
                    Add    => $v($l + $r),
                    Sub    => $v($l - $r),
                    Mul    => $v($l * $r),
                    Div    => $v($l / $r),
                    Rem    => $v($l % $r),
                    BitXor => $v($l ^ $r),
                    BitAnd => $v($l & $r),
                    BitOr  => $v($l | $r),

                    // TODO(solson): Can have differently-typed RHS.
                    Shl => $v($l << $r),
                    Shr => $v($l >> $r),

                    Eq => Bool($l == $r),
                    Ne => Bool($l != $r),
                    Lt => Bool($l < $r),
                    Le => Bool($l <= $r),
                    Gt => Bool($l > $r),
                    Ge => Bool($l >= $r),
                }
            })
        }


        let val = Cell::Owned(match(left, right) {
            (I64(l), I64(r)) => int_binops!(I64, l, r),
            (U64(l), U64(r)) => int_binops!(U64, l, r),
            (Usize(l), Usize(r)) => int_binops!(Usize, l, r),

            // copied from miri
            (Bool(l), Bool(r)) => {
                Bool(match kind {
                    Eq => l == r,
                    Ne => l != r,
                    Lt => l < r,
                    Le => l <= r,
                    Gt => l > r,
                    Ge => l >= r,
                    BitOr => l | r,
                    BitXor => l ^ r,
                    BitAnd => l & r,
                    Add | Sub | Mul | Div | Rem | Shl | Shr =>
                        panic!("invalid binary operation on booleans: {:?}", kind),
                })

            },

            (l, r) => {
                println!("{:?} {:?}", l, r);
                unimplemented!();
            }
        });

        val
    }
}

pub fn run<'a, 'tcx>(
        program: &'a mut Program<'a, 'tcx>,
        main: DefId,
        tcx: TyCtxt<'a, 'tcx, 'tcx>,
        map: &MirMap<'tcx>
        ){

    let mut interpreter = Interpreter::new(program);

    interpreter.run(main);
}