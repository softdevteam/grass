
use std::rc::Rc;
use std::cell::RefCell;
use std::io;
use std::io::Write;

use rustc::mir::mir_map::MirMap;
use rustc::mir::repr::BinOp;
use rustc::ty::TyCtxt;
use rustc::hir::def_id::DefId;

use bc::translate::Program;
use bc::bytecode::{OpCode, InternalFunc};
use core::objects::{R_BoxedValue, CallFrame,
    R_Pointer, R_Function, R_Struct,
    InstructionPointer};


#[derive(Debug, Clone, RustcEncodable, RustcDecodable, PartialEq)]
pub enum StackVal {
    Owned(R_BoxedValue),
    Ref(Rc<RefCell<R_BoxedValue>>),
}

impl StackVal {

    // to_value should be clone
    pub fn into_owned(self) -> Self {
        match self {
            StackVal::Owned(..) => self,
            StackVal::Ref(cell) => {
                StackVal::Owned(cell.borrow().clone())
            }
        }
    }

    pub fn into_cell(self) -> Self {
        match self {
            StackVal::Owned(boxed) => StackVal::Ref(Rc::new(RefCell::new(boxed))),
            StackVal::Ref(..) => self,
        }
    }

    // self has to be owned
    pub fn unwrap_value(self) -> R_BoxedValue {
        if let StackVal::Owned(val) = self {
            val
        } else {
            panic!("expected owned val");
        }
    }

    pub fn unwrap_cell(self) -> Rc<RefCell<R_BoxedValue>> {
        if let StackVal::Ref(boxed) = self {
            boxed
        } else {
            panic!("expected ref val");
        }
    }

    /// Address as Value
    pub fn into_pointer(self) -> Self {
        let cell = self.unwrap_cell();
        StackVal::Owned(R_BoxedValue::Ptr(R_Pointer{cell:cell}))
    }

    /// Deref pointer
    pub fn deref(self) -> Self {
        // self contains an owned R_Pointer
        if let R_BoxedValue::Ptr(ptr) = self.into_owned().unwrap_value() {
            StackVal::Ref(ptr.cell)
        } else {
            panic!("expected val to be pointer");
        }
    }
}



pub struct Interpreter<'a, 'cx: 'a> {
    pub program: &'a mut Program<'a, 'cx>,
    pub stack: Vec<StackVal>,
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
            // debug!("#ST {:?}", self.active_frame().locals);
            debug!("#EX {:?}", opcode);

            match opcode {
                OpCode::Panic => panic!("assertion failed"),

                OpCode::ConstValue(val) => {
                    self.stack.push(StackVal::Owned(val));
                },

                OpCode::Tuple(size) => self.o_tuple(size),
                OpCode::TupleInit(size) => self.o_tuple_init(size),
                OpCode::TupleGet(idx) => self.o_tuple_get(idx),
                OpCode::TupleSet(idx) => self.o_tuple_set(idx),

                // XXX: proper implementation of unsize
                OpCode::Unsize
                | OpCode::Use => {
                    let val = self.stack.pop().unwrap().into_owned();
                    self.stack.push(val);
                },

                OpCode::Ref(..) => self.o_ref(),

                OpCode::Deref => self.o_deref(),

                OpCode::Load(local_index) => self.o_load(local_index),

                OpCode::Store(local_index) => self.o_store(local_index),

                OpCode::InternalFunc(kind) => {
                    match kind {
                        InternalFunc::Out => {
                            let val = self.pop_value();
                            println!("#OUT {:?}", val);
                        },
                        InternalFunc::MergePoint => (),

                        InternalFunc::Print => {
                            let val = self.pop_value();
                            if let R_BoxedValue::Usize(n) = val {
                                print!("{}", n as u8 as char);
                            } else {
                                panic!("expected number, got {:?}", val);
                            }
                        },
                        InternalFunc::Assert => {
                            let val = self.pop_value();
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
                OpCode::JumpBack(n) => { pc -= n; continue },

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
                OpCode::JumpBackIf(n) => {
                    let val = self.pop_value();
                    if let R_BoxedValue::Bool(b) = val {
                        if b {
                            pc -= n;
                            continue;
                        }
                    } else {
                        panic!("expected bool, git {:?}", val);
                    }
                },

                OpCode::GetIndex => self.o_get_index(),
                OpCode::AssignIndex => self.o_assign_index(),

                OpCode::Array(size) => self.o_array(size),

                OpCode::Repeat(size) => self.o_repeat(size),

                OpCode::Len => self.o_len(),

                OpCode::BinOp(kind) => self.o_binop(kind),
                OpCode::CheckedBinOp(kind) => self.o_checked_binop(kind),

                OpCode::Not => self.o_not(),
                OpCode::Neg => unimplemented!(),
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

    fn active_frame(&self) -> &CallFrame {
        self.stack_frames.last().unwrap()
    }

    fn o_load(&mut self, local_idx: usize) {
        let cell_ptr = self.active_frame().locals[local_idx].clone();
        self.stack.push(StackVal::Ref(cell_ptr))
    }

    fn o_store(&mut self, local_idx: usize) {
        let val = self.stack.pop().unwrap();
        let mut cell = self.active_frame().locals[local_idx].borrow_mut();
        *cell = val.unwrap_value();
    }

    fn o_ref(&mut self) {
        let addr = self.stack.pop().unwrap().into_pointer();
        self.stack.push(addr);
    }

    fn o_deref(&mut self) {
        let address = self.stack.pop().unwrap().deref();
        self.stack.push(address);
    }

    fn o_call(&mut self, cur_func: Rc<R_Function>, cur_pc: usize) -> Rc<R_Function> {
        if let R_BoxedValue::Func(def_id) = self.stack.pop().unwrap().into_owned().unwrap_value() {
            let func = self.program.get_func(def_id);
            let return_addr = InstructionPointer{ func: cur_func, pc: cur_pc };
            let mut frame = CallFrame::new(Some(return_addr), func.locals_cnt);
            for idx in (0..func.args_cnt).rev() {
                frame.locals[idx] = self.stack.pop().unwrap().into_cell().unwrap_cell();
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
        self.stack.push(StackVal::Owned(R_BoxedValue::Struct(tuple)));
    }

    fn o_tuple_init(&mut self, idx: usize) {
        let val = self.pop_value();
        if let R_BoxedValue::Struct(ref mut tuple) = self.stack.last().unwrap().clone().unwrap_value() {
            tuple.set(idx, val);
        } else {
            panic!("tuple init");
        }
    }

    fn o_tuple_set(&mut self, idx: usize) {
        let boxed_tuple = self.pop_value();
        let val = self.pop_value();

        if let R_BoxedValue::Struct(mut tuple) = boxed_tuple{
            tuple.set(idx, val);
        } else {
            panic!("expected struct, got {:?}", boxed_tuple);
        }
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
        // self.stack.pop().unwrap().into_owned()
        // match self.stack.pop().unwrap().unwrap_address() {
        let val = self.pop_value();
        if let R_BoxedValue::Struct(r_struct) = val{
            let ptr = r_struct.data[idx].clone();
            self.stack.push(StackVal::Ref(ptr));
        } else {
            panic!("expected struct got {:?}", val);
        }
    }

    fn load_const(&mut self, def_id: DefId) -> R_BoxedValue {
        let func = self.program.get_func(def_id);
        if let OpCode::ConstValue(ref val) = func.opcodes[0] {
            val.clone()
        } else {
            panic!("expected const");
        }
    }

    fn pop_value(&mut self) -> R_BoxedValue {
        let val = self.stack.pop().unwrap().into_owned().unwrap_value();
        if let R_BoxedValue::Static(def_id) = val {
            self.load_const(def_id)
        } else {
            val
        }
    }

    fn o_binop(&mut self, kind: BinOp) {
        let val = self._do_binop(kind);
        self.stack.push(StackVal::Owned(val));
    }

    fn o_checked_binop(&mut self, kind: BinOp) {
        //TODO: actually check binops
        let mut tuple = R_Struct::tuple(2);
        *tuple.data[0].borrow_mut() = self._do_binop(kind);
        // false == no error
        *tuple.data[1].borrow_mut() = R_BoxedValue::Bool(false);
        self.stack.push(StackVal::Owned(R_BoxedValue::Struct(tuple)));
    }

    fn _do_binop(&mut self, kind: BinOp) -> R_BoxedValue {

        use core::objects::R_BoxedValue::*;
        use rustc::mir::repr::BinOp::*;

        let right = self.pop_value();
        let left = self.pop_value();

        debug!("#EX2 left: {:?}, right: {:?} ", left, right);
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


        match(left, right) {
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
        }
    }

    fn o_not(&mut self) {
        if let R_BoxedValue::Bool(boolean) = self.pop_value() {
            self.stack.push(StackVal::Owned(R_BoxedValue::Bool(!boolean)));
        } else {
            panic!("expected bool");
        }
    }

    fn o_get_index(&mut self) {
        let target = self.pop_value();
        let index = self.pop_value();
        if let (R_BoxedValue::Struct(mut r_struct), R_BoxedValue::Usize(idx)) = (target, index) {
            let val = r_struct.get(idx);
            self.stack.push(StackVal::Ref(val));
        } else {
            panic!("error");
        }
    }

    fn o_assign_index(&mut self) {
        let target = self.pop_value();
        let index = self.pop_value();
        let val = self.pop_value();
        if let (R_BoxedValue::Struct(mut r_struct), R_BoxedValue::Usize(idx)) = (target, index) {
            r_struct.set(idx, val);
        } else {
            panic!("error");
        }
    }

    fn o_array(&mut self, size: usize) {
        let mut obj = R_Struct::with_size(size);
        for idx in (0..size).rev() {
            let val = self.pop_value();
            obj.set(idx, val.clone());
        }
        self.stack.push(StackVal::Owned(R_BoxedValue::Struct(obj)));
    }

    fn o_repeat(&mut self, size: usize) {
        let val = self.pop_value();

        let mut obj = R_Struct::with_size(size);
        for idx in 0..size {
            obj.set(idx, val.clone());
        }

        self.stack.push(StackVal::Owned(R_BoxedValue::Struct(obj)));
    }

    fn o_len(&mut self) {
        let x = self.pop_value();
        if let R_BoxedValue::Struct(s) = x {
            self.stack.push(StackVal::Owned(R_BoxedValue::Usize(s.data.len())));
        } else {
            panic!("can't get len of {:?}", x);
        }
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