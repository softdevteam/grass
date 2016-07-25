

// // impl FunctionTable {

// // }


//     fn run(&mut self, main: DefId) {
//         let main_func = self.program.get_func(main);
//         self.eval_func(main_func);

//         println!("{} traces generated", self.traces.len());
//         for trace in self.traces.values() {
//             print!("{:?}, ", trace.len());
//         }
//         println!("");
//         println!("{:?}", self.traces);
//     }


//     fn to_value(&mut self, data: &StackData) -> WrappedValue {
//         match data {
//             &StackData::Value(ref v) => v.clone(),
//             &StackData::Pointer(Address::StackLocal(ref other)) => {
//                 self.w_stack[*other].clone()
//             },
//             &StackData::Pointer(Address::StaticFunc(ref def_id)) => {
//                 WrappedValue::Address(Address::StaticFunc(def_id.clone()))
//             },
//             &StackData::Pointer(Address::StackComplex(a, b)) => {
//                 let tuple = self.w_stack[a].unwrap_tuple();
//                 tuple.data[b].clone()
//             }
//             _ => panic!("should not load interpreter level object {:?}", data)
//         }
//     }

//     fn pop_stack_value(&mut self) -> WrappedValue {
//         let something = self.stack.pop().unwrap();
//         self.to_value(&something)
//     }

//     //aquire space on the stack ahead of the current stack pointer
//     fn o_stackframe(&mut self, func_stacksize: usize) {
//         // aquire space on the stack
//         for _ in self.w_stack.len() .. self.w_stack_pointer + func_stacksize {
//             self.w_stack.push(WrappedValue::None);
//         }

//         if let Some(&StackData::ArgCount(n)) = self.stack.last() {
//             self.stack.pop();
//             for i in 0..n {
//                 let something = self.stack.pop().unwrap();
//                 let val = self.to_value(&something);
//                 self.w_stack[self.w_stack_pointer + i] = val;
//             }
//         } else {
//             // println!("expected ArgCount, got {:?}", self.stack.last());
//         }
//     }

//     fn o_return(&mut self) {
//         // can't return from main
//         if self.w_stack_pointer != 0 {
//             let old_pointer = self.w_stack_pointer_stack.pop().unwrap();
//             self.w_stack_pointer = old_pointer;
//         }
//     }


//     fn eval_func(&mut self, func: Rc<Function<'a>>) {

//         let mut func = func;
//         let mut pc: usize = 0;

//         loop {

//             let opcode = func[pc].clone();
//             debug!("#EX {:?}", opcode);

//             match *opcode {

//                 OpCode::RETURN => {
//                     self.o_return();
//                     if self.call_stack.len() == 0 {
//                         break;
//                     }

//                     let target = self.call_stack.pop().unwrap();
//                     func = target.0;
//                     pc = target.1;
//                     // break
//                 },

//                 OpCode::LoadFunc(defid) => {
//                     self.stack.push(StackData::Pointer(Address::StaticFunc(defid)));
//                 },

//                 OpCode::ArgCount(size) => {
//                     self.stack.push(StackData::ArgCount(size));
//                 },

//                 OpCode::InternalFunc(InternalFunc::Out) => {
//                     let val = self.pop_stack_value();
//                     if let WrappedValue::Usize(n) = val {
//                         println!("BF: {}", n);
//                     };
//                 },

//                 OpCode::InternalFunc(InternalFunc::Print) => {
//                     let val = self.pop_stack_value();
//                     if let WrappedValue::Usize(n) = val {
//                         print!("{}", n as u8 as char);
//                     };

//                 },

//                 OpCode::InternalFunc(InternalFunc::MergePoint) => {
//                     let val = self.pop_stack_value();
//                     if let WrappedValue::Usize(in_pc) = val {
//                         // println!("met_merge_point {:?}", in_pc);
//                         if self.traces.contains_key(&in_pc) {
//                             if let Some(guard) = self.eval_trace(in_pc) {
//                                 func = guard.recovery;
//                                 pc = guard.pc;
//                                 self.stack.push(StackData::Value(WrappedValue::Bool(!guard.expected)));
//                                 // println!("FAILED IN {:?}", func[pc]);
//                                 continue;
//                             }
//                         } else if !self.is_tracing {
//                             let count = {
//                                 let count = self.trace_counter.entry(in_pc).or_insert(0);
//                                 *count += 1;
//                                 *count
//                             };
//                             // println!("COUNT {:?} {}", in_pc, count);
//                             if count > HOT_LOOP {
//                                 self.active_trace.clear();
//                                 self.trace_counter.clear();
//                                 self.is_tracing = true;
//                                 self.loop_start = in_pc;
//                             }
//                         } else {
//                             if in_pc == self.loop_start {
//                                 self.finish_trace(in_pc);
//                             }
//                         }
//                     } else {
//                         panic!("expected usize got {:?}", val);
//                     }
//                 },

//                 OpCode::Call => {
//                     // self.w_stack_pointer += func_stacksize.unwrap();

//                     // mark frame on stack
//                     // self.stack.push(StackData::Frame(self.w_stack_pointer));
//                     self.call_stack.push((func.clone(), pc));
//                     self.w_stack_pointer_stack.push(self.w_stack_pointer);
//                     self.w_stack_pointer += func_stacksize.unwrap();

//                     let wrapped_address = self.pop_stack_value();
//                     if let WrappedValue::Address(address) = wrapped_address {
//                         if let Address::StaticFunc(def_id) = address {

//                             func = self.program.get_func(def_id);
//                             pc = 0;
//                             //avoid incrementing pc counter
//                             continue;
//                             // self.eval_func(func);
//                         } else {
//                             panic!("Expected func got {:?}", address);
//                         }
//                     } else {
//                         panic!("excpected address got {:?}", wrapped_address);
//                     }
//                 },

//                 OpCode::JUMP_REL(n) => {
//                     pc = (pc as i32 + n) as usize;
//                     continue
//                 },

//                 OpCode::JUMP_REL_IF(n) => {
//                     // let data = self.stack.pop().unwrap();
//                     let data = self.pop_stack_value();
//                     if let WrappedValue::Bool(b) = data {
//                         if b {
//                             pc = (pc as i32 + n) as usize;
//                             continue;
//                         }
//                     } else {
//                         panic!("expected bool got {:?}", data);
//                     }
//                 },

//                 OpCode::TUPLE(n) => self.o_tuple(n),
//                 OpCode::TUPLE_ASSIGN(idx) => self.o_tuple_assign(idx),
//                 OpCode::TUPLE_GET(idx) => self.o_tuple_get(idx),
//                 OpCode::TUPLE_SET(idx) => self.o_tuple_set(idx),

//                 OpCode::VEC(n) => self.o_vec(n),
//                 OpCode::Repeat(n) => self.o_repeat(n),

//                 OpCode::AssignIndex => self.o_assign_index(),
//                 OpCode::GetIndex => self.o_get_index(),

//                 OpCode::Len => self.o_len(),

//                 OpCode::SignedInteger(n) => {
//                     self.stack.push(StackData::Value(WrappedValue::I64(n)));
//                 },
//                 OpCode::UnsignedInteger(n) => {
//                     self.stack.push(StackData::Value(WrappedValue::U64(n)));
//                 },
//                 OpCode::Usize(size) => {
//                     self.stack.push(StackData::Value(WrappedValue::Usize(size)));
//                 },
//                 OpCode::Bool(b) => {
//                     self.stack.push(StackData::Value(WrappedValue::Bool(b)));
//                 },

//                 OpCode::StoreLocal(idx) => {
//                     self.o_store_local(idx);
//                     self.stack.pop();
//                 },

//                 OpCode::QuickStoreLocal(idx) => self.o_store_local(idx),
//                 OpCode::LoadLocal(idx) => self.o_load_local(idx),
//                 OpCode::BINOP(op) => self.o_binop(op),

//                 OpCode::BORROW(..) => {
//                     let address = self.stack.pop().unwrap().unwrap_address();
//                     self.stack.push(StackData::Value(
//                         WrappedValue::Address(address)))
//                 },

//                 OpCode::DEREF => {
//                     let wrapped_target = self.pop_stack_value();
//                     if let WrappedValue::Address(target) = wrapped_target {
//                         match target {
//                             Address::StackLocal(_idx) => {
//                                 self.stack.push(StackData::Pointer(target));
//                             },
//                             _ => unimplemented!()
//                         }
//                     }  else {
//                         panic!("can't resolve {:?}", wrapped_target);
//                     }
//                 },

//                 OpCode::DEREF_STORE => {
//                     let wrapped_target = self.pop_stack_value();
//                     let value = self.pop_stack_value();

//                     if let WrappedValue::Address(target) = wrapped_target {
//                         match target {
//                             Address::StackLocal(idx) => {
//                                 self.w_stack[idx] = value;
//                             }
//                             _ => unimplemented!()
//                         }
//                     } else {
//                         panic!("can't resolve {:?}", wrapped_target);
//                     }
//                 },

//                 OpCode::Use => {
//                     //XXX DO SOMETHING
//                 },

//                 OpCode::Pop => { self.pop_stack_value(); },


//                 _ => {
//                     println!("TODO {:?}", opcode);
//                     // unimplemented!();
//                 },
//             }
//             pc += 1;
//         }

//         // println!("\nLocals: {:?}", self.w_stack);
//     }

//     fn o_vec(&mut self, size: usize) {
//         // let mut array: Vec<WrappedValue> = Vec::with_capacity(size);
//         // for _ in 0..size {
//             // array.push(WrappedValue::None)
//         // }

//         let mut array = vec![WrappedValue::None; size];

//         for idx in (0..size).rev() {
//             let val = self.pop_stack_value();
//             array[idx] = val;
//         }
//         self.stack.push(StackData::Value(WrappedValue::Array(array)));
//     }

//     fn o_repeat(&mut self, size: usize) {
//         let mut array: Vec<WrappedValue> = Vec::with_capacity(size);
//         for _ in 0..size {
//             array.push(WrappedValue::None)
//         }

//         let val = self.pop_stack_value();
//         for idx in 0..size {
//             array[idx] = val.clone();
//         }

//         self.stack.push(StackData::Value(WrappedValue::Array(array)));
//     }

//     fn o_len(&mut self) {
//         let v = self.pop_stack_value();

//         if let WrappedValue::Array(ref array) = v {
//             self.stack.push(StackData::Value(WrappedValue::Usize(array.len())));
//         } else {
//             panic!("expected array got {:?}", v);
//         }
//     }

//     fn o_get_index(&mut self) {
//         let array_address = self.stack.pop().unwrap().unwrap_address();
//         let index = self.pop_stack_value().unwrap_usize();

//         let object = match array_address {
//             Address::StackLocal(addr) => {
//                 &self.w_stack[addr]
//             },
//             Address::StackComplex(a, b) => {
//                 &self.w_stack[a].unwrap_tuple().data[b]
//             },
//             _ => unimplemented!(),
//         };

//         if let WrappedValue::Array(ref array) = *object {
//             //XXX clone
//             let val = array[index].clone();
//             self.stack.push(StackData::Value(val));
//         }
//     }

//     fn o_assign_index(&mut self) {
//         let array_address = self.stack.pop().unwrap().unwrap_address();
//         let index = self.pop_stack_value().unwrap_usize();
//         let value = self.pop_stack_value();

//         let mut obj = match array_address {
//             Address::StackLocal(addr) => {
//                 &mut self.w_stack[addr]
//             },
//             Address::StackComplex(a, b) => {
//                 &mut self.w_stack[a].unwrap_tuple().data[b]
//             }
//             _ => unimplemented!(),
//         };

//         if let WrappedValue::Array(ref mut array) = *obj {
//             array[index] = value;
//         } else {
//             unimplemented!()
//         }

//     }

//     fn o_tuple(&mut self, size: usize) {
//         self.stack.push(StackData::Value(WrappedValue::Tuple(WrappedTuple::with_size(size))));
//     }

//     fn o_tuple_set(&mut self, idx: usize) {
//         let tuple_address = self.stack.pop().unwrap().unwrap_address();
//         let value = self.pop_stack_value();

//         match tuple_address {
//             Address::StackLocal(addr) => {
//                 if let WrappedValue::Tuple(ref mut tuple) = self.w_stack[addr] {
//                     tuple.data[idx] = value;
//                 }
//             },
//             _ => panic!("can not load tuple at {:?}", tuple_address),
//         }
//     }

//     fn o_tuple_assign(&mut self, idx: usize) {
//         let value = self.pop_stack_value();
//         let mut s_tuple = self.stack.last_mut().unwrap();

//         if let StackData::Value(WrappedValue::Tuple(ref mut tuple)) = *s_tuple  {
//             tuple.data[idx] = value;
//         } else {
//             panic!("Expected tuple found {:?}", s_tuple);
//         }
//     }

//     fn o_tuple_get(&mut self, idx: usize) {
//         // let s_tuple = self.pop_stack_value();
//         match self.stack.pop().unwrap().unwrap_address() {
//             Address::StackLocal(tuple_address) => {
//                 self.stack.push(StackData::Pointer(
//                     Address::StackComplex(tuple_address, idx)));
//             },
//             _ => unimplemented!(),
//         }
//     }

//     fn o_store_local(&mut self, idx: usize) {
//         // let v = self.stack.pop().unwrap();
//         let v = self.stack.last().unwrap().clone();
//         //XXXX
//         let val = match v {
//             StackData::Value(v) => v,
//             StackData::Pointer(Address::StackLocal(other)) => {
//                 self.w_stack[other].clone()
//             },
//             StackData::Pointer(Address::StaticFunc(defid)) => {
//                 WrappedValue::Address(Address::StaticFunc(defid))
//             },
//             StackData::Pointer(Address::StackComplex(a, b)) => {
//                 let tuple = self.w_stack[a].unwrap_tuple();
//                 tuple.data[b].clone()
//             },

//             _ => panic!("should not store interpreter level object {:?}", v)
//         };

//         self.w_stack[self.w_stack_pointer + idx] = val;
//     }

//     fn o_load_local(&mut self, idx: usize) {
//         self.stack.push(StackData::Pointer(Address::StackLocal(self.w_stack_pointer + idx)))
//     }

//     fn o_binop(&mut self, op: BinOp) {
//         use self::WrappedValue::*;
//         use rustc::mir::repr::BinOp::*;

//         let right = self.pop_stack_value();
//         let left = self.pop_stack_value();

//         // copied from miri
//         macro_rules! int_binops {
//             ($v:ident, $l:ident, $r:ident) => ({
//                 match op {
//                     Add    => $v($l + $r),
//                     Sub    => $v($l - $r),
//                     Mul    => $v($l * $r),
//                     Div    => $v($l / $r),
//                     Rem    => $v($l % $r),
//                     BitXor => $v($l ^ $r),
//                     BitAnd => $v($l & $r),
//                     BitOr  => $v($l | $r),

//                     // TODO(solson): Can have differently-typed RHS.
//                     Shl => $v($l << $r),
//                     Shr => $v($l >> $r),

//                     Eq => Bool($l == $r),
//                     Ne => Bool($l != $r),
//                     Lt => Bool($l < $r),
//                     Le => Bool($l <= $r),
//                     Gt => Bool($l > $r),
//                     Ge => Bool($l >= $r),
//                 }
//             })
//         }


//         let val = StackData::Value(match(left, right) {
//             (I64(l), I64(r)) => int_binops!(I64, l, r),
//             (U64(l), U64(r)) => int_binops!(U64, l, r),
//             (Usize(l), Usize(r)) => int_binops!(Usize, l, r),

//             // copied from miri
//             (Bool(l), Bool(r)) => {
//                 Bool(match op {
//                     Eq => l == r,
//                     Ne => l != r,
//                     Lt => l < r,
//                     Le => l <= r,
//                     Gt => l > r,
//                     Ge => l >= r,
//                     BitOr => l | r,
//                     BitXor => l ^ r,
//                     BitAnd => l & r,
//                     Add | Sub | Mul | Div | Rem | Shl | Shr =>
//                         panic!("invalid binary operation on booleans: {:?}", op),
//                 })

//             },

//             (l, r) => {
//                 println!("{:?} {:?}", l, r);
//                 unimplemented!();
//             }
//         });
//         self.stack.push(val);

//     }
// }

use std::rc::Rc;
use std::cell::RefCell;

use rustc::mir::mir_map::MirMap;
use rustc::mir::repr::BinOp;
use rustc::ty::TyCtxt;
use rustc::hir::def_id::DefId;

use bc::translate::Program;
use bc::bytecode::OpCode;
use core::objects::{R_BoxedValue, CallFrame,
    R_Pointer, R_Function, R_Struct,
    InstructionPointer};


#[derive(Debug, Clone)]
pub enum StackValue {
    Owned(R_BoxedValue),
    Ref(Rc<RefCell<R_BoxedValue>>),
}

impl StackValue {
    // as_value should be clone
    fn as_value(self) -> Self {
        match self {
            StackValue::Owned(..) => self,
            StackValue::Ref(cell) => {
                StackValue::Owned(cell.borrow().clone())
            }
        }
    }

    // self has to be owned
    fn unwrap_value(self) -> R_BoxedValue {
        if let StackValue::Owned(val) = self {
            val
        } else {
            panic!("expected owned val");
        }
    }

    fn to_ref(self) -> Rc<RefCell<R_BoxedValue>> {
        match self {
            StackValue::Ref(cell) => cell,
            StackValue::Owned(boxed) => Rc::new(RefCell::new(boxed))
        }
    }

    /// Address as Value
    /// needs to be owned. return contents
    fn as_ref(self) -> Self {
        let cell = self.to_ref();
        StackValue::Owned(R_BoxedValue::Ptr(R_Pointer{cell:cell}))
    }

    /// Deref pointer
    fn deref(self) -> Self {
        // self contains an owned R_Pointer
        if let R_BoxedValue::Ptr(ptr) = self.as_value().unwrap_value() {
            StackValue::Ref(ptr.cell)
        } else {
            panic!("expected val to be pointer");
        }
    }
}

pub struct Interpreter<'a, 'cx: 'a> {
    pub program: &'a mut Program<'a, 'cx>,
    pub stack: Vec<StackValue>,
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
                    self.stack.push(StackValue::Owned(val));
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
                    // if let Some(ret) = self.o_return() {
                    //     func = ret.func;
                    //     pc = ret.pc;
                    // } else {
                    //     return;
                    // }
                    return;
                },

                OpCode::Skip(n) => { pc += n; continue },

                OpCode::BinOp(kind) => self.o_binop(kind),
                OpCode::CheckedBinOp(kind) => self.o_checked_binop(kind),

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
        self.stack.push(StackValue::Ref(cell_ptr))
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

    // fn o_return(&mut self) -> Option<InstructionPointer> {
        // let old_frame = self.stack_frames.pop().unwrap();
        // old_frame.return_addr
    // }

    fn o_tuple(&mut self, size: usize) {
        // let mut tuple = R_Struct::tuple(size);
        // for idx in (0..size).rev() {
            // tuple.data[idx] = self.stack.pop().unwrap().as_value(self).unwrap_value();
        // }
        // self.stack.push(StackValue::Val(R_BoxedValue::Struct(tuple)));
    }

    fn o_tuple_set(&mut self, idx: usize) {
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
        //     Address::StackLocal(tuple_address) => {
        //         self.stack.push(StackData::Pointer(
        //             Address::StackComplex(tuple_address, idx)));
        //     },
        //     _ => unimplemented!(),
        // }
    }

    // fn pop_value(&mut self) -> R_BoxedValue {
        // self.stack.pop().unwrap().as_value(self).unwrap_value()
    // }

    fn o_binop(&mut self, kind: BinOp) {
        // let val = self._do_binop(kind);
        // self.stack.push(val);
    }

    fn o_checked_binop(&mut self, kind: BinOp) {
        //TODO: actually check binops
        // let val = self._do_binop(kind);
        // let mut tuple = R_Struct::tuple(2);
        // tuple.data[0] = R_BoxedValue::Bool(true);
        // tuple.data[1] = val.unwrap_value();
        // self.stack.push(StackValue::Val(R_BoxedValue::Struct(tuple)));
    }

    // fn _do_binop(&mut self, kind: BinOp) -> StackValue {

    //     use core::objects::R_BoxedValue::*;
    //     use rustc::mir::repr::BinOp::*;

    //     let right = self.pop_value();
    //     let left = self.pop_value();

    //     // copied from miri
    //     macro_rules! int_binops {
    //         ($v:ident, $l:ident, $r:ident) => ({
    //             match kind {
    //                 Add    => $v($l + $r),
    //                 Sub    => $v($l - $r),
    //                 Mul    => $v($l * $r),
    //                 Div    => $v($l / $r),
    //                 Rem    => $v($l % $r),
    //                 BitXor => $v($l ^ $r),
    //                 BitAnd => $v($l & $r),
    //                 BitOr  => $v($l | $r),

    //                 // TODO(solson): Can have differently-typed RHS.
    //                 Shl => $v($l << $r),
    //                 Shr => $v($l >> $r),

    //                 Eq => Bool($l == $r),
    //                 Ne => Bool($l != $r),
    //                 Lt => Bool($l < $r),
    //                 Le => Bool($l <= $r),
    //                 Gt => Bool($l > $r),
    //                 Ge => Bool($l >= $r),
    //             }
    //         })
    //     }


    //     let val = StackValue::Val(match(left, right) {
    //         (I64(l), I64(r)) => int_binops!(I64, l, r),
    //         (U64(l), U64(r)) => int_binops!(U64, l, r),
    //         (Usize(l), Usize(r)) => int_binops!(Usize, l, r),

    //         // copied from miri
    //         (Bool(l), Bool(r)) => {
    //             Bool(match kind {
    //                 Eq => l == r,
    //                 Ne => l != r,
    //                 Lt => l < r,
    //                 Le => l <= r,
    //                 Gt => l > r,
    //                 Ge => l >= r,
    //                 BitOr => l | r,
    //                 BitXor => l ^ r,
    //                 BitAnd => l & r,
    //                 Add | Sub | Mul | Div | Rem | Shl | Shr =>
    //                     panic!("invalid binary operation on booleans: {:?}", kind),
    //             })

    //         },

    //         (l, r) => {
    //             println!("{:?} {:?}", l, r);
    //             unimplemented!();
    //         }
    //     });

    //     val
    // }
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