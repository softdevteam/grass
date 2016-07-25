#![allow(non_camel_case_types)]


use std::rc::Rc;
use std::cell::RefCell;

use rustc::hir::def_id::DefId;

use bc::bytecode::OpCode;
use interp::Interpreter;



#[derive(Debug, Clone, RustcEncodable, RustcDecodable, PartialEq)]
pub struct R_Function {
    pub args_cnt: usize,
    pub locals_cnt: usize,
    pub opcodes: Vec<OpCode>,
}



// Since we don't have a flat memory model, pointers can point to different
// locations. E.g. a pointer to the stack is different from one to the heap.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, RustcEncodable, RustcDecodable, PartialEq)]
pub struct R_Pointer {
    pub cell: Rc<RefCell<R_BoxedValue>>,
}

// pub enum R_Pointer {

//     // XXX: Can we used that? Or do we have to use Stack?
//     // Whenever we leave the current stack (call another function)
//     // we need a reference to the stack.
//     // Data in the same frame
//     // Local(usize),

//     /// Simple values on stack
//     Stack(R_StackPointer),

//     /// A pointer to a field within a struct on the stack.
//     // let x = &foo.bar;
//     // XXX: This can be recursive, what happens with &.a.b.c ?
//     Nested(R_NestedPointer),

//     // XXX: do we just have const pointers or can we have special const-func pointers?
//     ConstFunc(Rc<R_Function>),
// }

// impl R_Pointer {
//     pub fn deref<'a>(&self, env: &'a Interpreter) -> &'a R_BoxedValue {
//         match *self {
//             R_Pointer::Stack(ptr) => {
//                 &env.stack_frames[ptr.frame].locals[ptr.idx]
//             },

//             _ => unimplemented!(),
//         }
//     }
// }


// Boxed rust values.
// Only these values can life on the stack. XXX: is this true?

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, RustcEncodable, RustcDecodable, PartialEq)]
pub enum R_BoxedValue {
    Null,
    Ptr(R_Pointer),
    I64(i64),
    U64(u64),
    F64(f64),
    Usize(usize),
    Bool(bool),
    Struct(R_Struct),
    Func(DefId),
    // Array(R_Array),
}

// For not we allocate the meta information on the stack with a pointer to the
// host-level heap.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, RustcEncodable, RustcDecodable, PartialEq)]
pub struct R_Struct {
    pub alive: bool,
    pub behaviour: MoveSemantics,
    pub data: Vec<Rc<RefCell<R_BoxedValue>>>
}

impl R_Struct {
    pub fn tuple(size: usize) -> Self {
        R_Struct { alive: true, behaviour: MoveSemantics::Move,
                   data: vec![Rc::from(RefCell::new(R_BoxedValue::Null)); size] }
    }

    pub fn with_size(size: usize) -> Self {
        R_Struct { alive: true, behaviour: MoveSemantics::Copy,
                   data: vec![Rc::from(RefCell::new(R_BoxedValue::Null)); size] }
    }
}

/// MoveSemantics
///
/// **FIXME:** Do we have to consider the difference between Move and Drop here?
// Have a look at type_needs_drop_given_env

/// ## Example
/// In the following the variants are exlained using this code example:
///
///     let a = X(1);
///     let b = a

#[derive(Debug, Clone, RustcEncodable, RustcDecodable, PartialEq)]
pub enum MoveSemantics {
    /// **Copy semantics**
    ///
    /// `b` is a copy of `a`, with `a.alive == true` and `b.alive == true`
    ///
    ///     impl copy -> !impl drop
    ///
    /// Both a and b have to be considered when going out of scope
    /// However a is bitwise identical to b. Thus this limits which types can
    /// actually implement Copy. For example it is not possible to have an Rc
    /// field, since the copy does not increase the ref counter. Also &mut T
    /// is not allowed, as this would result in two mutable references to the
    /// same resource.
    /// No destruction needed.
    Copy,

    /// **Default.**
    ///
    /// `b` is still a bitwise copy of `a`. However `a.alive == false`.
    ///
    /// In other words, *ownership has been transferred*. The stack slot of a is
    /// still available but is set to unitialised. Thus this is possible:
    ///
    ///     let mut a = X(1);
    ///     let b = a;
    ///     a = X(2);
    ///
    /// Transfer of ownership does not mean that b takes control of a's memory space.
    /// Instead it takes control of a's data (for example it inherits mutable
    /// references or Rc's).
    ///
    /// Consequently if possible `a` type should implement copy if it can.
    /// If a goes out of scope it can be ignored since nothing needs to be dropped.
    /// Destruction may be needed (?).
    Move,

    /// Same as Move, but it has a Drop handler attached. This normally means
    /// that the type manages other special resources. rustc inserts explicit drops
    /// whenever a type is moved.
    Drop,
}



#[derive(Debug, Clone)]
pub struct InstructionPointer {
    pub func: Rc<R_Function>,
    pub pc: usize,
}

// A stack frame.
#[derive(Debug, Clone)]
pub struct CallFrame {
    pub return_addr: Option<InstructionPointer>,
    pub locals: Vec<Rc<RefCell<R_BoxedValue>>>,
}

impl CallFrame {
    pub fn new(return_addr: Option<InstructionPointer>, locals_len: usize) -> Self {
        CallFrame {
            return_addr: return_addr,
            locals: vec![Rc::from(RefCell::new(R_BoxedValue::Null)); locals_len]
        }
    }
}

