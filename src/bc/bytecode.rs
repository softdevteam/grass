
use std::fmt;


pub use rustc::mir::repr::{BinOp, BorrowKind, BasicBlock};
use rustc::hir::def_id::DefId;

use core::objects::{R_BoxedValue};



#[derive(Debug, Clone, RustcEncodable, RustcDecodable, PartialEq)]
pub enum OpCode{
    Noop,
    Pop,

    Load(usize),
    Store(usize),

    Use,

    StoreStatic(DefId),

    // Const(Constant<'tcx>),
    Static(DefId),
    // Functions should be loaded normally
    // let x = main;
    // x()
    // TODO: remove
    LoadFunc(DefId),

    Call,
    Return,

    Resume, //resume stack unwinding


    ConstValue(R_BoxedValue),
    // UnsignedInteger(u64),
    // Usize(usize),
    // SignedInteger(i64),
    // Float(f64),
    // Bool(bool),


    Ref(BorrowKind),

    Deref,
    DerefStore,

    CheckedBinOp(BinOp),
    BinOp(BinOp),

    Array(usize), // let x = [1, 2, 3, 4];
    Repeat(usize), // let x = [0u32; 10];
    Len,
    AssignIndex,
    GetIndex,


    // let x = (a, b);
    // let x = Foo{a: a, b: b};
    Tuple(usize),

    // x.?
    TupleGet(usize),

    // x.? = 42
    TupleSet(usize),
    TupleInit(usize),

    Skip(usize),
    JumpBack(usize),

    SkipIf(usize),
    JumpBackIf(usize),

    InternalFunc(InternalFunc),

    Guard(Guard),

    Todo(String),
}

impl OpCode {
    pub fn todo_s(s: &str) -> Self {
        OpCode::Todo(String::from(s))
    }
}

#[derive(Clone, Debug, RustcEncodable, RustcDecodable, PartialEq)]
pub enum InternalFunc {
    MergePoint,
    Out,
    Print,
}

#[derive(Clone, RustcEncodable, RustcDecodable, PartialEq)]
pub struct Guard {
    pub expected: bool,
    // pub recovery: Rc<Function>,
    pub pc: usize,
}

impl fmt::Debug for Guard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Guard({})", self.expected)
    }
}

