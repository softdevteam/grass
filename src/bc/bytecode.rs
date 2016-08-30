
use std::fmt;
use std::rc::Rc;

pub use rustc::mir::repr::{self, BorrowKind, BasicBlock};
// use rustc::hir::def_id::DefId;

use core::objects::{R_BoxedValue, InstructionPointer, R_Function};

use std::marker::Sync;

#[derive(Debug, Clone, PartialEq)]
pub enum OpCode{
    Noop,
    Panic,
    Pop,

    Load(usize),
    Store(usize),

    Use,

    StoreStatic(usize),

    // Const(Constant<'tcx>),
    Static(usize),
    // Functions should be loaded normally
    // let x = main;
    // x()

    RunTrace(usize),

    Call,

    // a call in traced execution
    // save the return address
    FlatCall(usize, InstructionPointer, Rc<R_Function>),
    Return,

    Resume, //resume stack unwinding


    ConstValue(R_BoxedValue),
    // UnsignedInteger(u64),
    // Usize(usize),
    // SignedInteger(i64),
    // Float(f64),
    // Bool(bool),


    // Ref(BorrowKind),
    Ref,

    Deref,
    DerefStore,

    Unsize,

    CheckedBinOp(BinOp),
    BinOp(BinOp),

    Not,
    Neg,

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

unsafe impl Sync for OpCode {

}

impl OpCode {
    pub fn todo_s(s: &str) -> Self {
        OpCode::Todo(String::from(s))
    }

    pub fn to_rs(&self) -> String {
        format!("OpCode::{:?}", self)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum InternalFunc {
    MergePoint,
    Out,
    Print,
    Assert,
}

#[derive(Clone, PartialEq)]
pub struct Guard {
    pub expected: bool,
    // pub recovery: Rc<Function>,
    // pub pc: usize,
    pub recovery: InstructionPointer,
}

impl fmt::Debug for Guard {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Guard({})", self.expected)
    }
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum BinOp {
    /// The `+` operator (addition)
    Add,
    /// The `-` operator (subtraction)
    Sub,
    /// The `*` operator (multiplication)
    Mul,
    /// The `/` operator (division)
    Div,
    /// The `%` operator (modulus)
    Rem,
    /// The `^` operator (bitwise xor)
    BitXor,
    /// The `&` operator (bitwise and)
    BitAnd,
    /// The `|` operator (bitwise or)
    BitOr,
    /// The `<<` operator (shift left)
    Shl,
    /// The `>>` operator (shift right)
    Shr,
    /// The `==` operator (equality)
    Eq,
    /// The `<` operator (less than)
    Lt,
    /// The `<=` operator (less than or equal to)
    Le,
    /// The `!=` operator (not equal to)
    Ne,
    /// The `>=` operator (greater than or equal to)
    Ge,
    /// The `>` operator (greater than)
    Gt,
}

impl BinOp {
    pub fn new(kind: repr::BinOp) -> Self {
        match kind {
            repr::BinOp::Add => BinOp::Add,
            repr::BinOp::Sub => BinOp::Sub,
            repr::BinOp::Mul => BinOp::Mul,
            repr::BinOp::Div => BinOp::Div,
            repr::BinOp::Rem => BinOp::Rem,
            repr::BinOp::BitXor => BinOp::BitXor,
            repr::BinOp::BitAnd => BinOp::BitAnd,
            repr::BinOp::BitOr => BinOp::BitOr,
            repr::BinOp::Shl => BinOp::Shl,
            repr::BinOp::Shr => BinOp::Shr,
            repr::BinOp::Eq => BinOp::Eq,
            repr::BinOp::Lt => BinOp::Lt,
            repr::BinOp::Le => BinOp::Le,
            repr::BinOp::Ne => BinOp::Ne,
            repr::BinOp::Ge => BinOp::Ge,
            repr::BinOp::Gt => BinOp::Gt,
        }
    }
}

