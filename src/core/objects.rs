

use std::rc::Rc;

use bc::bytecode::Function;

// Since we don't have a flat memory model, pointers can point to different
// locations. E.g. a pointer to the stack is different from one to the heap.
#[derive(Debug, Clone)]
pub enum R_Pointer {

    // XXX: Can we used that? Or do we have to use Stack?
    // Whenever we leave the current stack (call another function)
    // we need a reference to the stack.
    // Data in the same frame
    // Local(usize),

    // Simple values on stack
    Stack(R_StackPointer),

    // A pointer to a field within a struct on the stack.
    // let x = &foo.bar;
    // XXX: This can be recursive, what happens with &.a.b.c ?
    StackField(R_StackPointer, usize),

    // XXX: do we just have const pointers or can we have special const-func pointers?
    ConstFunc(Rc<Function>),
}

// a pointer to a value on the stack
#[derive(Debug, Clone)]
pub struct R_StackPointer {
    pub frame: usize,
    pub idx: usize,
}


// Boxed rust values.
// Only these values can life on the stack. XXX: is this true?
#[derive(Debug, Clone)]
pub enum R_BoxedValue {
    Null,
    Ptr(R_Pointer),
    i64(i64),
    u64(u64),
    usize(usize),
    bool(bool),
    Struct(R_Struct),
    // Array(R_Array),
}

// For not we allocate the meta information on the stack with a pointer to the
// host-level heap.

#[derive(Debug, Clone)]
pub struct R_Struct {
    pub alive: bool,
    pub behaviour: MoveSemantics,
    pub data: Vec<R_BoxedValue>
}

#[derive(Debug, Clone)]
pub enum MoveSemantics {
    // let a = X(1);
    // let b = a

    // Copy. `b` is a copy of `a`, with `a.alive == true` and `b.alive == true`
    // impl copy -> !impl drop
    // both a and b have to be considered when going out of scope
    // However a is bitwise identical to b. Thus this limits which types can
    // actually implement Copy. For example it is not possible to have an Rc
    // field, since the copy does not increase the ref counter. Also &mut T
    // is not allowed, as this would result in two mutable references to the
    // same resource.
    // No destruction needed.
    Copy,

    // Default. `b` is still a bitwise copy of `a`. However `a.alive == false`.
    // In other words, ownership has been transferred. The stack slot of a is
    // still available but is set to unitialised. Thus this is possible:
    // let mut a = X(1);
    // let b = a;
    // a = X(2);
    // Transfer of ownership does not mean that b takes control of a's memory space.
    // Instead it takes control of a's data (for example it inherits mutable
    // references or Rc's).
    // Consequently if possible a type should implement copy if it can.
    // If a goes out of scope it can be ignored since nothing needs to be dropped.
    // Destruction may be needed.
    Move,

    // Drop. Same as Move, but it has a Drop handler attached. This normally means
    // that the type manages other special resources. rustc inserts explicit drops
    // whenever a type is moved.
    Drop,

    // FIXME: Do we have to consider the difference between Move and Drop here?
    // Have a look at type_needs_drop_given_env
}

impl R_Struct {
    fn with_size(size: usize) -> Self {
        R_Struct { data: vec![R_BoxedValue::Null; size], alive: true, behaviour: MoveSemantics::Copy }
    }
}
