



struct R_Crate {
    // types: Vec<TypeDefinition>,
    statics: Vec<R_Value>,
    constants: Vec<R_Value>,
    functions: Vec<R_Function>,
}

pub struct R_Function {
    args_cnt: usize,
    locals_cnt: usize,
    opcodes: Vec<OpCode>,
}

pub struct R_Object<'a> {
    ty: &'a R_Type,
    fields: Vec<R_Value>,
}