

/**
 * Uses rust's MIR to generate opcodes.
 * HIR: tree, MIR: flow-graph, LIR: linear
 */

use std::collections::BTreeMap;
use std::rc::Rc;


use bc::bytecode::{OpCode, Function, InternalFunc};
use core::objects::{R_BoxedValue};


use rustc::mir::repr::{
    BasicBlock, BasicBlockData, Mir,
    Constant, Literal, Operand,
    Lvalue, Rvalue,
    Statement, StatementKind, Terminator, TerminatorKind,
    ProjectionElem, AggregateKind,
    Field, CastKind
};

use rustc::mir::mir_map::MirMap;
use rustc::middle::const_val::ConstVal;

use rustc::hir::map::Node;
use rustc::hir::def_id::DefId;

use rustc::ty::{TyCtxt, AdtKind, VariantKind};


// use syntax_pos::DUMMY_SP;
use rustc_data_structures::indexed_vec::Idx;


pub type KrateTree<'a> = BTreeMap<DefId, Rc<Function>>;

pub struct Program<'a, 'tcx: 'a> {
    context: &'a Context<'a, 'tcx>,
    pub krates: KrateTree<'a>
}

impl<'a, 'tcx> Program<'a, 'tcx> {
    fn new(context: &'a Context<'a, 'tcx>) -> Program<'a, 'tcx> {
        Program {context: context, krates: BTreeMap::new() }
    }

    fn get_func<'b>(&'b mut self, def_id: DefId) -> Rc<Function> {
        let context = &self.context;

        self.krates.entry(def_id).or_insert_with(|| {
            // println!("load function {:?}", def_id);
            let cs = &context.tcx.sess.cstore;
            let mir = cs.maybe_get_item_mir(context.tcx, def_id).unwrap_or_else(||{
                panic!("no mir for {:?}", def_id);
            });
            Rc::new(context.mir_to_bytecode(&mir))
        }).clone()
    }
}

/// enum to avoid polluting OpCodes with later unused variants
enum MetaOpCode {
    Goto(BasicBlock),
    GotoIf(BasicBlock),
    OpCode(OpCode),
}

pub trait ByteCode {
    fn to_opcodes(&self, &mut Analyser) {}
    fn as_rvalue(&self, &mut Analyser) {}
}


pub struct Context<'a, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub map: &'a MirMap<'tcx>
}


impl<'a, 'tcx> Context<'a, 'tcx> {

    pub fn mir_to_bytecode(&'a self, func: &Mir<'a>) -> Function {
        let blocks = func.basic_blocks().iter().map(
            |bb| {
                let mut gen = Analyser::new(self.tcx, func);
                gen.analyse_block(bb);
                gen.opcodes
            }).collect();

        self.flatten_blocks(&blocks, func)
    }


    // replace Gotos with Jumps
    fn flatten_blocks(&'a self, blocks: &Vec<Vec<MetaOpCode>>, func: &Mir<'a>) -> Function {
        let mut indicies = Vec::new();
        let mut n = 0usize;
        for block in blocks {
            indicies.push(n);
            n += block.len();
        }


        let mut opcodes = Vec::new();

        let new_target = |block: &BasicBlock| indicies[block.index()];

        for block in blocks {
            for (current, opcode) in block.iter().enumerate() {
                let oc = match *opcode {

                    MetaOpCode::Goto(ref bb) => {
                        if new_target(bb) < current {
                            OpCode::JumpBack(current - new_target(bb))
                        } else {
                            OpCode::Skip(new_target(bb) - current)
                        }
                    },

                    MetaOpCode::GotoIf(ref bb) => {
                        if new_target(bb) < current {
                            OpCode::JumpBackIf(current - new_target(bb))
                        } else {
                            OpCode::SkipIf(new_target(bb) - current)
                        }
                    },

                    MetaOpCode::OpCode(ref oc) => oc.clone(),
                };
                opcodes.push(oc);
            }
        }
        opcodes
    }
}


/// Convert single function MIR to bytecode.
///
/// # Summary
/// There are three different
/// Block:
/// Each block constists of two parts: a list of statements and a terminator.
/// The statements are assignments of rvalues to lvalues.
///
///     [ lvalue := rvalue, ... ], terminator
///
/// Lvalues can be understood as cells. Theses inlcudes local variables,
/// fields in structs and tuples and space on the heap.
///
/// Rvalues are 'single' instructions, meaning a more complex right hand side
/// is split up into several steps using temporary variables.
///
/// For example:
///
///     a = 1 + 2 + 3
///
/// is represented by something like
///
///     tmp_0 = 1 + 2
///     a = tmp_0 = 3
///
/// Terminators ensure control flow within a function and is where all
/// branching happens. Termniators have up to two pointers to consecutive
/// basic blocks. Thus there are no explicit loops within MIR.
///

pub struct Analyser<'a, 'tcx: 'a>{
    opcodes: Vec<MetaOpCode>,
    tcx: TyCtxt<'a, 'tcx, 'tcx>,
    func_mir: &'a Mir<'a>,
}

impl<'a, 'tcx> Analyser<'a, 'tcx> {

    fn new(tcx: TyCtxt<'a, 'tcx, 'tcx>, func_mir: &'a Mir<'a>) -> Self {
        Analyser{ opcodes: Vec::new(), tcx: tcx, func_mir: func_mir }
    }

    fn arg_to_local(&self, n: usize) -> usize {
        n
    }

    fn var_to_local(&self, n: usize) -> usize {
        self.func_mir.arg_decls.len() + n
    }

    fn tmp_to_local(&self, n: usize) -> usize {
        self.var_to_local(n) + self.func_mir.var_decls.len()
    }

    // fn struct_implements_copy(&self, ty: Ty) -> bool {
        // let param_env = &self.tcx.empty_parameter_environment();
        // !ty.moves_by_default(tcx, param_env, DUMMY_SP);
    // }

    fn add(&mut self, opcode: OpCode) {
        self.opcodes.push(MetaOpCode::OpCode(opcode));
    }

    fn analyse_block(&mut self, block: &BasicBlockData<'a>) {
        for statement in &block.statements {
            statement.to_opcodes(self);
        }
        block.terminator().to_opcodes(self);
    }

    fn unpack_const(&mut self, literal: &Literal) {
        let oc = OpCode::ConstValue(match *literal {
            Literal::Value{ ref value } => {

                use rustc_const_math::ConstInt::*;
                use rustc_const_math::ConstFloat::*;
                use rustc_const_math::{Us16, Us32, Us64};

                use rustc::middle::const_val::ConstVal::{
                    Integral, Float, Bool, Function, Array,
                    Str, ByteStr, Tuple, Struct, Repeat, Char, Dummy};

                //TODO: float
                match *value {
                    Integral( U8(u)) => R_BoxedValue::U64(u as u64),
                    Integral(U16(u)) => R_BoxedValue::U64(u as u64),
                    Integral(U32(u)) => R_BoxedValue::U64(u as u64),
                    Integral(U64(u)) => R_BoxedValue::U64(u),

                    Integral( I8(i)) => R_BoxedValue::I64(i as i64),
                    Integral(I16(i)) => R_BoxedValue::I64(i as i64),
                    Integral(I32(i)) => R_BoxedValue::I64(i as i64),
                    Integral(I64(i)) => R_BoxedValue::I64(i),

                    Integral(Usize(Us16(us16))) => R_BoxedValue::Usize(us16 as usize),
                    Integral(Usize(Us32(us32))) => R_BoxedValue::Usize(us32 as usize),
                    Integral(Usize(Us64(us64))) => R_BoxedValue::Usize(us64 as usize),

                    Integral(Isize(_i)) => unimplemented!(),
                    Integral(Infer(_u)) => unimplemented!(),
                    Integral(InferSigned(_i)) => unimplemented!(),

                    Float(F32(f)) => R_BoxedValue::F64(f as f64),
                    Float(F64(f)) => R_BoxedValue::F64(f),

                    // should this ever happen?
                    Float(FInfer{f32: _, f64: _}) => unimplemented!(),

                    Bool(b) => R_BoxedValue::Bool(b),

                    Str(_)
                    | ByteStr(_)
                    | Tuple(_)
                    | Struct(_)
                    | Function(_)
                    | Array(_, _)
                    | Repeat(_, _)
                    | Char(_) => unimplemented!(),

                    Dummy => panic!("Dummy"),


                }
            },

            // let x = &42; will generate a reference to a static variable
            Literal::Item{ def_id, .. } => {
                R_BoxedValue::Func(def_id.clone())
            },

            // TODO: what is this doing?
            Literal::Promoted {index} => {
                R_BoxedValue::Usize(index.index())
            },
        });

        self.add(oc);
    }
}


impl<'a> ByteCode for Statement<'a> {
    fn to_opcodes(&self, env: &mut Analyser) -> () {
        let StatementKind::Assign(ref lvalue, ref rvalue) = self.kind;
        rvalue.to_opcodes(env);
        lvalue.to_opcodes(env);
    }
}


impl<'a> ByteCode for Terminator<'a> {

    fn to_opcodes(&self, env: &mut Analyser) {
        let op = match self.kind {

            // Gotos are resolved to JUMPs in a second step later
            TerminatorKind::Goto{target} => {
                MetaOpCode::Goto(target)
            },

            TerminatorKind::If{ref cond, targets: (ref bb1, ref bb2)} => {
                cond.as_rvalue(env);
                env.opcodes.push(MetaOpCode::GotoIf(*bb1));
                MetaOpCode::Goto(*bb2)
            },

            TerminatorKind::Call{ref func, ref args, ref destination, ..} => {
                for arg in args {
                    arg.as_rvalue(env);
                }

                // FIXME: argcount should be encoded in function object
                // env.add(OpCode::ArgCount(args.len()));
                func.as_rvalue(env);

                // destination: Option<(Lvalue<'tcx>, BasicBlock)>,
                match destination.as_ref() {
                    Some(dest) => {
                        env.add(OpCode::Call);
                        dest.0.to_opcodes(env);
                        MetaOpCode::Goto(dest.1)
                    },
                    None => {
                        MetaOpCode::OpCode(OpCode::TODO("NO RETURN"))
                    }
                }
            },

            ref other => {
                MetaOpCode::OpCode(match *other {
                    TerminatorKind::Return => OpCode::Return,

                    TerminatorKind::Resume => OpCode::Resume,



                    TerminatorKind::Drop{location: ref lvalue, target: _, unwind: _} => {
                        lvalue.as_rvalue(env);
                        OpCode::TODO("Drop")
                    },

                    _ => OpCode::TODO("Terminator"),
                })
            }

        };
        env.opcodes.push(op);
    }
}


impl<'a> ByteCode for Lvalue<'a> {
    /// lvalue = <rvalue>
    fn to_opcodes(&self, env: &mut Analyser) {
        match *self {
            //a = <rvalue>
            Lvalue::Var(n)  => {
                let n = env.var_to_local(n.index());
                env.add(OpCode::Store(n));
            },

            //tmp_x = <rvalue>
            Lvalue::Temp(n) => {
                let n = env.tmp_to_local(n.index());
                env.add(OpCode::Store(n));
            },

            Lvalue::Arg(_n)  => unreachable!(),
            Lvalue::Static(def_id)  => {
                env.add(OpCode::StoreStatic(def_id));
            },

            Lvalue::Projection(ref proj) => {
                match proj.elem {
                    //*a = <rvalue>
                    ProjectionElem::Deref => {
                        // let opcode = self.load_lvalue(&proj.base);
                        proj.base.as_rvalue(env);
                        env.add(OpCode::DerefStore);
                    },

                    //a.field = <rvalue>
                    ProjectionElem::Field(field, _type) => {
                        proj.base.as_rvalue(env);

                        let index = field.index();
                        env.add(OpCode::TupleSet(index));
                    },

                    // a[index] = z;
                    ProjectionElem::Index(ref index) => {
                        //index
                        index.as_rvalue(env);

                        //x
                        proj.base.as_rvalue(env);

                        env.add(OpCode::AssignIndex);
                    },

                    ProjectionElem::Subslice{..}
                    | ProjectionElem::ConstantIndex{..}
                    | ProjectionElem::Downcast(..) =>
                        panic!("assign projection {:?}", proj.elem),
                }

            },
            // ignore, value is just left on the stack
            Lvalue::ReturnPointer => {},
        };
    }

    // load lvalue as rvalue
    fn as_rvalue(&self, env: &mut Analyser) {
        let opcode = match *self {
            Lvalue::Var(n) => {
                let n = env.var_to_local(n.index());
                OpCode::Load(n)
            },
            Lvalue::Temp(n) => {
                let n = env.tmp_to_local(n.index());
                OpCode::Load(n)
            },
            Lvalue::Arg(n) => {
                let n = env.arg_to_local(n.index());
                OpCode::Load(n)
            },

            Lvalue::Static(def_id) => OpCode::Static(def_id),
            Lvalue::Projection(ref proj) => {
                match proj.elem {
                    //*lvalue
                    ProjectionElem::Deref => {
                        proj.base.as_rvalue(env);
                        OpCode::Deref
                    },

                    //lvalue.field
                    ProjectionElem::Field(field, _ty) => {
                        //XXX: is type arg needed here?
                        proj.base.as_rvalue(env);
                        OpCode::TupleGet(field.index())
                    },

                    //lvalue[index]
                    ProjectionElem::Index(ref index) => {
                        index.to_opcodes(env);
                        //x
                        proj.base.as_rvalue(env);

                        OpCode::GetIndex
                    },
                    ProjectionElem::Subslice{ .. }
                    | ProjectionElem::ConstantIndex{..}
                    | ProjectionElem::Downcast(..) => panic!("load projection {:?}", proj.elem),
                }
            },
            Lvalue::ReturnPointer => unreachable!(),
        };
        env.add(opcode);
    }
}


impl<'a> ByteCode for Rvalue<'a> {

    fn to_opcodes(&self, env: &mut Analyser) {
        match *self {
            Rvalue::Use(ref op) => {
                op.as_rvalue(env);
                // in other words copy the value
                env.add(OpCode::Use);
            },

            Rvalue::BinaryOp(binop, ref left, ref right) => {
                right.as_rvalue(env);
                left.as_rvalue(env);
                env.add(OpCode::BinOp(binop));
            },

            Rvalue::Aggregate(AggregateKind::Tuple, ref vec) => {
                env.add(OpCode::Tuple(vec.len()));
                for (i, value) in vec.iter().enumerate() {
                    value.as_rvalue(env);
                    env.add(OpCode::TupleInit(i));
                }
            },

            Rvalue::Aggregate(AggregateKind::Vec, ref vec) => {
                for value in vec {
                    value.as_rvalue(env);
                }
                env.add(OpCode::Array(vec.len()));
            },

            Rvalue::Aggregate(AggregateKind::Adt(adt_def, _size, _subst), ref operands) => {
                /*
                    Adt (abstract data type) is an enum. Structs are enums with only one variant.
                    To check whether an adt is an enum or a struct one can use `.adt_kind`.


                    Variants are either VariantKind::{Struct, Tuple, Unit}
                */

                if adt_def.adt_kind() == AdtKind::Struct {
                    // the struct definition is the first variant
                    let ref struct_def = adt_def.variants[0];
                    env.add(OpCode::Tuple(struct_def.fields.len()));
                    for (i, operand) in operands.iter().enumerate() {
                        operand.as_rvalue(env);
                        env.add(OpCode::TupleInit(i));
                    }
                }
            },

            Rvalue::Aggregate(AggregateKind::Closure(_def_id, _subst), ref _aggr) => {
                panic!("TODO closure");
            },

            Rvalue::Ref(ref _region, ref kind, ref lvalue) => {
                lvalue.as_rvalue(env);
                env.add(OpCode::Ref(*kind));
            },

            // example: [0; 5] -> [0, 0, 0, 0, 0]
            Rvalue::Repeat(ref op, ref times) => {
                let size = times.value.as_u64(env.tcx.sess.target.uint_type);
                op.as_rvalue(env);
                env.add(OpCode::Repeat(size as usize));
            },

            Rvalue::Len(ref lvalue) => {
                lvalue.as_rvalue(env);
                env.add(OpCode::Len);
            },

            Rvalue::Cast(ref kind, ref operand, ref ty) => {
               match *kind {
                    CastKind::Unsize => {
                        panic!("todo unisze");
                    },
                    _ => unimplemented!(),
               }
            },

            _ => unimplemented!(),
        }
    }
}


impl<'a> ByteCode for Operand<'a> {

    fn as_rvalue(&self, env: &mut Analyser) {
        match *self {
            Operand::Consume(ref lvalue) => {
                lvalue.as_rvalue(env);
            },

            Operand::Constant(ref constant) => {
                env.unpack_const(&constant.literal);
                // constant.literal is either Item, Value or Promoted
                // assumption: `Item`s are functions.
            }
        }
    }
}

pub fn generate_bytecode<'a, 'tcx>(context: &'a Context<'a, 'tcx>) -> (Program<'a, 'tcx>, DefId) {

    //map krate num -> node id
    let mut program = Program::new(context);
    // let mut build_ins: BTreeMap<u32, BTreeMap<u32, &'a InternedString>> = BTreeMap::new();
    let mut main: Option<DefId> = None;

    for (key, func_mir) in &context.map.map {
        // let mir = map.map.get(key).unwrap();
        // println!("{:?}", mir.id);
        let def_index = context.tcx.map.local_def_id(*key);

        if let Node::NodeItem(item) = context.tcx.map.get(key.to_owned()) {
            // println!("Function: {:?} {:?}", item.name.as_str(), def_index.index.as_u32());
                // let mut collector = FuncGen::new(&context.tcx, context.map);
                // collector.analyse(&func_mir);
                // for (i, block) in collector.blocks.iter().enumerate() {
                //     // println!("{} {:?}", i, block);
                // }
                // let blocks = optimize_blocks(&collector.blocks, func_mir);

                // check for special functions
                let mut opcodes = if item.name.as_str().starts_with("__") {
                    let name = item.name.as_str()[2..].to_string();
                    let command = OpCode::InternalFunc(if name == "out" {
                        InternalFunc::Out
                    } else if name == "print" {
                        InternalFunc::Print
                    } else if name == "met_merge_point" {
                        InternalFunc::MergePoint
                    } else {
                        panic!("Unkown Function {:?}", name);
                    });

                    vec![OpCode::Load(0), command, OpCode::Tuple(0), OpCode::Return]
                    // vec![OpCode::StackFrame(1, 1), OpCode::Load(0), command, OpCode::Tuple(0), OpCode::Return]
                } else {
                    context.mir_to_bytecode(func_mir)
                };

                // trace::detect_load_after_store(&mut opcodes);

                // opcodes = trace::detect_unused_variables(&opcodes);
                // opcodes = trace::remove_unused_consts(&opcodes);

                // opcodes = trace::remove_none(&opcodes);

                println!("{:?}", def_index);
                for opcode in &opcodes {
                    println!("{:?}", opcode);
                }
                println!("");


                program.krates.insert(def_index, Rc::new(opcodes));


                if def_index.krate == 0 && item.name.as_str() == "main" {
                    main = Some(def_index);
                }
        }
    }

    (program, main.unwrap())
}
