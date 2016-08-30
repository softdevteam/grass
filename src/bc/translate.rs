
use rustc_serialize::{json, hex};

// TODO: Ref on Projections

/**
 * Uses rust's MIR to generate opcodes.
 * HIR: tree, MIR: flow-graph, LIR: linear
 */


use std::collections::{BTreeMap, HashSet};
use std::rc::Rc;

use bc::bytecode::{self, OpCode, InternalFunc};
use core::objects::{R_BoxedValue, R_Function};

// XXX
pub type Function = Vec<OpCode>;

use rustc::mir::repr::{
    BasicBlock, BasicBlockData, Mir,
    Constant, Literal, Operand,
    Lvalue, Rvalue, BinOp, UnOp,
    Statement, StatementKind, Terminator, TerminatorKind,
    ProjectionElem, AggregateKind,
    Field, CastKind
};

use rustc::mir::mir_map::MirMap;
use rustc::middle::const_val::ConstVal;

use rustc::hir::map::Node;
use rustc::hir::def_id::{DefId, DefIndex};

use rustc::ty::{TyCtxt, AdtKind, VariantKind, TyS, TypeVariants};
use rustc::middle::cstore::LinkagePreference;

// use syntax_pos::DUMMY_SP;
use rustc_data_structures::indexed_vec::Idx;

use rustc::middle::cstore::CrateStore;

pub struct IdMap {
    map: BTreeMap<DefId, usize>,
    cstore: Rc<for<'a> CrateStore<'a>>,
}

impl IdMap {
    fn new(cstore: Rc<for<'a> CrateStore<'a>>) -> Self {
        let mut map = BTreeMap::new();
        // 0 is reserved for merge_point
        map.insert(DefId::local(DefIndex::new(0)), 0);
        IdMap { map: map, cstore: cstore }
    }

    fn is_merge_point(&self, def_id: &DefId) -> bool {
        if def_id.krate == 0 {
            return false;
        }
        // println!("{:?}", def_id);
        let krate = self.cstore.crate_name(def_id.krate);
        let name = self.cstore.item_name(*def_id);

        "merge_point" == &self.cstore.item_name(*def_id).as_str()
    }

    fn get_index(&mut self, key: &DefId) -> usize {
        // 0 is reserved for merge_point
        if self.is_merge_point(key) {
            self.map.insert(*key, 0);
            return 0;
        }

        let length = self.map.len();
        self.map.entry(*key).or_insert(length).clone()
    }
}

pub struct Program<'a, 'tcx: 'a> {
    context: &'a Context<'a, 'tcx>,
    // cache: BTreeMap<DefId, Vec<OpCode>>,
    cache: BTreeMap<usize, R_Function>,

    defid_map: IdMap,

}

impl<'a, 'tcx> Program<'a, 'tcx> {
    fn new(context: &'a Context<'a, 'tcx>) -> Program<'a, 'tcx> {
        Program {context: context, cache: BTreeMap::new(),
            defid_map: IdMap::new(context.tcx.sess.cstore.clone()) }
    }


    pub fn load_fn_from_def_id(&mut self, def_id: DefId) {
        let local_id = self.defid_map.get_index(&def_id);

        if !self.cache.contains_key(&local_id) {
            let mir = match self.context.map.map.get(&def_id) {
                Some(mir) => mir,
                None => {
                    // panic!("could not load function {:?}", def_id),
                    // let cs = &self.context.tcx.sess.cstore;
                    // let mir = cs.maybe_get_item_mir(self.context.tcx, def_id).unwrap_or_else(||{
                        // panic!("no mir for {:?}", def_id);
                    // });
                    // unimplemented!();
                    return;
                }
            };

            // let cs = &self.context.tcx.sess.cstore;
            // println!("71 {:?}", self.context.tcx.map.as_local_node_id(def_id));
            // let mir = match self.context.tcx.map.as_local_node_id(def_id) {
                // Some(node_id) => {
                    // let mir = self.context.map.map.get(node_id).unwrap();
                    // mir
                // },
                // None => {
            //         let mir = cs.maybe_get_item_mir(self.context.tcx, def_id).unwrap_or_else(||{
            //             panic!("no mir for {:?}", def_id);
            //         });
            //         *mir
                // },
            // };

            let mir_analyser = MirAnalyser::analyse(mir, &self.context.tcx, &mut self.defid_map);
            let mut func = R_Function::default();
            func.opcodes = mir_analyser.opcodes;

            self.cache.insert(local_id, func);
            for func in &mir_analyser.seen_fns {
                if !self.defid_map.is_merge_point(&def_id) {
                    self.load_fn_from_def_id(*func);
                }
            }
        }
    }
}

/// enum to avoid polluting `OpCodes` with later unused variants
#[derive(Debug)]
enum MetaOpCode {
    Goto(BasicBlock),
    GotoIf(BasicBlock),
    OpCode(OpCode),
}


pub trait ByteCode {
    fn to_opcodes(&self, &mut BlockAnalyser) {unimplemented!()}
    fn as_rvalue(&self, &mut BlockAnalyser) {unimplemented!()}
}


pub struct Context<'a, 'tcx: 'a> {
    pub tcx: TyCtxt<'a, 'tcx, 'tcx>,
    pub map: &'a MirMap<'tcx>,
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

/// Analyse a single function (in form of a mir)
pub struct MirAnalyser<'a, 'tcx: 'a> {
    /// typecontext is needed to lookup types (e.g. real size of usize)
    tcx: &'a TyCtxt<'a, 'tcx, 'tcx>,

    /// the function to analyse
    mir: &'a Mir<'a>,

    blocks: Vec<Vec<MetaOpCode>>,

    opcodes: Vec<OpCode>,

    seen_fns: HashSet<DefId>,
}


impl<'a, 'tcx> MirAnalyser<'a, 'tcx> {
    pub fn analyse<'b>(mir: &'a Mir, tcx: &'a TyCtxt<'a, 'tcx, 'tcx>, defid_map: &'b mut IdMap)
    -> MirAnalyser<'a, 'tcx> {
        let mut analyser = MirAnalyser { mir: mir,
                      tcx: tcx,
                      blocks: Vec::new(),
                      opcodes: Vec::new(),
                      seen_fns: HashSet::new() };


        analyser.analyse_blocks(defid_map);
        analyser.opcodes = analyser.flatten_blocks();

        analyser
    }

    fn analyse_blocks(&mut self, defid_map: &mut IdMap) {
        for bb in self.mir.basic_blocks().iter() {
            let block = {
                BlockAnalyser::analyse(bb, self, defid_map).opcodes
            };
            self.blocks.push(block);
        }
    }

    // replace Gotos with Jumps
    fn flatten_blocks(&self) -> Vec<OpCode> {
        let mut indicies = Vec::new();
        let mut n = 0usize;
        for block in &self.blocks {
            indicies.push(n);
            n += block.len();
        }


        let mut opcodes = Vec::new();

        let new_target = |block: &BasicBlock| indicies[block.index()];

        for (current, opcode) in self.blocks.iter().flat_map(|v| v).enumerate() {
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

        opcodes
    }

}

trait LocalIndex {
    fn arg_to_local(&self, n: usize) -> usize;
    fn var_to_local(&self, n: usize) -> usize;
    fn tmp_to_local(&self, n: usize) -> usize;
}

impl<'a> LocalIndex for Mir<'a> {
    fn arg_to_local(&self, n: usize) -> usize {
        n
    }

    fn var_to_local(&self, n: usize) -> usize {
        self.arg_decls.len() + n
    }

    fn tmp_to_local(&self, n: usize) -> usize {
        self.var_to_local(n) + self.var_decls.len()
    }
}

pub struct BlockAnalyser<'b, 'a:'b, 'tcx: 'a> {
    block: &'a BasicBlockData<'a>,
    env: &'b mut MirAnalyser<'a, 'tcx>,
    opcodes: Vec<MetaOpCode>,
    defid_map: &'b mut IdMap,
}

impl<'b, 'a, 'tcx> BlockAnalyser<'b, 'a, 'tcx> {
    pub fn analyse(block: &'a BasicBlockData<'a>,
                   env: &'b mut MirAnalyser<'a, 'tcx>,
                   defid_map: &'b mut IdMap
        ) -> Self {
        let mut analyser = BlockAnalyser { block: block,
                        env: env,
                        opcodes: Vec::new(),
                        defid_map: defid_map };

        for statement in &block.statements {
            statement.to_opcodes(&mut analyser);
        }
        block.terminator().to_opcodes(&mut analyser);

        analyser
    }

    fn add(&mut self, opcode: OpCode) {
        self.opcodes.push(MetaOpCode::OpCode(opcode));
    }
}

impl<'a> ByteCode for Statement<'a> {
    fn to_opcodes(&self, env: &mut BlockAnalyser) -> () {
        match self.kind {
            StatementKind::Assign(ref lvalue, ref rvalue) => {
                rvalue.to_opcodes(env);
                lvalue.to_opcodes(env);
            },
            StatementKind::StorageLive(_) | StatementKind::StorageDead(_) => {},

            StatementKind::SetDiscriminant{..} => {
                println!("{:?}", self.kind);
                unimplemented!();
            }
        }
    }
}


impl<'a> ByteCode for Terminator<'a> {

    fn to_opcodes(&self, env: &mut BlockAnalyser) {
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
                        MetaOpCode::OpCode(OpCode::todo_s("NO RETURN"))
                    }
                }
            },

            TerminatorKind::Assert{ref cond, expected, msg: _, target, cleanup} => {
                cond.as_rvalue(env);
                env.add(OpCode::ConstValue(R_BoxedValue::Bool(expected)));
                env.add(OpCode::BinOp(bytecode::BinOp::Eq));
                env.opcodes.push(MetaOpCode::GotoIf(target));
                // XXX: handle msg
                if let Some(bb) = cleanup {
                    MetaOpCode::Goto(bb)
                } else {
                    MetaOpCode::OpCode(OpCode::Panic)
                }
            },


            ref other => {
                MetaOpCode::OpCode(match *other {
                    TerminatorKind::Return => OpCode::Return,

                    TerminatorKind::Resume => OpCode::Resume,

                    TerminatorKind::Drop{location: ref lvalue, target: _, unwind: _} => {
                        lvalue.as_rvalue(env);
                        OpCode::todo_s("Drop")
                    },


                    _ => OpCode::Todo(format!("Terminator {:?}", other)),
                })
            }

        };
        env.opcodes.push(op);
    }
}


impl<'a> ByteCode for Lvalue<'a> {
    /// lvalue = <rvalue>
    fn to_opcodes(&self, env: &mut BlockAnalyser) {
        match *self {
            //a = <rvalue>
            Lvalue::Var(n)  => {
                let n = env.env.mir.var_to_local(n.index());
                env.add(OpCode::Store(n));
            },

            //tmp_x = <rvalue>
            Lvalue::Temp(n) => {
                let n = env.env.mir.tmp_to_local(n.index());
                env.add(OpCode::Store(n));
            },

            Lvalue::Arg(_n)  => unreachable!(),
            Lvalue::Static(def_id)  => {
                let local = env.defid_map.get_index(&def_id);
                env.add(OpCode::StoreStatic(local));
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
                        //a
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
    fn as_rvalue(&self, env: &mut BlockAnalyser) {
        let opcode = match *self {
            Lvalue::Var(n) => {
                let n = env.env.mir.var_to_local(n.index());
                OpCode::Load(n)
            },
            Lvalue::Temp(n) => {
                let n = env.env.mir.tmp_to_local(n.index());
                OpCode::Load(n)
            },
            Lvalue::Arg(n) => {
                let n = env.env.mir.arg_to_local(n.index());
                OpCode::Load(n)
            },

            Lvalue::Static(def_id) => {
                let local = env.defid_map.get_index(&def_id);
                OpCode::Static(local)
            },
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
                        // [index]
                        index.as_rvalue(env);
                        //lvlaue
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

    fn to_opcodes(&self, env: &mut BlockAnalyser) {
        match *self {
            Rvalue::Use(ref op) => {
                op.as_rvalue(env);
                // in other words copy the value
                env.add(OpCode::Use);
            },

            Rvalue::CheckedBinaryOp(binop, ref left, ref right) => {
                left.as_rvalue(env);
                right.as_rvalue(env);
                env.add(OpCode::CheckedBinOp(bytecode::BinOp::new(binop)));
            },

            Rvalue::BinaryOp(binop, ref left, ref right) => {
                left.as_rvalue(env);
                right.as_rvalue(env);
                env.add(OpCode::BinOp(bytecode::BinOp::new(binop)));
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
                    let struct_def = &adt_def.variants[0];
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
                // env.add(OpCode::Ref(*kind));
                env.add(OpCode::Ref);
            },

            // example: [0; 5] -> [0, 0, 0, 0, 0]
            Rvalue::Repeat(ref op, ref times) => {
                let size = times.value.as_u64(env.env.tcx.sess.target.uint_type);
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
                        operand.as_rvalue(env);
                        env.add(OpCode::Unsize);
                        // panic!("todo unsize");
                    },
                    _ => unimplemented!(),
               }
            },

            Rvalue::UnaryOp(kind, ref operand) => {
                operand.as_rvalue(env);
                env.add(match kind {
                    UnOp::Not => OpCode::Not,
                    UnOp::Neg => OpCode::Neg,
                });
            },

            Rvalue::Box(ref t) => {
                println!("{:?}", t);
            },

            ref other => {
                println!("{:?}", other);
                unimplemented!();
            },
        }
    }
}

fn unpack_const(literal: &Literal, ty: &TyS, env: &mut BlockAnalyser) -> OpCode {
    OpCode::ConstValue(match *literal {
        Literal::Value{ ref value } => {

            use rustc_const_math::ConstInt::*;
            use rustc_const_math::ConstFloat::*;
            use rustc_const_math::{Us16, Us32, Us64};

            use rustc::middle::const_val::ConstVal::{
                Integral, Float, Bool, Function, Array,
                Str, ByteStr, Tuple, Struct, Repeat, Char, Dummy
            };

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

                Float(F32(f)) => R_BoxedValue::F64(f as f64),
                Float(F64(f)) => R_BoxedValue::F64(f),

                // should this ever happen?
                Float(FInfer{f32: _, f64: _}) => unimplemented!(),
                Integral(Infer(_u)) => unimplemented!(),
                Integral(InferSigned(_i)) => unimplemented!(),

                Bool(b) => R_BoxedValue::Bool(b),

                Str(ref interned_str) => {
                    unimplemented!();
                },

                ByteStr(_)
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
            // println!("TTT {:?} - {:?}", ty.sty, def_id);
            match ty.sty {
                TypeVariants::TyFnDef(..) => {
                    // XXX: sort of hacky
                    env.env.seen_fns.insert(def_id);
                    let local = env.defid_map.get_index(&def_id);
                    R_BoxedValue::Func(local)
                },
                _ => {
                    env.env.seen_fns.insert(def_id);

                    let local = env.defid_map.get_index(&def_id);
                    R_BoxedValue::Static(local)
                },
            }
        },

        Literal::Promoted {index} => {
            R_BoxedValue::Usize(index.index())
        },
    })
}

impl<'a> ByteCode for Operand<'a> {

    fn as_rvalue(&self, env: &mut BlockAnalyser) {
        match *self {
            Operand::Consume(ref lvalue) => {
                lvalue.as_rvalue(env);
            },

            Operand::Constant(ref constant) => {
                let const_val = unpack_const(&constant.literal, constant.ty, env);
                env.add(const_val);
                // constant.literal is either Item, Value or Promoted
                // assumption: `Item`s are functions.
            }
        }
    }
}


pub fn generate_bytecode<'a, 'tcx>(context: &'a Context<'a, 'tcx>, main: DefId) -> Program<'a, 'tcx> {

    let mut program = Program::new(context);
    program.load_fn_from_def_id(main);
    // println!("];");

    println!("let program = [ &[], ");
    for idx in 1..program.cache.len() {
        if let Some(func) = program.cache.get(&idx) {
            let output: Vec<String> = func.opcodes.iter().map(|oc|oc.to_rs()).collect();
            println!("    [{}],", output.join(", "));
        } else {
            println!("    [],", );
        }
    }
    println!("];");
            // let output: Vec<String> = mir_analyser.opcodes.iter().map(|oc|oc.to_rs()).collect();
            // println!("  &[{}]", output.join(", "));


    // {
    //     use core::objects::R_BoxedValue::*;
    //         let program = [
    //             vec![OpCode::ConstValue(Func(1)), OpCode::Use, OpCode::Store(0), OpCode::Load(0), OpCode::Use, OpCode::Store(2), OpCode::Load(2), OpCode::Call, OpCode::Store(1), OpCode::Skip(1), OpCode::Tuple(0), OpCode::Return],
    //             vec![OpCode::Tuple(0), OpCode::Skip(1), OpCode::Return],
    //         ];
    // }

    program
}
