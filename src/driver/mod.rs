

mod meta;


use std::collections::BTreeMap;

use bc::bytecode::OpCode;


#[derive(Default)]
pub struct Driver {
    tracer: Tracer,
}

impl Driver {
    pub fn merge_point(&mut self, program: &[&[OpCode]], pc: usize) {
        let res = self.tracer.handle_mergepoint(pc as u64);
    }
}


type HashValue = u64;
const HOT_LOOP_THRESHOLD: usize = 10;

#[derive(Default)]
struct Tracer {
    /// counter for program positions
    counter: BTreeMap<HashValue, usize>,
    traces: BTreeMap<HashValue, Vec<OpCode>>,
    loop_start: HashValue,

    active: Option<Vec<OpCode>>,
}

// glorified Option
pub enum MergePointResult<'a> {
    Trace(&'a Vec<OpCode>),
    StartTrace,
    None,
}

impl Tracer {
    pub fn handle_mergepoint(&mut self, key: HashValue) -> MergePointResult {

        if self.traces.contains_key(&key) {
            return MergePointResult::Trace(self.traces.get(&key).unwrap());
        }
        // increase counter for program position
        else if self.active.is_none() {
            let count = {
                let count = self.counter.entry(key).or_insert(0);
                *count += 1;
                *count
            };

            if count > HOT_LOOP_THRESHOLD {
                self.active = Some(Vec::new());
                self.counter.clear();
                self.loop_start = key;
                return MergePointResult::StartTrace;
            }
        }
        // close the loop
        else if key == self.loop_start {
            let active = self.active.take().unwrap();
            self.traces.insert(key, active);
        }

        MergePointResult::None
    }
}

// #[macro_export]
// macro_rules! new_merge_point {
//     ($target:ident for $core:ty where $($x:ident : $t:ty),+) => {

//         pub struct $target<'a, 'core:'a> {
//             pub _core: &'a mut $core,
//             $(pub $x: &'a mut $t),*
//         }

//         impl<'a, 'core> Deref for $target<'a, 'core> {
//             type Target = $core;

//             fn deref(&self) -> &$core {
//                 &self._core
//             }
//         }

//         impl<'a, 'core> DerefMut for $target<'a, 'core> {
//             fn deref_mut(&mut self) -> &mut $core {
//                 &mut self._core
//             }
//         }

//         // make `bind_mut` available on source type
//         impl<'core> $core {
//             pub fn bind_mut<'a>(&'a mut self, $($x: &'a mut $t),*) -> $target<'a, 'core> {
//                 $target { _core: self, $($x: $x),* }
//             }

//             // pub fn get_trace_for()
//         }

//         impl<'a, 'core> $target<'a, 'core> {

//         }
//     }
// }

// use std::cmp::PartialEq;

// pub trait MergePoint: PartialEq {
// }


// pub fn merge_point<T : MergePoint>(mp: &mut T) {

// }
