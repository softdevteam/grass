

pub struct Memory {
    // Frames/Locals/
    stack: Vec<Vec<Rc<Cell<R_BoxedValue>>>>,
}


fn o_load(&mut self, local_idx: usize) {
    let cell_ptr = self.memory.stack[self.active_frame_idx()][local_idx].clone();
    self.stack.push(StackVal::Ref(cell_ptr));
}

fn o_use(&mut self) {
    let val = self.stack.pop().unwrap().as_val();
    self.stack.push(val);
}

fn o_store(&mut self, local_idx) {
    let val = self.stack.pop().unwrap().unwrap_val();
    let cell = self.memory.stack[self.active_frame_idx()][local_idx].clone();
    let boxed = cell.borrow_mut();
    *boxed = val;
}