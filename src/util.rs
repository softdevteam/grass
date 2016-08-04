
macro_rules! create_bind {
    ($target:ident for $core:ty where $($x:ident : $t:ty),+) => {
        use std::ops::{Deref, DerefMut};

        struct $target<'a> {
            _core: &'a mut $core,
            $($x: &'a mut $t),*
        }

        impl<'a> Deref for $target<'a> {
            type Target = $core;

            fn deref(&self) -> &$core {
                &self._core
            }
        }

        impl<'a> DerefMut for $target<'a> {
            fn deref_mut(&mut self) -> &mut $core {
                &mut self._core
            }
        }

        // make `bind_mut` available on source type
        impl $core {
            fn bind_mut<'a>(&'a mut self, $($x: &'a mut $t),*) -> $target {
                $target { _core: self, $($x: $x),* }
            }
        }

    }
}
