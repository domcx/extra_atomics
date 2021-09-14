use std::{
    ptr::{self, null_mut},
    sync::atomic::{AtomicPtr, Ordering},
    sync::{atomic::AtomicU32, Arc},
};

pub struct AtomicBox<T: Send> {
    ptr: AtomicPtr<T>,
}

impl<T: Send> AtomicBox<T> {
    pub fn new(value: Option<T>) -> Self {
        let atomic_ptr = match value {
            Some(value) => {
                let boxed = Box::new(value);
                let leaked = Box::leak(boxed);

                AtomicPtr::new(leaked)
            }
            None => AtomicPtr::new(ptr::null_mut()),
        };

        Self { ptr: atomic_ptr }
    }

    pub fn swap(&self, new: Option<T>) -> Option<T> {
        let old_ptr = match new {
            Some(new) => {
                // Allocate the value on the heap.
                let boxed = Box::new(new);
                // Leak the memory so that it is not dropped when this function ends.
                let leaked = Box::leak(boxed);

                // Push the leaked pointer into the AtomicPtr.
                self.ptr.swap(leaked, Ordering::AcqRel)
            }
            None => self.ptr.swap(ptr::null_mut(), Ordering::AcqRel),
        };

        if old_ptr.is_null() {
            return None;
        }

        // Read the old value allocated as a box.
        let box_t = unsafe { Box::from_raw(old_ptr) };

        // Unbox
        Some(*box_t)
    }

    pub fn take(&self) -> Option<T> {
        self.swap(None)
    }

    pub fn unwrap(self) -> Option<T> {
        self.take()
    }
}

impl<T: Send> Drop for AtomicBox<T> {
    fn drop(&mut self) {
        let ptr = self.ptr.load(Ordering::Acquire);
        if ptr.is_null() {
            // There is nothing to de-allocate.
            return;
        }

        unsafe {
            Box::from_raw(ptr);
        }
    }
}

pub struct AtomicArc<T> {
    ptr: AtomicPtr<T>,
    readers: AtomicU32,
}

impl<T> AtomicArc<T> {
    pub fn new(value: Arc<T>) -> Self {
        let raw = Arc::into_raw(value);
        let ptr = AtomicPtr::new(raw as *mut T);
        Self {
            ptr,
            readers: Default::default(),
        }
    }

    pub fn load(&self) -> Arc<T> {
        let ptr = loop {
            // First load a reference. Relaxed is OK because of the stricter Acquire ordering
            // below.
            let relaxed = self.ptr.load(Ordering::Relaxed);
            if relaxed.is_null() {
                // The pointer is null. This will occur when the pointer is in the process of being
                // replaced.
                // When this is reached the next pointer should be ready almost immediately.
                continue;
            }

            // Eagerly increase the number of readers. This ensures that one we have a ptr, that it
            // cannot be de-allocated before we read it.
            self.readers.fetch_add(1, Ordering::AcqRel);

            // Load the pointer with Acquire ordering.
            // This second load is necessary for a couple reasons.
            // 1. The Relaxed ordering above is not a strong enough guarantee.
            // 2. We must ensure that the pointer did not change after we increased the number of
            //    readers. If it did, then that means a swap was ongoing and the pointer is not
            //    safe to read.
            let acquire = self.ptr.load(Ordering::Acquire);
            if acquire.is_null() || relaxed != acquire {
                // If the most recently loaded pointer was null, or was not the same as the first
                // loaded, then this pointer is not safe to read. We must decrement our reader
                // count and start over.
                self.readers.fetch_sub(1, Ordering::AcqRel);
                continue;
            }

            // This pointer will not be deallocated while we are reading it (at least by this
            // code).
            break acquire;
        };

        // SAFETY: So long as the pointer loaded above is not deallocated prior to this read, we
        // know this to be safe. This must be true, because we have prevented the pointer from
        // being deallocated by increasing the number of readers.
        let loaded = unsafe {
            Arc::increment_strong_count(ptr);
            Arc::from_raw(ptr as *const T)
        };

        // "Release" this reader.
        self.readers.fetch_sub(1, Ordering::AcqRel);

        loaded
    }

    pub fn swap(&self, new_value: Arc<T>) -> Result<Arc<T>, Arc<T>> {
        // Load the current value.
        let read = self.ptr.load(Ordering::Acquire);
        if read.is_null() {
            // Something else is still writing so we cannot do anything.
            return Err(new_value);
        }

        // Confirm that the pointer that we have is the one still stored, and swap it with null. We
        // want to make sure that we do not compete with other threads to swap in the new pointer.
        // If we did, we might have spurious reads of the newly written value and it would not be
        // safe to
        if self
            .ptr
            .compare_exchange(read, null_mut(), Ordering::AcqRel, Ordering::Relaxed)
            .is_err()
        {
            // Another thread beat us to the punch, so we drop out of the race.
            return Err(new_value);
        }

        if self.readers.load(Ordering::Acquire) > 0 {
            loop {
                // Relaxed loads guarantee that a value that a thread cannot "see" a value older
                // than the last read. If the last read was greater than 0, then we are waiting for
                // all readers to wait and the count to drop to 0.
                if self.readers.load(Ordering::Relaxed) == 0 {
                    break;
                }
            }
        }

        self.ptr
            .store(Arc::into_raw(new_value) as *mut T, Ordering::Release);

        Ok(unsafe { Arc::from_raw(read) })
    }
}

impl<T> Drop for AtomicArc<T> {
    fn drop(&mut self) {
        let ptr = self.ptr.load(Ordering::Acquire);
        if ptr.is_null() {
            return;
        }

        unsafe {
            Arc::from_raw(ptr);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::AtomicArc;
    use super::AtomicBox;
    use std::sync::atomic::AtomicUsize;
    use std::sync::mpsc;
    use std::sync::Arc;
    use std::sync::RwLock;
    use std::time::SystemTime;

    struct RwLockTests<T> {
        lock: RwLock<Arc<T>>,
    }

    impl<T> RwLockTests<T> {
        fn new(init: Arc<T>) -> Self {
            Self {
                lock: RwLock::new(init),
            }
        }

        fn swap(&self, new_value: Arc<T>) -> Result<Arc<T>, Arc<T>> {
            let mut locked = self.lock.write().unwrap();
            let old = locked.clone();
            *locked = new_value;

            Ok(old)
        }

        fn load(&self) -> Arc<T> {
            self.lock.read().unwrap().clone()
        }
    }

    #[test]
    fn can_swap_option() {
        let a = AtomicBox::new(None);

        assert_eq!(a.swap(Some(1)), None);
    }

    #[test]
    fn arc_swap() {
        let new_arc = Arc::new(0);
        let a = AtomicArc::new(new_arc);

        {
            let load = a.load();
            assert_eq!(Arc::strong_count(&load), 2);
        }

        let old = a.swap(Arc::new(1));
        let old = old.unwrap();

        assert_eq!(*old, 0);
        assert_eq!(Arc::strong_count(&old), 1);
    }

    #[test]
    fn threaded_swaps_loads() {
        const COUNT: u128 = 10000;
        let swap = Arc::new(AtomicArc::new(Arc::new(0)));

        let failure = Arc::new(AtomicUsize::new(0));
        let (send, recv) = mpsc::channel();
        let total: Arc<AtomicUsize> = Default::default();
        for _ in 0..3 {
            let swapped = swap.clone();
            let sender = send.clone();
            std::thread::spawn(move || {
                let t0 = SystemTime::now();
                // Just holds the loop up below.
                let _ = sender;
                for _ in 0..COUNT {
                    swapped.load();
                }

                let elapsed = t0.elapsed().unwrap();
                println!("loads in {:?}; ", elapsed,)
            });
        }
        for _ in 0..3 {
            let failures = failure.clone();
            let sender = send.clone();
            let swapped = swap.clone();
            let total = total.clone();
            std::thread::spawn(move || {
                let t0 = SystemTime::now();
                let sender = sender;
                for _ in 0..COUNT {
                    let next = total.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                    match swapped.swap(Arc::new(next)) {
                        Ok(swap) => {
                            drop(swap);
                        }
                        Err(_) => {
                            failures.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
                        }
                    }
                }

                let elapsed = t0.elapsed().unwrap();
                println!("swaps in {:?}; ", elapsed,)
            });
        }
        drop(send);

        let mut all: Vec<Arc<usize>> = vec![];

        while let Ok(a) = recv.recv() {
            assert_eq!(all.iter().any(|y| Arc::ptr_eq(y, &a)), false);
            all.push(a);
        }

        drop(swap);

        let count: Vec<_> = all
            .iter()
            .map(|y| Arc::strong_count(y))
            .filter(|y| *y > 1)
            .collect();

        let failures = failure.load(std::sync::atomic::Ordering::Acquire);
        println!(
            "Failures ({}); More than 1 ({}) {:?}",
            failures,
            count.len(),
            count
        );
    }
}
