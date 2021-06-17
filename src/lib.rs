#![allow(dead_code)]
use std::borrow::Borrow;
use std::collections::hash_map::HashMap;
use std::hash::{self, Hash};
use std::num::NonZeroUsize;
use std::ptr;

#[derive(Debug)]
pub struct Node<K, V> {
    key: K,
    val: V,
    next: *mut Node<K, V>,
    prev: *mut Node<K, V>,
}

#[derive(Debug)]
struct KeyRef<K>(*const K);

impl<K: Hash> Hash for KeyRef<K> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        unsafe { (*self.0).hash(state) }
    }
}

impl<K: PartialEq> PartialEq for KeyRef<K> {
    fn eq(&self, other: &KeyRef<K>) -> bool {
        unsafe { (*self.0).eq(&(*other.0)) }
    }
}

impl<K: Eq> Eq for KeyRef<K> {}

impl<K> Borrow<K> for KeyRef<K> {
    fn borrow(&self) -> &K {
        unsafe { &*self.0 }
    }
}

impl<K, V> Node<K, V> {
    fn new(key: K, val: V) -> Self {
        Self {
            key,
            val,
            next: ptr::null_mut(),
            prev: ptr::null_mut(),
        }
    }
}

#[derive(Debug)]
struct LinkedList<K, V> {
    head: *mut Node<K, V>,
    tail: *mut Node<K, V>,
}

impl<K, V> Default for LinkedList<K, V> {
    fn default() -> Self {
        Self {
            head: ptr::null_mut(),
            tail: ptr::null_mut(),
        }
    }
}

impl<K, V> LinkedList<K, V> {
    // invariant: Node must not be null
    #[inline]
    fn push(&mut self, node: *mut Node<K, V>) {
        assert!(!node.is_null());
        unsafe {
            let old_head = self.head;
            (*node).next = old_head;
            (*node).prev = ptr::null_mut();
            self.head = node;
            if old_head.is_null() {
                self.tail = node;
            } else {
                (*old_head).prev = node;
            }
        }
    }

    #[inline]
    fn pop_back(&mut self) -> *mut Node<K, V> {
        let lru = self.tail;

        if !lru.is_null() {
            unsafe {
                self.tail = (*lru).prev;
                (*lru).prev = ptr::null_mut();
                if self.tail.is_null() {
                    self.head = ptr::null_mut()
                } else {
                    (*self.tail).next = ptr::null_mut();
                }
            }
        }

        lru
    }

    // Safety: Assumes this node is from the list and non-null
    #[inline]
    fn move_front(&mut self, node: *mut Node<K, V>) {
        assert!(!node.is_null());
        unsafe {
            match (!(*node).prev.is_null(), !(*node).next.is_null()) {
                // is the head node or the only element in the list
                (false, true) | (false, false) => return,
                // pointing to tail node, remove
                (true, false) => {
                    self.pop_back();
                }
                (true, true) => {
                    let next = (*node).next;
                    let prev = (*node).prev;
                    (*prev).next = next;
                    (*next).prev = prev;
                    // not nulling out the node's next and prev pointers since they will be
                    // overwritten by the push method anyways
                }
            }
        }
        self.push(node);
    }

    #[inline]
    fn unlink(&mut self, node: *mut Node<K, V>) {
        assert!(!node.is_null());
        unsafe {
            if (*node).prev.is_null() {
                self.head = (*node).next;
            } else {
                (*(*node).prev).next = (*node).next;
            }

            if (*node).next.is_null() {
                self.tail = (*node).prev;
            } else {
                (*(*node).next).prev = (*node).prev;
            }
        }
    }
}

#[derive(Debug)]
struct LRUCache<K, V> {
    cache: HashMap<KeyRef<K>, *mut Node<K, V>>,
    list: LinkedList<K, V>,
    cap: NonZeroUsize,
}

impl<K, V> LRUCache<K, V> {
    pub fn with_capacity(cap: NonZeroUsize) -> Self {
        Self {
            cache: HashMap::with_capacity(cap.get()),
            list: LinkedList::default(),
            cap,
        }
    }

    pub fn len(&self) -> usize {
        self.cache.len()
    }

    pub fn cap(&self) -> usize {
        self.cap.get()
    }

    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    pub fn clear(&mut self) {
        for (_, entry) in self.cache.drain() {
            let _ = unsafe { Box::from_raw(entry) };
        }
        self.list.head = ptr::null_mut();
        self.list.tail = ptr::null_mut();
    }
}

impl<K, V> LRUCache<K, V>
where
    K: Hash + Eq,
{
    pub fn contains(&self, key: &K) -> bool {
        self.cache.contains_key(key)
    }

    pub fn get<'a>(&'a mut self, key: &K) -> Option<&'a V> {
        match self.cache.get_mut(key) {
            Some(&mut node) => {
                self.list.move_front(node);
                Some(unsafe { &(*node).val })
            }
            _ => None,
        }
    }

    pub fn get_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V> {
        match self.cache.get_mut(key) {
            Some(node) => {
                self.list.move_front(*node);
                Some(unsafe { &mut (**node).val })
            }
            _ => None,
        }
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.cache.remove(key).map(|node| {
            self.list.unlink(node);
            let boxed = unsafe { Box::from_raw(node) };
            boxed.val
        })
    }

    pub fn remove_lru(&mut self) -> Option<(K, V)> {
        let lru = self.list.pop_back();
        (!lru.is_null()).then(|| {
            let boxed = unsafe { Box::from_raw(lru) };
            self.cache.remove(&boxed.key);
            (boxed.key, boxed.val)
        })
    }

    pub fn put(&mut self, key: K, val: V) -> Option<V> {
        let key_ref = KeyRef(&key as *const _);
        if let Some(&mut existing_entry) = self.cache.get_mut(&key_ref) {
            let old_val = unsafe { std::mem::replace(&mut (*existing_entry).val, val) };
            self.list.move_front(existing_entry);
            Some(old_val)
        } else {
            let (key_ref, new_node) = if self.cache.len() + 1 > self.cap.get() {
                unsafe {
                    // SAFETY: the above condition + non-zero capacity means we should never pop a null node
                    let lru_key = &(*self.list.pop_back()).key;
                    let mut lru_node = self.cache.remove(lru_key).unwrap();
                    (*lru_node).key = key;
                    (*lru_node).val = val;
                    (KeyRef(&(*lru_node).key as *const _), lru_node)
                }
            } else {
                let new_node = Box::new(Node::new(key, val));
                (KeyRef(&new_node.key as *const _), Box::into_raw(new_node))
            };
            self.list.push(new_node);
            self.cache.insert(key_ref, new_node);
            None
        }
    }
}

impl<K, V> Drop for LRUCache<K, V> {
    fn drop(&mut self) {
        self.clear()
    }
}

#[cfg(test)]
pub mod tests {
    use super::LRUCache;
    use std::num::NonZeroUsize;

    #[test]
    fn lru_regression() {
        let mut lru_cache = LRUCache::with_capacity(NonZeroUsize::new(3).unwrap());
        lru_cache.put(1, 1); // cache is { 1=1 }
        lru_cache.put(2, 2); // cache is { 1=1, 2=2 }
        lru_cache.put(3, 3); // cache is { 1=1, 2=2, 3=3 }
        lru_cache.put(4, 4); // LRU key was 1, evicts key 1, cache is { 2=2, 3=3, 4=4 }
        assert_eq!(lru_cache.get(&4), Some(&4));
        assert_eq!(lru_cache.get(&3), Some(&3));
        assert_eq!(lru_cache.get(&2), Some(&2));
        assert_eq!(lru_cache.get(&1), None);
        lru_cache.put(5, 5); // LRU key was 4, evicts key 4, cache is { 2=2, 3=3, 5=5 }
        assert_eq!(lru_cache.get(&1), None);
        assert_eq!(lru_cache.get(&2), Some(&2));
        assert_eq!(lru_cache.get(&3), Some(&3));
        assert_eq!(lru_cache.get(&4), None);
        assert_eq!(lru_cache.get(&5), Some(&5));
    }

    #[test]
    fn lru_regression_2() {
        let mut lru_cache = LRUCache::with_capacity(NonZeroUsize::new(2).unwrap());
        lru_cache.put(-87, -17);
        lru_cache.put(72, -109);
        lru_cache.put(47, 82);
        assert_eq!(lru_cache.get(&-87), None);
    }

    #[test]
    fn lru_works() {
        let mut lru_cache = LRUCache::with_capacity(NonZeroUsize::new(2).unwrap());
        lru_cache.put(1, 1); // cache is {1=1}
        lru_cache.put(2, 2); // cache is {1=1, 2=2}
        assert_eq!(lru_cache.get(&1), Some(&1)); // return 1
        lru_cache.put(3, 3); // LRU key was 2, evicts key 2, cache is {1=1, 3=3}
        assert_eq!(lru_cache.get(&2), None); // returns -1 (not found)
        lru_cache.put(4, 4); // LRU key was 1, evicts key 1, cache is {4=4, 3=3}
        assert_eq!(lru_cache.get(&1), None); // return -1 (not found)
        assert_eq!(lru_cache.get(&3), Some(&3)); // return 3
        assert_eq!(lru_cache.get(&4), Some(&4)); // return 4
    }

    #[test]
    fn pop_lru_on_empty_cache_is_none() {
        let mut lru_cache = LRUCache::with_capacity(NonZeroUsize::new(2).unwrap());
        lru_cache.put(1, 1);
        lru_cache.put(2, 2);
        lru_cache.remove(&2);
        lru_cache.remove(&1);
        assert_eq!(lru_cache.remove_lru(), None);
    }
}

#[cfg(test)]
mod quickcheck_tests {
    use super::LRUCache;
    use lru::LruCache as SafeLRU;
    use quickcheck::{self, Arbitrary, Gen};
    use quickcheck_macros::*;
    use std::num::{NonZeroU16, NonZeroUsize};

    #[derive(Debug, Clone)]
    enum Op {
        Get(i8),
        Put(i8, i8),
        Remove(i8),
        RemoveLru,
    }

    impl Arbitrary for Op {
        fn arbitrary(g: &mut Gen) -> Self {
            let (a, b) = (bool::arbitrary(g), bool::arbitrary(g));
            match (a, b) {
                (true, true) => Self::Put(i8::arbitrary(g), i8::arbitrary(g)),
                (true, false) => Self::Get(i8::arbitrary(g)),
                (false, true) => Self::Remove(i8::arbitrary(g)),
                (false, false) => Self::RemoveLru,
            }
        }
    }

    #[quickcheck]
    fn same_results(cap: NonZeroU16, operations: Vec<Op>) -> bool {
        let cap = NonZeroUsize::from(cap);
        let mut unsafe_lru = LRUCache::with_capacity(cap);
        let mut safe_lru = SafeLRU::new(cap.get());

        operations.into_iter().all(|op| match op {
            Op::Get(key) => unsafe_lru.get(&key) == safe_lru.get(&key),
            Op::Put(key, val) => unsafe_lru.put(key, val) == safe_lru.put(key, val),
            Op::Remove(key) => unsafe_lru.remove(&key) == safe_lru.pop(&key),
            Op::RemoveLru => unsafe_lru.remove_lru() == safe_lru.pop_lru(),
        })
    }

    #[test]
    fn regression() {
        let cap = NonZeroUsize::new(2).unwrap();
        let operations = vec![
            Op::Put(51, 47),
            Op::Put(-127, 56),
            Op::Remove(-127),
            Op::Put(120, -89),
            Op::RemoveLru,
        ];
        run_scenario(cap, operations);
    }

    fn run_scenario(cap: NonZeroUsize, operations: Vec<Op>) {
        let mut safe_lru = SafeLRU::new(cap.get());
        let mut unsafe_lru = LRUCache::with_capacity(cap);
        for op in operations {
            match op {
                Op::Get(get_key) => {
                    assert_eq!(unsafe_lru.get(&get_key), safe_lru.get(&get_key));
                }
                Op::Put(put_key, val) => {
                    assert_eq!(unsafe_lru.put(put_key, val), safe_lru.put(put_key, val),);
                }
                Op::Remove(remove_key) => {
                    assert_eq!(unsafe_lru.remove(&remove_key), safe_lru.pop(&remove_key))
                }
                Op::RemoveLru => {
                    assert_eq!(unsafe_lru.remove_lru(), safe_lru.pop_lru())
                }
            }
        }
    }
}
