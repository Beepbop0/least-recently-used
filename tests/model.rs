#![cfg(test)]

use least_recently_used::LRUCache;
use quickcheck::{self, Arbitrary, Gen};
use quickcheck_macros::*;
use std::collections::hash_map::{Entry, HashMap};
use std::hash::Hash;
use std::num::{NonZeroU16, NonZeroUsize};

// Fully safe reference LRU to test against
struct SafeLRU<K, V> {
    cache: HashMap<K, V>,
    recency: Vec<K>,
    cap: NonZeroUsize,
}

impl<K: Hash + Eq, V> SafeLRU<K, V> {
    fn new(cap: NonZeroUsize) -> Self {
        Self {
            cache: HashMap::with_capacity(cap.get().saturating_add(1)),
            recency: Vec::with_capacity(cap.get()),
            cap,
        }
    }

    fn len(&self) -> usize {
        self.cache.len()
    }

    fn cap(&self) -> usize {
        self.cap.get()
    }

    fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }

    fn clear(&mut self) {
        self.cache.clear();
        self.recency.clear();
    }

    fn contains(&self, key: &K) -> bool {
        self.cache.contains_key(key)
    }

    fn get(&mut self, key: &K) -> Option<&V> {
        match self.cache.get(key) {
            Some(val) => {
                move_front(&mut self.recency, key);
                Some(val)
            }
            _ => None,
        }
    }

    fn peek(&self, key: &K) -> Option<&V> {
        self.cache.get(key)
    }

    fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        match self.cache.get_mut(key) {
            Some(val) => {
                move_front(&mut self.recency, key);
                Some(val)
            }
            _ => None,
        }
    }

    fn peek_mut(&mut self, key: &K) -> Option<&mut V> {
        self.cache.get_mut(key)
    }

    fn peek_lru(&self) -> Option<(&K, &V)> {
        let lru_key = self.recency.last()?;
        self.cache.get_key_value(lru_key)
    }

    fn put(&mut self, key: K, val: V) -> Option<V>
    where
        K: Clone,
    {
        if let Entry::Occupied(mut occ) = self.cache.entry(key.clone()) {
            let old = occ.insert(val);
            move_front(&mut self.recency, &key);
            Some(old)
        } else {
            if self.cache.len() + 1 > self.cap.get() {
                let lru = self.recency.pop().unwrap();
                self.cache.remove(&lru);
            }
            self.cache.insert(key.clone(), val);
            self.recency.insert(0, key);
            None
        }
    }

    fn remove(&mut self, key: &K) -> Option<V> {
        let val = self.cache.remove(key)?;
        remove_key(&mut self.recency, key);
        Some(val)
    }

    fn remove_lru(&mut self) -> Option<(K, V)> {
        let lru = self.recency.pop()?;
        self.cache.remove_entry(&lru)
    }

    fn resize(&mut self, cap: NonZeroUsize) {
        if self.cap == cap {
            return;
        }

        while self.cache.len() > cap.get() {
            self.remove_lru();
        }
        self.cache.shrink_to_fit();
        self.cap = cap;
    }
}

fn move_front<K: Eq>(recency: &mut Vec<K>, key: &K) {
    let removed_key = remove_key(recency, key);
    recency.insert(0, removed_key);
}

fn remove_key<K: Eq>(recency: &mut Vec<K>, key: &K) -> K {
    let remove_index = find_first(recency.as_slice(), &key).unwrap();
    recency.remove(remove_index)
}

fn find_first<T: Eq>(slice: &[T], elem: &T) -> Option<usize> {
    slice
        .into_iter()
        .enumerate()
        .find(|(_, x)| elem == *x)
        .map(|(i, _)| i)
}

impl<'a, K, V> IntoIterator for &'a SafeLRU<K, V>
where
    K: Hash + Eq,
{
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            cache: self,
            iter: self.recency.iter(),
        }
    }
}

struct Iter<'a, K, V> {
    cache: &'a SafeLRU<K, V>,
    iter: std::slice::Iter<'a, K>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V>
where
    K: Hash + Eq,
{
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|k| {
            let v = self.cache.peek(k).unwrap();
            (k, v)
        })
    }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V>
where
    K: Hash + Eq,
{
    fn next_back(&mut self) -> Option<Self::Item> {
        self.iter.next_back().map(|k| {
            let v = self.cache.peek(k).unwrap();
            (k, v)
        })
    }
}

#[derive(Debug, Clone)]
enum Op<T> {
    Get(T),
    Put(T, T),
    Remove(T),
    Resize(NonZeroUsize),
    RemoveLru,
    Clear,
    Len,
    IsEmpty,
}

impl<T: Arbitrary> Arbitrary for Op<T> {
    fn arbitrary(g: &mut Gen) -> Self {
        match (bool::arbitrary(g), bool::arbitrary(g), bool::arbitrary(g)) {
            (false, false, false) => Self::Put(T::arbitrary(g), T::arbitrary(g)),
            (false, false, true) => Self::Get(T::arbitrary(g)),
            (false, true, false) => Self::Remove(T::arbitrary(g)),
            (false, true, true) => Self::Resize(NonZeroUsize::arbitrary(g)),
            (true, false, false) => Self::RemoveLru,
            (true, false, true) => Self::Clear,
            (true, true, false) => Self::Len,
            (true, true, true) => Self::IsEmpty,
        }
    }
}

#[quickcheck]
fn same_results(cap: NonZeroU16, operations: Vec<Op<Box<i8>>>) -> bool {
    let cap = NonZeroUsize::from(cap);
    let mut unsafe_lru = LRUCache::with_capacity(cap);
    let mut safe_lru = SafeLRU::new(cap);

    operations.into_iter().all(|op| match op {
        Op::Get(key) => unsafe_lru.get(&key) == safe_lru.get(&key),
        Op::Put(key, val) => unsafe_lru.put(key.clone(), val.clone()) == safe_lru.put(key, val),
        Op::Remove(key) => unsafe_lru.remove(&key) == safe_lru.remove(&key),
        Op::Resize(cap) => {
            unsafe_lru.resize(cap);
            safe_lru.resize(cap);
            unsafe_lru.peek_lru() == safe_lru.peek_lru()
        }
        Op::RemoveLru => unsafe_lru.remove_lru() == safe_lru.remove_lru(),
        Op::Clear => {
            unsafe_lru.clear();
            safe_lru.clear();
            unsafe_lru.is_empty() && safe_lru.is_empty()
        }
        Op::Len => unsafe_lru.len() == safe_lru.len(),
        Op::IsEmpty => unsafe_lru.is_empty() == safe_lru.is_empty(),
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

fn run_scenario(cap: NonZeroUsize, operations: Vec<Op<i8>>) {
    let mut safe_lru = SafeLRU::new(cap);
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
                assert_eq!(unsafe_lru.remove(&remove_key), safe_lru.remove(&remove_key))
            }
            Op::Resize(cap) => {
                unsafe_lru.resize(cap);
                safe_lru.resize(cap);
                assert_eq!(unsafe_lru.peek_lru(), safe_lru.peek_lru())
            }
            Op::RemoveLru => {
                assert_eq!(unsafe_lru.remove_lru(), safe_lru.remove_lru())
            }
            Op::Clear => {
                unsafe_lru.clear();
                safe_lru.clear();
                assert!(unsafe_lru.is_empty() && safe_lru.is_empty());
            }

            Op::Len => {
                assert_eq!(unsafe_lru.len(), safe_lru.len())
            }
            Op::IsEmpty => {
                assert_eq!(unsafe_lru.is_empty(), safe_lru.is_empty())
            }
        }
    }
}
