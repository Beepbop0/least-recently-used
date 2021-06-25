use std::borrow::Borrow;
use std::collections::hash_map::HashMap;
use std::hash::{self, Hash};
use std::marker::PhantomData;
use std::num::NonZeroUsize;
use std::ptr::NonNull;

#[derive(Debug)]
struct Node<K, V> {
    key: K,
    val: V,
    next: Option<NonNull<Node<K, V>>>,
    prev: Option<NonNull<Node<K, V>>>,
}

#[derive(Debug)]
struct KeyRef<K>(NonNull<K>);

impl<K: Hash> Hash for KeyRef<K> {
    fn hash<H: hash::Hasher>(&self, state: &mut H) {
        (unsafe { self.0.as_ref() }).hash(state)
    }
}

impl<K: PartialEq> PartialEq for KeyRef<K> {
    fn eq(&self, other: &KeyRef<K>) -> bool {
        unsafe { self.0.as_ref().eq(other.0.as_ref()) }
    }
}

impl<K: Eq> Eq for KeyRef<K> {}

impl<K> Borrow<K> for KeyRef<K> {
    fn borrow(&self) -> &K {
        unsafe { self.0.as_ref() }
    }
}

impl<K, V> Node<K, V> {
    fn new(key: K, val: V) -> Self {
        Self {
            key,
            val,
            next: None,
            prev: None,
        }
    }
}

#[derive(Debug)]
struct LinkedList<K, V> {
    head: Option<NonNull<Node<K, V>>>,
    tail: Option<NonNull<Node<K, V>>>,
}

impl<K, V> Default for LinkedList<K, V> {
    fn default() -> Self {
        Self {
            head: None,
            tail: None,
        }
    }
}

impl<K, V> LinkedList<K, V> {
    #[inline]
    fn push(&mut self, mut node: NonNull<Node<K, V>>) {
        unsafe {
            let old_head = std::mem::replace(&mut self.head, Some(node));
            let node_ref = node.as_mut();
            node_ref.next = old_head;
            node_ref.prev = None;
            match old_head {
                None => self.tail = Some(node),
                Some(mut old_head) => old_head.as_mut().prev = Some(node),
            }
        }
    }

    #[inline]
    fn pop_back(&mut self) -> Option<NonNull<Node<K, V>>> {
        match self.tail.take() {
            Some(mut lru) => unsafe {
                let lru_ref = lru.as_mut();
                self.tail = lru_ref.prev.take();
                match self.tail {
                    None => self.head = None,
                    Some(mut tail) => tail.as_mut().next = None,
                }
                Some(lru)
            },
            _ => None,
        }
    }

    // Safety: Assumes this node is from the list
    #[inline]
    unsafe fn move_front(&mut self, mut node: NonNull<Node<K, V>>) {
        let node_ref = node.as_mut();
        match (node_ref.prev, node_ref.next) {
            // already at the front
            (None, Some(_)) | (None, None) => return,
            (Some(_), None) => {
                self.pop_back();
            }
            (Some(mut prev), Some(mut next)) => {
                prev.as_mut().next = Some(next);
                next.as_mut().prev = Some(prev);
            }
        }
        self.push(node);
    }

    // Safety: Assumes this node is from the list
    #[inline]
    unsafe fn unlink(&mut self, mut node: NonNull<Node<K, V>>) {
        let node = node.as_mut();
        match node.prev {
            None => self.head = node.next,
            Some(mut prev) => prev.as_mut().next = node.next,
        }
        match node.next {
            None => self.tail = node.prev,
            Some(mut next) => next.as_mut().prev = node.prev,
        }
    }
}

#[derive(Debug)]
pub struct LRUCache<K, V> {
    cache: HashMap<KeyRef<K>, NonNull<Node<K, V>>>,
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
            let _ = unsafe { Box::from_raw(entry.as_ptr()) };
        }
        self.list.head = None;
        self.list.tail = None;
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
            Some(&mut node) => unsafe {
                self.list.move_front(node);
                Some(&node.as_ref().val)
            },
            _ => None,
        }
    }

    pub fn peek<'a>(&'a self, key: &K) -> Option<&'a V> {
        self.cache
            .get(key)
            .map(|node| unsafe { &node.as_ref().val })
    }

    pub fn get_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V> {
        match self.cache.get_mut(key) {
            Some(&mut mut node) => unsafe {
                self.list.move_front(node);
                Some(&mut node.as_mut().val)
            },
            _ => None,
        }
    }

    pub fn peek_mut<'a>(&'a mut self, key: &K) -> Option<&'a mut V> {
        self.cache
            .get_mut(key)
            .map(|node| unsafe { &mut node.as_mut().val })
    }

    pub fn peek_lru<'a>(&'a self) -> Option<(&'a K, &'a V)> {
        self.list
            .tail
            .map(|node| unsafe { (&node.as_ref().key, &node.as_ref().val) })
    }

    pub fn remove(&mut self, key: &K) -> Option<V> {
        self.cache.remove(key).map(|node| unsafe {
            self.list.unlink(node);
            let boxed = Box::from_raw(node.as_ptr());
            boxed.val
        })
    }

    pub fn remove_lru(&mut self) -> Option<(K, V)> {
        match self.list.pop_back() {
            Some(lru) => {
                let key = unsafe { &lru.as_ref().key };
                let node = self.cache.remove(key).unwrap();
                let boxed = unsafe { Box::from_raw(node.as_ptr()) };
                Some((boxed.key, boxed.val))
            }
            _ => None,
        }
    }

    pub fn put(&mut self, key: K, val: V) -> Option<V> {
        let key_ref = KeyRef(NonNull::from(&key));
        if let Some(&mut mut existing_entry) = self.cache.get_mut(&key_ref) {
            unsafe {
                let old_val = std::mem::replace(&mut existing_entry.as_mut().val, val);
                self.list.move_front(existing_entry);
                Some(old_val)
            }
        } else {
            let (key_ref, new_node) = if self.cache.len() + 1 > self.cap.get() {
                unsafe {
                    // SAFETY: the above condition + non-zero capacity means we should never pop a null node
                    let lru_key = &self.list.pop_back().unwrap().as_ref().key;
                    let mut lru_node = self.cache.remove(lru_key).unwrap();
                    lru_node.as_mut().key = key;
                    lru_node.as_mut().val = val;
                    (KeyRef(NonNull::from(&lru_node.as_ref().key)), lru_node)
                }
            } else {
                let new_node = Box::new(Node::new(key, val));
                (KeyRef(NonNull::from(&new_node.key)), unsafe {
                    NonNull::new_unchecked(Box::into_raw(new_node))
                })
            };
            self.list.push(new_node);
            self.cache.insert(key_ref, new_node);
            None
        }
    }

    pub fn resize(&mut self, cap: NonZeroUsize) {
        if self.cap == cap {
            return;
        }

        while self.cache.len() > cap.get() {
            self.remove_lru();
        }
        self.cache.shrink_to_fit();
        self.cap = cap;
    }

    pub fn iter<'a>(&'a self) -> Iter<'a, K, V> {
        IntoIterator::into_iter(self)
    }

    pub fn iter_mut<'a>(&'a mut self) -> IterMut<'a, K, V> {
        IntoIterator::into_iter(self)
    }
}

impl<K, V> Drop for LRUCache<K, V> {
    fn drop(&mut self) {
        self.clear()
    }
}

impl<K, V> Clone for LRUCache<K, V>
where
    K: Hash + Eq + Clone,
    V: Clone,
{
    fn clone(&self) -> Self {
        let mut new_cache = LRUCache::with_capacity(self.cap);
        // insert from least-used to most used
        for (k, v) in self.into_iter().rev() {
            new_cache.put(k.clone(), v.clone());
        }

        new_cache
    }
}

impl<'a, K, V> IntoIterator for &'a LRUCache<K, V> {
    type Item = (&'a K, &'a V);
    type IntoIter = Iter<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            next: self.list.head,
            prev: self.list.tail,
            len: self.len(),
            _marker: PhantomData,
        }
    }
}

pub struct Iter<'a, K, V> {
    next: Option<NonNull<Node<K, V>>>,
    prev: Option<NonNull<Node<K, V>>>,
    len: usize,
    _marker: PhantomData<&'a Node<K, V>>,
}

impl<'a, K, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        self.next.map(|node| unsafe {
            let ret = (&node.as_ref().key, &node.as_ref().val);
            self.next = node.as_ref().next;
            ret
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, K, V> DoubleEndedIterator for Iter<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.prev.map(|node| unsafe {
            let ret = (&node.as_ref().key, &node.as_ref().val);
            self.prev = node.as_ref().prev;
            ret
        })
    }
}

impl<'a, K, V> ExactSizeIterator for Iter<'a, K, V> {}

impl<'a, K, V> IntoIterator for &'a mut LRUCache<K, V> {
    type Item = (&'a K, &'a mut V);
    type IntoIter = IterMut<'a, K, V>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut {
            next: self.list.head,
            prev: self.list.tail,
            len: self.len(),
            _marker: PhantomData,
        }
    }
}

pub struct IterMut<'a, K, V> {
    next: Option<NonNull<Node<K, V>>>,
    prev: Option<NonNull<Node<K, V>>>,
    len: usize,
    _marker: PhantomData<&'a mut Node<K, V>>,
}

impl<'a, K, V> Iterator for IterMut<'a, K, V> {
    type Item = (&'a K, &'a mut V);

    fn next(&mut self) -> Option<Self::Item> {
        self.next.map(|mut node| unsafe {
            let ret = (&node.as_ref().key, &mut node.as_mut().val);
            self.next = node.as_mut().next;
            ret
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len, Some(self.len))
    }
}

impl<'a, K, V> DoubleEndedIterator for IterMut<'a, K, V> {
    fn next_back(&mut self) -> Option<Self::Item> {
        self.prev.map(|mut node| unsafe {
            let ret = (&node.as_ref().key, &mut node.as_mut().val);
            self.prev = node.as_mut().prev;
            ret
        })
    }
}

impl<'a, K, V> ExactSizeIterator for IterMut<'a, K, V> {}

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

    #[test]
    fn clone_creates_lru_with_same_recency_ordering() {
        let mut lru_cache = LRUCache::with_capacity(NonZeroUsize::new(2).unwrap());
        lru_cache.put('a', 'a');
        lru_cache.put('z', 'z');
        lru_cache.put('i', 'j');
        let duplicate = lru_cache.clone();
        for ((k1, v1), (k2, v2)) in lru_cache.iter().zip(duplicate.iter()) {
            assert_eq!(k1, k2);
            assert_eq!(v1, v2);
        }
    }

    #[test]
    fn foward_backwards_iters_work() {
        let mut lru_cache = LRUCache::with_capacity(NonZeroUsize::new(2).unwrap());
        lru_cache.put('y', 'x');
        lru_cache.put('a', 'z');
        lru_cache.put('1', '0');
        let mut iter = lru_cache.iter();
        assert_eq!(iter.next(), Some((&'1', &'0')));
        assert_eq!(iter.next_back(), Some((&'a', &'z')));
        assert_eq!(iter.next(), Some((&'a', &'z')));
        assert_eq!(iter.next_back(), Some((&'1', &'0')));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next_back(), None);
    }
}
