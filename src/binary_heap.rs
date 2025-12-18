use std::{cell::UnsafeCell, iter};

/**
 * A fast implementation of a fixed-size priority queue
 * Stores the prices of diagonal objects
 */
pub struct BinaryHeap {
    //Flat representation of the binary tree:
    //
    //              A
    //             / \
    //            B   C
    //           / \ /
    //           D E F
    //
    //is stored as [@, A, B, C, D, E, F], where @ is a dummy element
    //That way, the parent of a node at index i is at index i / 2
    //and the children are at indices i * 2 and i * 2 + 1
    locations: Vec<u32>, //Where is each object in the tree
    objects: Vec<UnsafeCell<(u32, f64)>>,
}

impl BinaryHeap {
    /**
     * Creates the binary heap. All the prices are initialized to zero.
     */
    pub fn new(object_count: usize) -> Self {
        assert!(object_count - 1 <= u32::MAX as usize);
        assert!(object_count >= 2);

        Self {
            locations: (0..object_count as u32).map(|object| object + 1).collect(),
            objects: iter::once(UnsafeCell::new((0, f64::NAN)))
                .chain((0..object_count as u32).map(|i| UnsafeCell::new((i, 0.0))))
                .collect(),
        }
    }

    /**
     * Returns the two objects of lowest price, as well as their
     * cost (there price)
     */
    pub fn query(&self) -> ((f64, u32), (f64, u32)) {
        let (best_object, best_price) = unsafe { *self.objects[1].get() };
        let (left_object, left_price) = unsafe { *self.objects[2].get() };
        if let Some(right) = self.objects.get(3) {
            let (right_object, right_price) = unsafe { *right.get() };
            if left_price < right_price {
                ((best_price, best_object), (left_price, left_object))
            } else {
                ((best_price, best_object), (right_price, right_object))
            }
        } else {
            ((best_price, best_object), (left_price, left_object))
        }
    }

    /**
     * Returns the price of an object
     */
    pub fn get_price(&self, object: u32) -> f64 {
        let location = self.locations[object as usize] as usize;
        unsafe { *self.objects[location].get() }.1
    }

    /**
     * Increases the price of an object.
     * amount MUST be nonnegative.
     */
    pub fn increase_price(&mut self, object: u32, amount: f64) {
        //Here we use UnsafeCells to be able to keep mutable references.
        //There can be no aliasing, as the three mutable references that are
        //kept simultaneously correspond to a node, its left child, and its right child
        let mut location = self.locations[object as usize] as usize;
        let mut current = unsafe { &mut *self.objects[location as usize].get() };
        let new_price = current.1 + amount;
        loop {
            //If the price has become higher than one of the children, the cheapest child
            //gets promoted and we continue recursively
            let left_location = location * 2;
            let Some(left_child) = self.objects.get(left_location) else {
                break; //No children
            };
            let left_child = unsafe { &mut *left_child.get() };
            let right_location = location * 2 + 1;
            if let Some(right_child) = self.objects.get(right_location) {
                let right_child = unsafe { &mut *right_child.get() };
                if left_child.1 < right_child.1 {
                    if new_price > left_child.1 {
                        *current = *left_child;
                        self.locations[left_child.0 as usize] = location as u32;
                        location = left_location;
                        current = left_child;
                    } else {
                        break;
                    }
                } else {
                    if new_price > right_child.1 {
                        *current = *right_child;
                        self.locations[right_child.0 as usize] = location as u32;
                        location = right_location;
                        current = right_child;
                    } else {
                        break;
                    }
                }
            } else {
                if new_price > left_child.1 {
                    *current = *left_child;
                    self.locations[left_child.0 as usize] = location as u32;
                    location = left_location;
                    current = left_child;
                }
                break; //No more children
            }
        }
        *current = (object, new_price);
        self.locations[object as usize] = location as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::BinaryHeap;

    #[test]
    fn price_increase() {
        //Test all possible orders of prices
        //and all possible orders to increase prices
        //and check that they all lead to a correct query
        for n in 2..=5 {
            let mut permutations: Vec<Vec<usize>> = vec![vec![]];
            for i in 0..n {
                permutations = permutations
                    .iter()
                    .flat_map(|permutation| {
                        (0..=permutation.len()).map(|j| {
                            let mut new_permutation = permutation.clone();
                            new_permutation.insert(j, i);
                            new_permutation
                        })
                    })
                    .collect();
            }

            for price_permutation in permutations.iter() {
                let prices: Vec<f64> = price_permutation.iter().map(|c| *c as f64).collect();
                let mut best = 0;
                let mut second_best = 0;
                for (index, i) in price_permutation.iter().enumerate() {
                    if *i == 0 {
                        best = index as u32;
                    } else if *i == 1 {
                        second_best = index as u32;
                    }
                }

                for increase_permutation in permutations.iter() {
                    let mut heap = BinaryHeap::new(n);

                    for i in increase_permutation.iter() {
                        heap.increase_price(*i as u32, prices[*i]);
                    }

                    let ((_, heap_best), (_, heap_second_best)) = heap.query();
                    assert_eq!(heap_best, best);
                    assert_eq!(heap_second_best, second_best);
                }
            }
        }
    }
}
