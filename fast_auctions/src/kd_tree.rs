use core::f64;
use std::cmp::Ordering;

use numpy::ndarray::{ArrayView1, ArrayView2};

/**
 * Stores the off-diagonal objects and their corresponding prices
 */
pub struct KDTree {
    leaves: Vec<Leaf>,
    internal_nodes: Vec<InternalNode>, //Of length leaves.len() - 1
}

impl KDTree {
    /**
     * Constructs the k-d tree from the given points, initializing all the prices to zero
     */
    pub fn construct(points: &ArrayView2<f64>) -> Self {
        assert!(points.shape()[0] - 1 <= u32::MAX as usize);
        assert!(points.shape()[0] >= 2);

        //Allocate and initialize the leaves
        assert_eq!(points.shape()[1], 2);
        let mut leaves = points
            .outer_iter()
            .map(|point| unsafe {
                Leaf {
                    point: Point {
                        x: *point.uget(0),
                        y: *point.uget(1),
                    },
                    price: 0.0,
                    parent: None, //Temporary
                }
            })
            .collect::<Vec<_>>();

        //Construct the internal nodes recursively
        let mut indices = (0..leaves.len()).collect::<Vec<_>>();

        let mut internal_nodes = Vec::new();
        internal_nodes.reserve_exact(leaves.len() - 1);

        let mut stack: Vec<(&mut [usize], u32, Option<(usize, bool)>)> = Vec::new();
        stack.push((&mut indices, 0u32, None)); //(indices, coordinate index, parent)
        while let Some((indices, coord, parent)) = stack.pop() {
            let child = if indices.len() == 1 {
                let index = indices[0];
                leaves[index].parent = parent;
                Child::Leaf(index)
            } else {
                let middle = indices.len() / 2;
                indices.select_nth_unstable_by(middle, |&i_a, &i_b| unsafe {
                    Point::compare(
                        coord,
                        &leaves.get_unchecked(i_a).point,
                        &leaves.get_unchecked(i_b).point,
                    )
                });
                let internal_node_index = internal_nodes.len();
                internal_nodes.push(InternalNode {
                    min: Point {
                        x: f64::INFINITY,
                        y: f64::INFINITY,
                    },
                    max: Point {
                        x: f64::NEG_INFINITY,
                        y: f64::NEG_INFINITY,
                    },
                    min_price: 0.0,
                    left: Child::None,
                    right: Child::None,
                    parent: parent,
                });
                let (left, right) = indices.split_at_mut(middle);
                let new_coord = (coord + 1) & 1;
                stack.push((left, new_coord, Some((internal_node_index, false))));
                stack.push((right, new_coord, Some((internal_node_index, true))));
                Child::Internal(internal_node_index)
            };
            if let Some((parent_index, parent_side)) = parent {
                if !parent_side {
                    internal_nodes[parent_index].left = child;
                } else {
                    internal_nodes[parent_index].right = child;
                }
            }
        }

        //Update the internal bounds
        for i in (0..internal_nodes.len()).rev() {
            let node = &internal_nodes[i];
            let children = [node.left, node.right];
            for child in children {
                let (child_min, child_max) = match child {
                    Child::Internal(index) => {
                        let node = &internal_nodes[index];
                        (node.min, node.max)
                    }
                    Child::Leaf(index) => {
                        let leaf = &leaves[index];
                        (leaf.point, leaf.point)
                    }
                    Child::None => panic!("There should be no None child remaining! "),
                };

                let node = &mut internal_nodes[i];
                node.min.set_min_with(&child_min);
                node.max.set_max_with(&child_max);
            }
        }

        Self {
            leaves,
            internal_nodes,
        }
    }

    /**
     * bidder is a 1D array of length 2 containing the position of an off-diagonal bidder.
     * The function returns the indices and costs of the first and second best objects for the bidder.
     */
    pub fn query(&self, bidder: &ArrayView1<f64>) -> ((f64, u32), (f64, u32)) {
        let bidder = Point {
            x: bidder[0],
            y: bidder[1],
        };

        let mut best = (f64::INFINITY, u32::MAX);
        let mut second_best = (f64::INFINITY, u32::MAX);

        self.query_rec(&bidder, &Child::Internal(0), &mut best, &mut second_best);

        assert_ne!(best.0, f64::INFINITY);
        assert_ne!(second_best.0, f64::INFINITY);

        (best, second_best)
    }

    fn query_rec(
        &self,
        bidder: &Point,
        child: &Child,
        best: &mut (f64, u32),
        second_best: &mut (f64, u32),
    ) {
        match child {
            Child::Internal(index) => {
                let node = &self.internal_nodes[*index];
                if node.min_cost(bidder) >= second_best.0 {
                    //No need to explore
                    return;
                }
                //If there is no second best, or if it can be improved, we explore
                self.query_rec(bidder, &node.left, best, second_best);
                self.query_rec(bidder, &node.right, best, second_best);
            }
            Child::Leaf(index) => {
                let leaf = &self.leaves[*index];
                let cost = leaf.cost(bidder);
                if cost < best.0 {
                    *second_best = *best;
                    *best = (cost, *index as u32);
                } else if cost < second_best.0 {
                    *second_best = (cost, *index as u32);
                }
            }
            Child::None => panic!(),
        }
    }

    /**
     * Returns the price of an object
     */
    pub fn get_price(&self, object: u32) -> f64 {
        self.leaves[object as usize].price
    }

    /**
     * Increases the price of an object at a given index.
     * amount MUST be nonnegative.
     */
    pub fn increase_price(&mut self, object: u32, amount: f64) {
        let leaf = &mut self.leaves[object as usize];
        let mut old_price = leaf.price;
        let mut new_price = old_price + amount;
        leaf.price = new_price;
        let mut parent = leaf.parent;
        while let Some((parent_index, parent_side)) = parent {
            let other_child = if !parent_side {
                self.internal_nodes[parent_index].right
            } else {
                self.internal_nodes[parent_index].left
            };
            let other_price = match other_child {
                Child::Internal(index) => self.internal_nodes[index].min_price,
                Child::Leaf(index) => self.leaves[index].price,
                Child::None => panic!(),
            };
            if old_price < other_price && new_price >= other_price {
                let node = &mut self.internal_nodes[parent_index];
                old_price = node.min_price;
                new_price = other_price;
                node.min_price = new_price;
                parent = node.parent;
            } else {
                break; //Nothing has changed at this level
            }
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn compare(coord: u32, a: &Self, b: &Self) -> Ordering {
        if coord == 0 {
            a.x.total_cmp(&b.x)
        } else {
            a.y.total_cmp(&b.y)
        }
    }

    fn set_min_with(&mut self, other: &Self) {
        self.x = self.x.min(other.x);
        self.y = self.y.min(other.y);
    }

    fn set_max_with(&mut self, other: &Self) {
        self.x = self.x.max(other.x);
        self.y = self.y.max(other.y);
    }

    fn distance2(a: &Self, b: &Self) -> f64 {
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        dx * dx + dy * dy
    }
}

#[derive(Copy, Clone, Debug)]
enum Child {
    Internal(usize),
    Leaf(usize),
    None, //Only during construction
}

#[derive(Debug)]
struct Leaf {
    point: Point,
    price: f64,
    parent: Option<(usize, bool)>,
}

impl Leaf {
    fn cost(&self, bidder: &Point) -> f64 {
        self.price + Point::distance2(&self.point, &bidder)
    }
}

#[derive(Debug)]
struct InternalNode {
    min: Point,
    max: Point,
    min_price: f64,
    left: Child,
    right: Child,
    parent: Option<(usize, bool)>,
}

impl InternalNode {
    fn min_cost(&self, bidder: &Point) -> f64 {
        let mut projection = *bidder;
        projection.set_max_with(&self.min);
        projection.set_min_with(&self.max);
        self.min_price + Point::distance2(&projection, bidder)
    }
}

//Tests for the KD tree implementation
#[cfg(test)]
mod tests {
    use numpy::ndarray::Array2;

    use super::*;

    #[test]
    fn four_points() {
        let points: Array2<f64> =
            Array2::from_shape_vec((4, 2), vec![-1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0])
                .unwrap();

        let tree = KDTree::construct(&points.view());

        let mut test_points = vec![];
        let mut expected_class: Vec<u32> = vec![];
        for i in 0..10 {
            for j in 0..10 {
                let x = (i as f64) / 9.0 - 0.5;
                let y = (j as f64) / 9.0 - 0.5;
                test_points.push(x);
                test_points.push(y);
                expected_class.push(match (x > 0.0, y > 0.0) {
                    (false, false) => 0,
                    (false, true) => 1,
                    (true, false) => 2,
                    (true, true) => 3,
                });
            }
        }

        let test_points = Array2::from_shape_vec((100, 2), test_points).unwrap();

        for (point, class) in test_points.outer_iter().zip(expected_class) {
            assert_eq!(tree.query(&point).0.1, class);
        }
    }

    #[test]
    #[rustfmt::skip]
    fn many_points_without_prices() {
        let n = 20;

        let points: Array2<f64> = Array2::from_shape_vec((n, 2), vec![
            -0.55063425775095, -0.3241706691014574, 
            -0.9378301627601247, 0.964037188111158, 
            -0.5310359415398507, 0.29735940182731957, 
            0.5439569647590754, -0.8099115159346006, 
            -0.33958746021850494, -0.11329948934504719, 
            0.04850946988038296, -0.5338554503049837, 
            0.42531542300534864, -0.29451735392239065, 
            0.2544119569865606, -0.786806814076479, 
            -0.7721434833069085, 0.8226720494168642, 
            0.2842255291450142, 0.843474500671374, 
            -0.2971856237644701, 0.6183349316002436, 
            0.7872622485017418, 0.34501646128469643, 
            -0.2552575438306308, -0.29320987832795886, 
            -0.22390313758162317, 0.8837490277766415, 
            -0.06727048631836752, 0.6814366526446636, 
            0.2691553954061976, -0.11685740615578566, 
            0.2257117917156548, 0.35653764177427116, 
            0.7573884837807336, 0.05859419698984625, 
            -0.8133439997653662, 0.6648433040260022, 
            0.3165265605017993, 0.9237862827614367,
        ]).unwrap();

        let cost_matrix = Array2::from_shape_fn([n, n], |(i, j)| {
            let dx = points[(i, 0)] - points[(j, 0)];
            let dy = points[(i, 1)] - points[(j, 1)];
            dx * dx + dy * dy
        });

        let mut best = vec![];
        let mut second_best = vec![];
        for i in 0..n {
            let mut indices: Vec<usize> = (0..n).collect::<Vec<_>>();
            indices.sort_by(|&j0, &j1| cost_matrix[(i, j0)].total_cmp(&cost_matrix[(i, j1)]));
            best.push(indices[0] as u32);
            second_best.push(indices[1] as u32);
        }

        let tree = KDTree::construct(&points.view());

        for (bidder, (best, second_best)) in
            points.outer_iter().zip(best.iter().zip(second_best.iter()))
        {
            let ((_, kd_best), (_, kd_second_best)) = tree.query(&bidder);

            assert_eq!(*best, kd_best);
            assert_eq!(*second_best, kd_second_best);
        }
    }

    #[test]
    #[rustfmt::skip]
    fn many_points_with_prices() {
        let n = 20;

        let points: Array2<f64> = Array2::from_shape_vec((n, 2), vec![
            -0.55063425775095, -0.3241706691014574, 
            -0.9378301627601247, 0.964037188111158, 
            -0.5310359415398507, 0.29735940182731957, 
            0.5439569647590754, -0.8099115159346006, 
            -0.33958746021850494, -0.11329948934504719, 
            0.04850946988038296, -0.5338554503049837, 
            0.42531542300534864, -0.29451735392239065, 
            0.2544119569865606, -0.786806814076479, 
            -0.7721434833069085, 0.8226720494168642, 
            0.2842255291450142, 0.843474500671374, 
            -0.2971856237644701, 0.6183349316002436, 
            0.7872622485017418, 0.34501646128469643, 
            -0.2552575438306308, -0.29320987832795886, 
            -0.22390313758162317, 0.8837490277766415, 
            -0.06727048631836752, 0.6814366526446636, 
            0.2691553954061976, -0.11685740615578566, 
            0.2257117917156548, 0.35653764177427116, 
            0.7573884837807336, 0.05859419698984625, 
            -0.8133439997653662, 0.6648433040260022, 
            0.3165265605017993, 0.9237862827614367,
        ]).unwrap();

        let prices = vec![
            0.6709324851974211,
            0.2555538873826516,
            0.6771760194809718,
            0.8760234132949591,
            0.5810861574773328,
            0.30378871845760813,
            0.5199982794375773,
            0.29343551960944125,
            0.052540102766443386,
            0.3550239992622709,
            0.1845027668042475,
            0.2406402001575796,
            0.14138265778461234,
            0.992633701935369,
            0.7297763231159256,
            0.8269728841810199,
            0.0130163898882103,
            0.11583906609859584,
            0.7288831643740198,
            0.4550138518030491,
        ];

        let cost_matrix = Array2::from_shape_fn([n, n], |(i, j)| {
            let dx = points[(i, 0)] - points[(j, 0)];
            let dy = points[(i, 1)] - points[(j, 1)];
            dx * dx + dy * dy + prices[j]
        });

        let mut best = vec![];
        let mut second_best = vec![];
        for i in 0..n {
            let mut indices: Vec<usize> = (0..n).collect::<Vec<_>>();
            indices.sort_by(|&j0, &j1| cost_matrix[(i, j0)].total_cmp(&cost_matrix[(i, j1)]));
            best.push(indices[0] as u32);
            second_best.push(indices[1] as u32);
        }

        let mut tree = KDTree::construct(&points.view());
        for (i, price) in prices.iter().enumerate() {
            tree.increase_price(i as u32, *price);
        }

        for (bidder, (best, second_best)) in
            points.outer_iter().zip(best.iter().zip(second_best.iter()))
        {
            let ((_, kd_best), (_, kd_second_best)) = tree.query(&bidder);

            assert_eq!(*best, kd_best);
            assert_eq!(*second_best, kd_second_best);
        }
    }
}
