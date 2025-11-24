use core::f64;
use std::collections::HashSet;

use numpy::ndarray::{ArrayView1, ArrayView2, ArrayViewMut1, s};

use crate::{binary_heap::BinaryHeap, kd_tree::KDTree};

//Compute the square of the distances to the diagonal
fn projection_squared_distances(diagram: &ArrayView2<f64>) -> Vec<f64> {
    assert_eq!(diagram.shape()[1], 2);

    diagram
        .outer_iter()
        .map(|point| unsafe {
            let d = point.uget(0) - point.uget(1);
            0.5 * (d * d)
        })
        .collect()
}

pub fn distance_squared_bound(diagram: &ArrayView2<f64>) -> f64 {
    assert_eq!(diagram.shape()[1], 2);

    let mut min_p = f64::INFINITY;
    let mut max_p = f64::NEG_INFINITY;
    let mut max_o = f64::NEG_INFINITY;

    for point in diagram.outer_iter() {
        let x = unsafe { *point.uget(0) };
        let y = unsafe { *point.uget(1) };
        let p = f64::consts::FRAC_1_SQRT_2 * (x + y);
        let o = f64::consts::FRAC_1_SQRT_2 * (y - x);
        min_p = min_p.min(p);
        max_p = max_p.max(p);
        max_o = max_o.max(o);
    }

    let dp = max_p - min_p;
    dp * dp + max_o * max_o
}

pub fn max_cost_upper_bound(x: &ArrayView2<f64>, y: &ArrayView2<f64>) -> f64 {
    //Crude upper bound on max{ ||b - a||^2, a in x, b in y }, that is also
    //guaranteed to be larger than the distance squared of all the points
    //to their projection on the diagonal.
    //Used to initialize the auction algorithm
    f64::max(
        distance_squared_bound(x),
        distance_squared_bound(y)
    )
}

pub fn d2_from_matching(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    off_diag: &ArrayView1<u32>,
    diag: &ArrayView1<u32>,
    x_diag_distances: &[f64],
    y_diag_distances: &[f64],
) -> f64 {
    let mut output = 0.0;
    for (i, &j) in off_diag.iter().enumerate() {
        if j == u32::MAX {
            output += x_diag_distances[i];
        } else {
            let dx = x[(i, 0)] - y[(j as usize, 0)];
            let dy = x[(i, 1)] - y[(j as usize, 1)];
            output += dx * dx + dy * dy;
        }
    }
    for (i, &j) in diag.iter().enumerate() {
        if j == u32::MAX {
            output += y_diag_distances[i];
        }
    }

    output
}

const NO_BIDDER: u32 = u32::MAX - 1;

/**
 * Performs an auction round.
 * This function is used in auction_algorithm, but also needs
 * to be exposed for the barycenter algorithm.
 */
pub fn auction_round(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    off_diag: &mut ArrayViewMut1<u32>,
    diag: &mut ArrayViewMut1<u32>,
    off_diag_objects: &mut KDTree,
    diag_objects: &mut BinaryHeap,
    x_diag_distances: &[f64],
    y_diag_distances: &[f64],
    epsilon: f64,
) {
    let x_size = x.shape()[0];
    let y_size = y.shape()[0];

    let mut unassigned_off_diag: HashSet<u32> = HashSet::new(); //Unassigned off-diagonal bidders
    let mut unassigned_diag: HashSet<u32> = HashSet::new(); //Unassigned diagonal bidders

    unassigned_off_diag.extend(0..(x_size as u32));
    unassigned_diag.extend(0..(y_size as u32));

    let mut reverse_off_diag = vec![NO_BIDDER; y_size]; //Buyers of off-diagonal objects
    let mut reverse_diag = vec![NO_BIDDER; x_size]; //Buyers of diagonal objects
    //These contain exactly what off_diag and diag would contain if x and y were swapped

    'round: loop {
        let mut stop = true;

        //Process an off-diagonal bidder
        if let Some(&off_diag_bidder) = unassigned_off_diag.iter().next() {
            stop = false;

            let point = x.slice(s![off_diag_bidder as usize, ..]);
            //Query off-diagonal objects
            let ((mut best_cost, mut best_object), (mut second_best_cost, _)) =
                off_diag_objects.query(&point);

            //Check the diagonal object
            let diag_cost = x_diag_distances[off_diag_bidder as usize]
                + diag_objects.get_price(off_diag_bidder);
            if diag_cost < best_cost {
                second_best_cost = best_cost;
                best_cost = diag_cost;
                best_object = u32::MAX;
            } else if diag_cost < second_best_cost {
                second_best_cost = diag_cost;
            }

            //Change object bidder
            if best_object == u32::MAX {
                //Diagonal object
                //The old bidder (if there is one) has to be diagonal
                let r = &mut reverse_diag[off_diag_bidder as usize];
                assert_ne!(*r, u32::MAX); //Can't be us already
                if *r != NO_BIDDER {
                    unassigned_diag.insert(*r);
                }

                *r = u32::MAX;
                off_diag[off_diag_bidder as usize] = u32::MAX;
            } else {
                //Off-diagonal object
                //The old bidder can be either the projection or an off-diagonal bidder
                let r = &mut reverse_off_diag[best_object as usize];
                if *r != NO_BIDDER {
                    if *r == u32::MAX {
                        //Old bidder is the projection
                        unassigned_diag.insert(best_object);
                    } else {
                        unassigned_off_diag.insert(*r);
                    }
                }

                *r = off_diag_bidder;
                off_diag[off_diag_bidder as usize] = best_object;
            }
            unassigned_off_diag.remove(&off_diag_bidder);

            //Increase price
            let amount = second_best_cost - best_cost + epsilon;
            if best_object == u32::MAX {
                //Diagonal object
                diag_objects.increase_price(off_diag_bidder, amount);
            } else {
                //Off-diagonal object
                off_diag_objects.increase_price(best_object, amount);
            }
        }

        //Process a diagonal bidder
        if let Some(&diag_bidder) = unassigned_diag.iter().next() {
            stop = false;

            //Query diagonal objects
            let ((mut best_cost, mut best_object), (mut second_best_cost, _)) =
                diag_objects.query();

            //Check the off-diagonal object
            let off_diag_cost =
                y_diag_distances[diag_bidder as usize] + off_diag_objects.get_price(diag_bidder);
            if off_diag_cost < best_cost {
                second_best_cost = best_cost;
                best_cost = off_diag_cost;
                best_object = u32::MAX;
            }

            //Change object bidder
            if best_object == u32::MAX {
                //Off-diagonal object
                //The old bidder (if there is one) has to be off-diagonal
                let r = &mut reverse_off_diag[diag_bidder as usize];
                assert_ne!(*r, u32::MAX);
                if *r != NO_BIDDER {
                    unassigned_off_diag.insert(*r);
                }

                *r = u32::MAX;
                diag[diag_bidder as usize] = u32::MAX;
            } else {
                //Diagonal object
                //The old bidder can be either the unprojection or a diagonal bidder
                let r = &mut reverse_diag[best_object as usize];
                if *r != NO_BIDDER {
                    if *r == u32::MAX {
                        //Old bidder is the unprojection
                        unassigned_off_diag.insert(best_object);
                    } else {
                        unassigned_diag.insert(*r);
                    }
                }

                *r = diag_bidder;
                diag[diag_bidder as usize] = best_object;
            }
            unassigned_diag.remove(&diag_bidder);

            //Increase price
            let amount = second_best_cost - best_cost + epsilon;
            if best_object == u32::MAX {
                //Off-diagonal object
                off_diag_objects.increase_price(diag_bidder, amount);
            } else {
                //Diagonal object
                diag_objects.increase_price(best_object, amount);
            }
        }

        if stop {
            break 'round;
        }
    }
}

/**
 * Performs an auction, recomputing the k-d tree and the binary heap
 * from scratch. This is necessary if y is updated between rounds
 */
pub fn auction_round_recompute_objects(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    off_diag: &mut ArrayViewMut1<u32>,
    diag: &mut ArrayViewMut1<u32>,
    off_diag_prices: &mut [f64],
    diag_prices: &mut [f64],
    epsilon: f64,
) -> f64 {
    let x_size = x.shape()[0];
    let y_size = y.shape()[0];

    assert_eq!(off_diag_prices.len(), y_size);
    assert_eq!(diag_prices.len(), x_size);

    let x_diag_distances = projection_squared_distances(x);
    let y_diag_distances = projection_squared_distances(y);

    let mut off_diag_objects = KDTree::construct(y);
    let mut diag_objects = BinaryHeap::new(x_size);

    //Load prices
    for (i, &price) in off_diag_prices.iter().enumerate() {
        off_diag_objects.increase_price(i as u32, price);
    }
    for (i, &price) in diag_prices.iter().enumerate() {
        diag_objects.increase_price(i as u32, price);
    }

    //Perform the auction round
    auction_round(x, y, off_diag, diag, &mut off_diag_objects, &mut diag_objects, &x_diag_distances, &y_diag_distances, epsilon);

    //Store prices
    for (i, price) in off_diag_prices.iter_mut().enumerate() {
        *price = off_diag_objects.get_price(i as u32);
    }
    for (i, price) in diag_prices.iter_mut().enumerate() {
        *price = diag_objects.get_price(i as u32);
    }

    d2_from_matching(&x, &y, &off_diag.view(), &diag.view(), &x_diag_distances, &y_diag_distances)
}

/**
 * If o2 is the actual optimal cost (squared Wasserstein L^2 distance), the returned
 * value d2 satisfies o2 <= d2 <= o2(1 + delta)
 */
pub fn auction_algorithm(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    off_diag: &mut ArrayViewMut1<u32>,
    diag: &mut ArrayViewMut1<u32>,
    delta: f64,
) -> f64 {
    let x_size = x.shape()[0];
    let y_size = y.shape()[0];

    let n = usize::max(x_size, y_size) as f64;

    let x_diag_distances = projection_squared_distances(x);
    let y_diag_distances = projection_squared_distances(y);

    let mut off_diag_objects = KDTree::construct(y);
    let mut diag_objects = BinaryHeap::new(x_size);

    let mut d2 = 0.0;
    let mut epsilon = 1.25 * max_cost_upper_bound(x, y);

    while d2 > (1.0 + delta) * (d2 - n * epsilon) {
        epsilon *= 0.2;

        auction_round(
            x,
            y,
            off_diag,
            diag,
            &mut off_diag_objects,
            &mut diag_objects,
            &x_diag_distances,
            &y_diag_distances,
            epsilon,
        );

        //Recompute d2
        d2 = d2_from_matching(
            x,
            y,
            &off_diag.view(),
            &diag.view(),
            &x_diag_distances,
            &y_diag_distances,
        );
    }

    d2
}
