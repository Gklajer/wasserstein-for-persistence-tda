use std::{
    mem,
    time::{Duration, Instant},
};

use itertools::izip;
use numpy::ndarray::{self, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut2, Axis, s};
use rand::{Rng, SeedableRng, rngs::SmallRng};

use crate::auction;

/**
 * Performs a binary search to determine the largest slice 0..n
 * such that all the persistences are larger than rho
 */
fn threshold_index(atom: &ArrayView2<f64>, rho: f64) -> usize {
    let mut low = 0;
    let mut high = atom.shape()[0];

    while low < high {
        let mid = (low + high) / 2;

        let persistence = atom[(mid, 1)] - atom[(mid, 0)];

        if persistence >= rho {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    low
}

/**
 * Finds the minimum value that can be assigned to rho for a specific input diagram.
 * The final value used for persistence scaling will be the maximum of those values
 * for all the dictionary atoms.
 */
fn min_rho(atom: &ArrayView2<f64>, previous_length: usize, epsilon: f64) -> f64 {
    const MAX_RATIO: f64 = 1.1;
    const TAU: f64 = 4.0;

    let max_new_length = (previous_length as f64) as usize;
    let max_new_length = usize::min(atom.shape()[0], max_new_length);
    let index = max_new_length.saturating_sub(1);

    let rho = atom[(index, 1)] - atom[(index, 0)];
    let rho = f64::max(rho, f64::sqrt(TAU * epsilon));

    rho
}

/**
 * Computes the initial value for the global epsilon
 */
fn intial_epsilon(dictionary: &[ArrayView2<f64>], atom_lengths: &[usize]) -> f64 {
    dictionary
        .iter()
        .zip(atom_lengths.iter())
        .map(|(atom, length)| auction::distance_squared_bound(&atom.slice(s![..*length, ..])))
        .max_by(f64::total_cmp)
        .unwrap()
}

/**
 * Updates the barycenter
 */
fn update_barycenter<'v1, 'v2>(
    dictionary: &[ArrayView2<f64>],
    lengths: &[usize],
    lambda: &ArrayView1<f64>,
    off_diag: impl Iterator<Item = ArrayView1<'v1, u32>>,
    diag: impl Iterator<Item = ArrayView1<'v2, u32>>,
    input: &ArrayView2<f64>,
    output: &mut ArrayViewMut2<f64>,
) {
    output.fill(0.0);
    for (atom, length, lambda, off_diag, diag) in izip!(dictionary, lengths, lambda, off_diag, diag)
    {
        for (point, index) in atom
            .slice(s![..*length, ..])
            .outer_iter()
            .zip(off_diag.iter())
        {
            output
                .slice_mut(s![*index as usize, ..])
                .scaled_add(*lambda, &point);
        }
        for (mut point_out, point_in, i) in izip!(output.outer_iter_mut(), input.outer_iter(), diag)
        {
            if *i != u32::MAX {
                continue;
            }
            //The point is paired with its projection on the diagonal
            let p = 0.5 * (point_in[0] + point_in[1]);
            point_out[0] += *lambda * p;
            point_out[1] += *lambda * p;
        }
    }
}

/**
 * Extends the barycenter with the new points in the dictionary
 */
fn extend_barycenter(
    dictionary: &[ArrayView2<f64>],
    old_lengths: &[usize],
    new_lengths: &[usize],
    output: &mut Array2<f64>,
) {
    //Compute the list of all added points in the atoms
    let added = izip!(dictionary, old_lengths, new_lengths)
        .map(|(atom, old_length, new_length)| atom.slice(s![*old_length..*new_length, ..]))
        .collect::<Vec<_>>();
    let added = ndarray::concatenate(Axis(0), &added).unwrap();

    let old_output_length = output.shape()[0];
    let added_length = added.shape()[0];
    let new_output_length = old_output_length + added_length;

    //Add random ponits from this list to the barycenter
    let mut new_output = Array2::zeros((new_output_length, 2));
    new_output
        .slice_mut(s![..old_output_length, ..])
        .assign(output);
    let mut output_added = new_output.slice_mut(s![old_output_length..new_output_length, ..]);
    let mut rng = SmallRng::from_rng(&mut rand::rng());

    for mut point_out in output_added.outer_iter_mut() {
        let i = rng.random_range(0..added_length);
        let point_in = added.slice(s![i, ..]);
        point_out.assign(&point_in);
    }

    *output = new_output;
}

//For the matching, the dictionary atoms are the sets
//of bidders and the barycenter the set of objects
pub struct BarycenterOutput {
    diagram: Array2<f64>,
    diag: Vec<Array1<u32>>,
    off_diag: Vec<Array1<u32>>,
}

/**
 * We assume that the input dictionary atoms are sorted by persistence
 * from the most persistent pairs to the least
 */
pub fn wasserstein_barycenter(
    dictionary: &[ArrayView2<f64>],
    lambda: &ArrayView1<f64>,
    max_duration: Duration,
) -> BarycenterOutput {
    let start = Instant::now();

    let mut rho = 0.5
        * dictionary
            .iter()
            .map(|atom| {
                let most_persistent = atom.outer_iter().next().unwrap();
                most_persistent[1] - most_persistent[0]
            })
            .max_by(f64::total_cmp)
            .unwrap();

    let mut atom_lengths = dictionary
        .iter()
        .map(|atom| threshold_index(atom, rho))
        .collect::<Vec<usize>>(); //For persistence scaling

    //All those variables are reallocated during persistence scaling
    let mut output_length = atom_lengths.iter().sum();
    let mut output = Array2::zeros((output_length, 2));
    let mut temp = Array2::zeros((output_length, 2));

    let mut off_diag = atom_lengths
        .iter()
        .map(|&length| Array1::zeros(length))
        .collect::<Vec<_>>();
    let mut diag = atom_lengths
        .iter()
        .map(|_| Array1::zeros(output_length))
        .collect::<Vec<_>>();

    let mut off_diag_prices = atom_lengths
        .iter()
        .map(|_| vec![0.0; output_length])
        .collect::<Vec<_>>();
    let mut diag_prices = atom_lengths
        .iter()
        .map(|&length| vec![0.0; length])
        .collect::<Vec<_>>();

    let mut epsilon = intial_epsilon(dictionary, &atom_lengths);

    let mut last_energy: Option<f64> = None;

    'l: loop {
        //Auction round
        let mut energy = 0.0;
        for (atom_length, atom, off_diag, diag, off_diag_prices, diag_prices, lambda) in itertools::izip!(
            &atom_lengths,
            dictionary,
            &mut *off_diag,
            &mut *diag,
            &mut off_diag_prices,
            &mut diag_prices,
            lambda,
        ) {
            let x = atom.slice(s![..*atom_length, ..]);
            let y = output.view();
            let d2 = auction::auction_round_recompute_objects(
                &x,
                &y,
                &mut off_diag.view_mut(),
                &mut diag.view_mut(),
                off_diag_prices,
                diag_prices,
                epsilon,
            );
            energy += *lambda * d2;
        }

        if let Some(last_energy) = last_energy {
            if energy >= last_energy && epsilon < 1.0e-5 {
                break 'l;
            }
        }

        //Barycenter update
        update_barycenter(
            dictionary,
            &atom_lengths,
            lambda,
            off_diag.iter().map(|a| a.view()),
            diag.iter().map(|a| a.view()),
            &output.view(),
            &mut temp.view_mut(),
        );
        mem::swap(&mut output, &mut temp);

        //Epsilon scaling
        epsilon /= 5.0;

        //Persistence scaling
        if start.elapsed() < max_duration / 10 {
            //Update rho
            rho = dictionary
                .iter()
                .zip(atom_lengths.iter())
                .map(|(atom, length)| min_rho(atom, *length, epsilon))
                .max_by(f64::total_cmp)
                .unwrap();

            let new_atom_lengths = dictionary
                .iter()
                .map(|atom| threshold_index(atom, rho))
                .collect::<Vec<usize>>();

            //Extend the output
            extend_barycenter(dictionary, &atom_lengths, &new_atom_lengths, &mut output);
            output_length = output.shape()[0];
            temp = Array2::zeros((output_length, 2));

            //Reallocate off_diag and diag for the new atom lengths
            off_diag = new_atom_lengths
                .iter()
                .map(|&length| Array1::zeros(length))
                .collect::<Vec<_>>();
            diag = new_atom_lengths
                .iter()
                .map(|_| Array1::zeros(output_length))
                .collect::<Vec<_>>();

            //Extend the prices with the minimum price
            for (off_diag_prices, diag_prices, new_atom_length) in izip!(&mut off_diag_prices, &mut diag_prices, &new_atom_lengths) {
                let minimum_price = off_diag_prices.iter()
                        .chain(diag_prices.iter())
                        .map(|c| *c)
                        .min_by(f64::total_cmp)
                        .unwrap();
                
                off_diag_prices.resize(output_length, minimum_price);
                diag_prices.resize(*new_atom_length, minimum_price);
            }

            atom_lengths = new_atom_lengths;
        }

        //Ran out of time
        if start.elapsed() >= max_duration {
            break 'l;
        }

        last_energy = Some(energy);
    }

    BarycenterOutput {
        diagram: output,
        diag,
        off_diag,
    }
}
