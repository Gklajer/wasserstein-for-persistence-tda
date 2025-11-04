use crate::barycenter::{self, compute_wasserstein_barycenter};
use crate::structs::{
    self, augment_diagram_set, augment_diagrams, compute_optimal_matching, project_onto_simplex,
    Matching, PersistenceDiagram, PersistencePair,
};

use core::{f64, net};
use std::collections::{HashMap, VecDeque};

/// Struct for recording the current state of the data
struct MatchingContext {
    phi_matchings: Vec<Matching>, // matchings between barycenter and atoms
    psi_matching: Matching,       // matching between barycenter and input
    barycenter: PersistenceDiagram,
}

impl MatchingContext {
    fn new(
        weights: &[f64],
        atoms: &[PersistenceDiagram],
        input_diagram: &PersistenceDiagram,
    ) -> Self {
        let barycenter = compute_wasserstein_barycenter(atoms, weights);

        let mut phi_matchings = Vec::with_capacity(atoms.len());
        for atom in atoms.iter() {
            let matching = compute_optimal_matching(&barycenter, atom);
            phi_matchings.push(matching);
        }

        let psi_matching = compute_optimal_matching(&barycenter, input_diagram);

        MatchingContext {
            phi_matchings,
            psi_matching,
            barycenter,
        }
    }
}

/// Computes the gradient of weight energy E_W
fn compute_weight_gradient(
    input_diagram: &PersistenceDiagram,
    atoms: &[PersistenceDiagram],
    ctx: &MatchingContext,
) -> Vec<f64> {
    let m = atoms.len();
    let k = ctx.barycenter.size();

    let mut gradient = vec![0.0; m];

    // For each point j in the barycenter
    for j in 0..k {
        let y_j = ctx.barycenter.pairs()[j];
        let x_psi_j = input_diagram.pairs()[*ctx.psi_matching.get(&j).unwrap()];

        // Build the matrix of matched atom points (one per atom)
        let mut atom_points = Vec::with_capacity(m);
        for i in 0..m {
            let matched_idx = ctx.phi_matchings[i].get(&j).unwrap();
            atom_points.push(atoms[i].pairs()[*matched_idx]);
        }

        // Compute the difference vector (a^{phi_i(j)}_i - x^{psi(j)})
        let mut diff_vectors = Vec::with_capacity(m);
        for i in 0..m {
            let a_phi_i_j = atom_points[i].to_point();
            let x_psi_j_point = x_psi_j.to_point();
            diff_vectors.push((a_phi_i_j.0 - x_psi_j_point.0, a_phi_i_j.1 - x_psi_j_point.1));
        }

        // Compute sum_i lambda_i * (a^{phi_i(j)}_i - x^{psi(j)})
        // This is y_j - x_psi_j by construction
        let y_j_point = y_j.to_point();
        let x_psi_j_point = x_psi_j.to_point();
        let weighted_sum = (y_j_point.0 - x_psi_j_point.0, y_j_point.1 - x_psi_j_point.1);

        // Gradient contribution from this point (Eq. 6)
        for i in 0..m {
            let dot_product =
                diff_vectors[i].0 * weighted_sum.0 + diff_vectors[i].1 * weighted_sum.1;
            gradient[i] += 2.0 * dot_product;
        }
    }

    gradient
}

/// Computes Lipschitz constant for weight gradient
fn compute_weight_lipschitz(
    input_diagram: &PersistenceDiagram,
    atoms: &[PersistenceDiagram],
    ctx: &MatchingContext,
) -> f64 {
    let k = ctx.barycenter.size();

    (0..k).fold(0.0, |sum_h_j_norm_sq, index| {
        let x_psi_j_point =
            input_diagram.pairs()[*ctx.psi_matching.get(&index).unwrap()].to_point();
        sum_h_j_norm_sq
            + ctx
                .phi_matchings
                .iter()
                .enumerate()
                .fold(0.0, |h_j_norm_sq, (i_idx, matching)| {
                    let matched_idx = matching.get(&index).unwrap();
                    let a_phi_i_j = atoms[i_idx].pairs()[*matched_idx].to_point();

                    let diff = (a_phi_i_j.0 - x_psi_j_point.0, a_phi_i_j.1 - x_psi_j_point.1);
                    h_j_norm_sq + diff.0 * diff.0 + diff.1 * diff.1
                })
    })
}

/// Performs one step of descent for weight optimization
pub fn optimize_weights_step(
    input_diagram: &PersistenceDiagram,
    atoms: &[PersistenceDiagram],
    weights: &[f64],
) -> Vec<f64> {
    let context = MatchingContext::new(weights, atoms, input_diagram);

    let gradient = compute_weight_gradient(input_diagram, atoms, &context);
    let lipschitz = compute_weight_lipschitz(input_diagram, atoms, &context);

    let rho = if lipschitz > 1e-10 {
        1.0 / lipschitz
    } else {
        0.1
    };

    let mut new_weights: Vec<f64> = weights
        .iter()
        .zip(gradient.iter())
        .map(|(&w, &g)| w - rho * g)
        .collect();

    new_weights = project_onto_simplex(&new_weights);

    new_weights
}

/// Computes gradient of atom energy e_A for point y_j
fn compute_atom_gradient(
    y_j_index: usize,
    input_diagram: &PersistenceDiagram,
    atoms: &[PersistenceDiagram],
    weights: &[f64],
    ctx: &MatchingContext,
) -> Vec<(f64, f64)> {
    let m = atoms.len();

    let x_psi_j_point =
        input_diagram.pairs()[*ctx.psi_matching.get(&y_j_index).unwrap()].to_point();

    let weighted_sum = ctx
        .phi_matchings
        .iter()
        .zip(atoms.iter())
        .zip(weights.iter())
        .fold((0.0, 0.0), |csum, ((matching, atom), lambda_i)| {
            let matched_idx = matching.get(&y_j_index).unwrap();
            let a_phi_i_j = atom.pairs()[*matched_idx].to_point();
            (
                csum.0 + lambda_i * (a_phi_i_j.0 - x_psi_j_point.0),
                csum.1 + lambda_i * (a_phi_i_j.1 - x_psi_j_point.1),
            )
        });

    let mut gradients = Vec::with_capacity(m);
    for &lambda_i in weights.iter() {
        gradients.push((
            2.0 * lambda_i * weighted_sum.0,
            2.0 * lambda_i * weighted_sum.1,
        ));
    }

    gradients
}

/// Projects a point to an admissible region
pub fn project_to_admissible_region(
    point: (f64, f64),
    scalar_min: f64,
    scalar_max: f64,
) -> (f64, f64) {
    let mut birth = point.0.max(scalar_min).min(scalar_max);
    let mut death = point.1.max(scalar_min).min(scalar_max);

    if death < birth {
        let mid = (birth + death) / 2.0;
        birth = mid;
        death = mid;
    }

    (birth, death)
}

/// Performs one step of gradient descent for atom optimization
pub fn optimize_atoms_step(
    input_diagram: &PersistenceDiagram,
    atoms: &[PersistenceDiagram],
    weights: &[f64],
    scalar_min: f64,
    scalar_max: f64,
) -> Vec<PersistenceDiagram> {
    let m = atoms.len();
    let context = MatchingContext::new(weights, atoms, input_diagram);

    let rho = 0.9 / (4.0 * m as f64);

    let mut new_atoms = Vec::with_capacity(m);
    for i in 0..m {
        let mut new_pairs = Vec::new();

        for j in 0..atoms[i].size() {
            let old_point = atoms[i].pairs()[j];

            let barycenter_idx = context.phi_matchings[i]
                .assignments()
                .iter()
                .find(|(_, &atom_idx)| atom_idx == j)
                .map(|(&bary_idx, _)| bary_idx);

            if let Some(bary_idx) = barycenter_idx {
                let gradients =
                    compute_atom_gradient(bary_idx, input_diagram, atoms, weights, &context);
                let grad = gradients[i];
                let old_point_coords = old_point.to_point();

                let new_point = (
                    old_point_coords.0 - rho * grad.0,
                    old_point_coords.1 - rho * grad.1,
                );

                let projected = project_to_admissible_region(new_point, scalar_min, scalar_max);

                new_pairs.push(PersistencePair::new(
                    projected.0,
                    projected.1,
                    old_point.birth_index,
                    old_point.death_index,
                ));
            } else {
                new_pairs.push(old_point);
            }
        }

        new_atoms.push(PersistenceDiagram::from_pairs(
            new_pairs,
            atoms[i].manifold_dimension(),
        ));
    }

    new_atoms
}

/// Computes the dictionary energy E_D (Eq. 2)
pub fn compute_dictionary_energy(
    input_diagrams: &[PersistenceDiagram],
    atoms: &[PersistenceDiagram],
    weights_set: &[Vec<f64>],
) -> f64 {
    let mut total_energy = 0.0;

    for (input, weights) in input_diagrams.iter().zip(weights_set.iter()) {
        let barycenter = compute_wasserstein_barycenter(atoms, weights);
        let matching = compute_optimal_matching(&barycenter, input);
        total_energy += matching.cost();
    }

    total_energy
}

/// Initialize dictionary using k-means++ style strategy (Section 4.1)
pub fn initialize_dictionary(
    input_diagrams: &[PersistenceDiagram],
    num_atoms: usize,
) -> Vec<PersistenceDiagram> {
    let n = input_diagrams.len();
    assert!(num_atoms > 0 && num_atoms <= n);

    // Precompute distance matrix
    let mut distance_matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let (aug1, aug2) = augment_diagrams(&input_diagrams[i], &input_diagrams[j]);
            let matching = compute_optimal_matching(&aug1, &aug2);
            let dist = matching.cost().sqrt(); // W distance (not squared)
            distance_matrix[i][j] = dist;
            distance_matrix[j][i] = dist;
        }
    }

    let mut selected_indices = Vec::with_capacity(num_atoms);

    // Select first atom: maximize sum of distances to all diagrams
    let mut max_sum = 0.0;
    let mut first_idx = 0;
    for i in 0..n {
        let sum: f64 = distance_matrix[i].iter().sum();
        if sum > max_sum {
            max_sum = sum;
            first_idx = i;
        }
    }
    selected_indices.push(first_idx);

    // Select remaining atoms iteratively
    for _ in 1..num_atoms {
        let mut max_dist_sum = 0.0;
        let mut next_idx = 0;

        for i in 0..n {
            if selected_indices.contains(&i) {
                continue;
            }

            // Sum distances to all previously selected atoms
            let dist_sum: f64 = selected_indices
                .iter()
                .map(|&selected| distance_matrix[i][selected])
                .sum();

            if dist_sum > max_dist_sum {
                max_dist_sum = dist_sum;
                next_idx = i;
            }
        }

        selected_indices.push(next_idx);
    }

    // Return copies of selected diagrams as initial atoms
    selected_indices
        .iter()
        .map(|&idx| input_diagrams[idx].clone())
        .collect()
}

pub struct WassersteinDictionary {
    pub atoms: Vec<PersistenceDiagram>,
    pub weights: Vec<Vec<f64>>, // weights[n] = weights for input_diagrams[n]
    pub final_energy: f64,
}

pub fn optimize_wasserstein_dictionary(
    input_diagrams: &[PersistenceDiagram],
    num_atoms: usize,
    max_iterations_per_scale: usize,
    num_scales: usize,
) -> WassersteinDictionary {
    let n = input_diagrams.len();
    let m = num_atoms;

    // Compute scalar range for each diagram
    let scalar_ranges: Vec<(f64, f64)> = input_diagrams
        .iter()
        .map(|diag| {
            let mut min_val = f64::INFINITY;
            let mut max_val = f64::NEG_INFINITY;
            for pair in diag.pairs() {
                min_val = min_val.min(pair.birth);
                max_val = max_val.max(pair.death);
            }
            (min_val, max_val)
        })
        .collect();

    // Global scalar range
    let global_min = scalar_ranges
        .iter()
        .map(|&(min, _)| min)
        .fold(f64::INFINITY, f64::min);
    let global_max = scalar_ranges
        .iter()
        .map(|&(_, max)| max)
        .fold(f64::NEG_INFINITY, f64::max);

    let tau_values: Vec<f64> = (0..num_scales)
        .map(|i| (0.2 - 0.05 * i as f64).max(0.0))
        .collect();

    // Initialize dictionary and weights
    let mut atoms = initialize_dictionary(input_diagrams, num_atoms);
    let mut weights_set: Vec<Vec<f64>> = vec![vec![1.0 / m as f64; m]; n];

    let mut best_energy = f64::INFINITY;
    let mut best_atoms = atoms.clone();
    let mut best_weights = weights_set.clone();
    let mut iterations_without_improvement = 0;

    for &tau in &tau_values {
        let filtered_inputs: Vec<PersistenceDiagram> = input_diagrams
            .iter()
            .zip(scalar_ranges.iter())
            .map(|(diag, &(min, max))| {
                let range = max - min;
                diag.filter_by_relative_persistence(tau, range)
            })
            .collect();

        let mut filtered_atoms: Vec<PersistenceDiagram> = atoms
            .iter()
            .map(|atom| {
                let range = global_max - global_min;
                atom.filter_by_relative_persistence(tau, range)
            })
            .collect();

        let mut all_diagrams = filtered_inputs.clone();
        all_diagrams.extend(filtered_atoms.clone());
        let augmented = augment_diagram_set(&all_diagrams);

        let augmented_inputs: Vec<_> = augmented.iter().take(n).cloned().collect();

        filtered_atoms = augmented.iter().skip(n).cloned().collect();

        // Actually do the optimization
        for iter in 0..max_iterations_per_scale {
            // Weight optimization
            for i in 0..n {
                weights_set[i] =
                    optimize_weights_step(&augmented_inputs[i], &filtered_atoms, &weights_set[i]);
            }

            // Atom optimization (do not fuse with above loop)
            for i in 0..n {
                filtered_atoms = optimize_atoms_step(
                    &augmented_inputs[i],
                    &filtered_atoms,
                    &weights_set[i],
                    global_min,
                    global_max,
                );
            }

            let current_energy =
                compute_dictionary_energy(&augmented_inputs, &filtered_atoms, &weights_set);

            // Track best solution
            if current_energy < best_energy {
                best_energy = current_energy;
                best_atoms = filtered_atoms.clone();
                best_weights = weights_set.clone();
                iterations_without_improvement = 0;
            } else {
                iterations_without_improvement += 1;
            }

            // Stopping condition: no improvement for 10 iterations
            if iterations_without_improvement >= 10 {
                println!(
                    "  Converged at iteration {} with energy {}",
                    iter, best_energy
                );
                break;
            }

            if iter % 10 == 0 {
                println!("  Iteration {}: energy = {}", iter, current_energy);
            }
        }

        atoms = best_atoms.clone();
        weights_set = best_weights.clone();
    }

    WassersteinDictionary {
        atoms: best_atoms,
        weights: best_weights,
        final_energy: best_energy,
    }
}
