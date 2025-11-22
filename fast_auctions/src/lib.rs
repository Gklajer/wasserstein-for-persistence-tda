use numpy::ndarray::{ArrayView2, ArrayViewMut1, ArrayViewMut2};

mod auction;
mod binary_heap;
mod kd_tree;
mod normal;

#[pyo3::pymodule]
mod wasp {
    use pyo3::{Python, prelude::*};

    use numpy::{
        PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2,
        ndarray::{ArrayView2, ArrayViewMut1},
    };

    /**
     * Python function to compute the L^2 Wasserstein distance of two diagrams, and their optimal matching.
     * The first output has the same length as x and contains the indices in y of the points paired
     * with the off-diagonal points. An index of u32::MAX indicates that the point was matched to its projection.
     * The second output has the same length as y and contains the indices in x of the projected points paired
     * with the diagonal points. An index of u32::MAX indicates that the points was matched to the point (in y)
     * of which it is the projection. Of course, in this array, only the u32::MAX values matter for the final cost,
     * as diagonal-diagonal pairings have zero cost.
     * The third output is the squared L^2 Wasserstein distance between x and y.
     */
    #[pyfunction]
    pub fn wasserstein_distance<'py>(
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray2<'py, f64>,
        delta: f64,
    ) -> (Py<PyArray1<u32>>, Py<PyArray1<u32>>, f64) {
        let x_view: ArrayView2<f64> = x.as_array();
        let y_view: ArrayView2<f64> = y.as_array();

        let off_diag_output = PyArray1::<u32>::zeros(x.py(), x_view.shape()[0], false);
        let diag_output = PyArray1::<u32>::zeros(y.py(), y_view.shape()[0], false);

        let mut off_diag_output_readwrite = off_diag_output.readwrite();
        let mut diag_output_readwrite = diag_output.readwrite();

        let off_diag_output_view: ArrayViewMut1<u32> = off_diag_output_readwrite.as_array_mut();
        let diag_output_view: ArrayViewMut1<u32> = diag_output_readwrite.as_array_mut();

        let distance = crate::wasserstein_distance(
            x_view,
            y_view,
            off_diag_output_view,
            diag_output_view,
            delta,
        )
        .unwrap();

        (off_diag_output.unbind(), diag_output.unbind(), distance)
    }

    /**
     * Sample a persistence diagram of size n (mainly for banchmarking purposes) from the
     * "normal" distribution described in arXiv:1606.03357.
     */
    #[pyfunction]
    pub fn sample_normal_diagram<'py>(py: Python<'py>, n: usize) -> Py<PyArray2<f64>> {
        let output = PyArray2::<f64>::zeros(py, [n, 2], false);
        let mut output_readwrite = output.readwrite();
        let output_view = output_readwrite.as_array_mut();
        crate::sample_normal_diagram(output_view);
        output.unbind()
    }
}

/**
 * Checks the shape of an input persistence diagram.
 */
fn validate_input_diagram(x: &ArrayView2<f64>) -> Result<(), String> {
    let shape = x.shape();
    if shape[0] >= u32::MAX as usize {
        return Err(String::from("One of the input diagrams is too large! "));
    }
    if shape[1] != 2 {
        return Err(String::from(
            "The second dimension of the input diagrams must be two! ",
        ));
    }
    Ok(())
}

/**
 * Rust equivalent of the wasp::wasserstein_distance function
 */
pub fn wasserstein_distance(
    x: ArrayView2<f64>,
    y: ArrayView2<f64>,
    mut off_diag: ArrayViewMut1<u32>,
    mut diag: ArrayViewMut1<u32>,
    delta: f64,
) -> Result<f64, String> {
    validate_input_diagram(&x)?;
    validate_input_diagram(&y)?;
    if x.shape()[0] != off_diag.shape()[0] {
        return Err(String::from("off_diag must have the same length as y! "));
    }
    if y.shape()[0] != diag.shape()[0] {
        return Err(String::from("diag must have the same length as x! "));
    }
    if !f64::is_finite(delta) || delta <= 0.0 {
        return Err(String::from("delta must be strictly positive! "));
    }

    Ok(auction::auction_algorithm(
        &x,
        &y,
        &mut off_diag,
        &mut diag,
        None, 
        None,
        delta,
    ))
}

/**
 * Rust equivalent of the wasp::sample_normal_diagram
 */
pub fn sample_normal_diagram(mut output: ArrayViewMut2<f64>) {
    normal::sample_diagram(100.0, &mut output);
}
