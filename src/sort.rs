use numpy::ndarray::ArrayViewMut2;

pub fn sort_by_persistence(diagram: &mut ArrayViewMut2<f64>) {
    assert_eq!(diagram.shape()[1], 2);

    let mut vector = diagram
        .outer_iter()
        .map(|point| unsafe { [*point.uget(0), *point.uget(1)] })
        .collect::<Vec<[f64; 2]>>();

    vector.sort_unstable_by(|a, b| (b[1] - b[0]).total_cmp(&(a[1] - a[0])));

    for (mut output, point) in diagram.outer_iter_mut().zip(vector.iter()) {
        unsafe {
            *output.uget_mut(0) = point[0];
            *output.uget_mut(1) = point[1];
        }
    }
}
