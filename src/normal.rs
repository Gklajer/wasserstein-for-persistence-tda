use numpy::ndarray::ArrayViewMut2;
use rand::{SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, StandardNormal};

pub fn sample_diagram(s: f64, output: &mut ArrayViewMut2<f64>) {
    assert_eq!(output.shape()[1], 2);

    let mut rng = SmallRng::from_os_rng();
    let std_dev = f64::sqrt(s);

    for mut point in output.outer_iter_mut() {
        unsafe {
            let a: f64 = rand::random::<f64>() * s;
            let b = Distribution::<f64>::sample(&StandardNormal, &mut rng) * std_dev;
            let half_abs_b = 0.5 * f64::abs(b);
            *point.uget_mut(0) = a - half_abs_b;
            *point.uget_mut(1) = a + half_abs_b;
        }
    }
}
