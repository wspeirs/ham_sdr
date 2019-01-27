use std::iter::{empty, once};
use std::iter::Iterator;

use num::complex::Complex32;


struct FMQuadratureDemodulate {
    last: Option<Complex32>
}

impl FMQuadratureDemodulate {
    pub fn new() -> FMQuadratureDemodulate {
        FMQuadratureDemodulate { last: None }
    }

    /// Demodulate IQ samples
    pub fn demodulate(&mut self, samples: &[Complex32]) -> Vec<f32> {
        if samples.len() < 2 {
            panic!("Samples must be >= 2");
        }

        let t = if self.last.is_some() {
            let mut v = Vec::with_capacity(samples.len() + 1);
            v.push(self.last.unwrap());
            v.extend_from_slice(samples);

            v
        } else {
            samples.to_vec()
        };

        // set our last value
        self.last = Some(*t.last().unwrap());

        // compute instantaneous phase via arctan2() function, also called arg()
        let t = t.into_iter().map(|s| s.arg()).collect::<Vec<_>>();

        // compute the difference
        t.windows(2).map(|s| s[1]-s[0]).collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use num::complex::Complex32;

    #[test]
    fn test() {
        let a = vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -2.0),
        ];

        let mut fm = FMQuadratureDemodulate::new();

        let res = fm.demodulate(&a);
        res.iter().for_each(|n| println!("{:?}", n));

        assert_eq!(1, res.len());
        assert_eq!(-1.5707964, res[0]);

        let a = vec![
            Complex32::new(3.0, 3.0),
            Complex32::new(4.0, -4.0),
            Complex32::new(5.0, 5.0),
            Complex32::new(6.0, -6.0),
        ];

        let res = fm.demodulate(&a);
        res.iter().for_each(|n| println!("{:?}", n));

        assert_eq!(4, res.len());
        assert_eq!(1.5707964, res[0]);
        assert_eq!(-1.5707964, res[1]);
        assert_eq!(1.5707964, res[2]);
        assert_eq!(-1.5707964, res[3]);

    }
}
