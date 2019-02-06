use std::iter::{empty, once};
use std::iter::Iterator;

use num::complex::Complex32;
use std::f32::consts::PI;
use std::cmp;

pub struct QuadratureDemodulator {
    last: Option<Complex32>
}

impl QuadratureDemodulator {
    pub fn new() -> QuadratureDemodulator {
        QuadratureDemodulator { last: None }
    }



    /// Demodulate wide-band FM IQ samples
    pub fn wide_band_fm(&mut self, quadrature_rate: f32, samples: &[Complex32]) -> Vec<f32> {

        let max_dev = 75_000.0; // defines wide-band?
        let demodulation_gain = quadrature_rate / (2.0 * PI * max_dev);
//        let audio_rate = quad_rate / audio_decimation;

        if samples.len() < 2 {
            panic!("Samples must be >= 2");
        }

//        let t = if self.last.is_some() {
//            let mut v = Vec::with_capacity(samples.len() + 1);
//            v.push(self.last.unwrap());
//            v.extend_from_slice(samples);
//
//            v
//        } else {
//            samples.to_vec()
//        };
//
//        // set our last value
//        self.last = Some(*t.last().unwrap());

        // from: https://libvolk.org/doxygen/volk_32fc_x2_multiply_conjugate_32fc.html
        let mut mul_conj = Vec::with_capacity(samples.len());

        // compute sample[1] * sample[0].conj
        samples.windows(2).map(|s| s[1] * s[0].conj()).for_each(|c| mul_conj.push(c));

        let mut demod = Vec::with_capacity(mul_conj.len());

        // compute instantaneous phase via arctan2() function, also called arg()
        mul_conj.into_iter().map(|s| s.arg() * demodulation_gain).for_each(|f| demod.push(f));

        // compute the difference
//        t.windows(2).map(|s| s[1]-s[0]).collect::<Vec<_>>()

        demod
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use num::complex::Complex32;
    use std::io::{Read, Write, BufReader, BufWriter};
    use std::fs::{File, OpenOptions};
    use byteorder::{LE, BE, WriteBytesExt, ReadBytesExt};

    #[test]
    fn test() {
        let a = vec![
            Complex32::new(1.0, 1.0),
            Complex32::new(2.0, -2.0),
        ];

        let mut fm = QuadratureDemodulator::new();

        let res = fm.wide_band_fm(10_000_000.0/20.0, &a);
        res.iter().for_each(|n| println!("{:?}", n));

        assert_eq!(1, res.len());
        assert_eq!(-1.5707964, res[0]);

        let a = vec![
            Complex32::new(3.0, 3.0),
            Complex32::new(4.0, -4.0),
            Complex32::new(5.0, 5.0),
            Complex32::new(6.0, -6.0),
        ];

        let res = fm.wide_band_fm(10_000_000.0/20.0, &a);
        res.iter().for_each(|n| println!("{:?}", n));

        assert_eq!(4, res.len());
        assert_eq!(1.5707964, res[0]);
        assert_eq!(-1.5707964, res[1]);
        assert_eq!(1.5707964, res[2]);
        assert_eq!(-1.5707964, res[3]);

    }

    #[test]
    fn test_file() {
        let mut in_file = BufReader::new(OpenOptions::new().read(true).create(false).open("post_low_pass.dat").expect("Cannot open post_low_pass.dat file"));
        let mut test_file = BufWriter::new(OpenOptions::new().write(true).create(true).truncate(true).open("test.dat").expect("Cannot open phase file"));

        let mut fm_demod = QuadratureDemodulator::new();

        let mut values = Vec::new();

        // read in all the values
        loop {
            let re = match in_file.read_f32::<LE>() {
                Err(e) => { println!("Error: {:?}", e); break; },
                Ok(v) => v
            };

            let im = in_file.read_f32::<LE>().unwrap();

            values.push(Complex32::new(re, im));
        }

        // demodulate them
        let out = fm_demod.wide_band_fm(10_000_000.0/20.0, &values);

        for o in out {
            test_file.write_f32::<LE>(o).unwrap();
        }
    }

}
