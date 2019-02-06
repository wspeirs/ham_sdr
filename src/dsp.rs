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

    /// Generates the taps for a low-pass filter
    /// Translated from https://github.com/gnuradio/gnuradio/blob/v3.7.9.3/gr-filter/lib/firdes.cc#L92
    pub fn generate_low_pass_taps(gain :f32, sampling_freq :f32, cutoff_freq :f32, transition_width :f32) -> Vec<f32> {
        // perform some sanity checks
        assert!(sampling_freq > 0.0, format!("sampling_freq ({}) < 0", sampling_freq));
        assert!(cutoff_freq > 0.0, format!("cutoff_freq ({}) <= 0", cutoff_freq));
        assert!(cutoff_freq <= sampling_freq / 2.0, format!("cutoff_freq ({}) > {}", cutoff_freq, sampling_freq/2.0));
        assert!(transition_width > 0.0, format!("transition_width <= 0"));

        // we're using a Hamming window
        const MAX_ATTENUATION :f32 = 53.0;

        let ntaps = (MAX_ATTENUATION * sampling_freq / (22.0 * transition_width)).floor() as isize;
        let ntaps  = if (ntaps & 1) == 0 { ntaps + 1 } else { ntaps };

        // construct the truncated ideal impulse response
        // [sin(x)/x for the low pass case]

        let mut window = Vec::with_capacity(ntaps as usize);

        // compute the window values
        for n in 0..ntaps {
            window.push(0.54 - 0.46 * ((2.0 * PI * n as f32) / (ntaps - 1) as f32).cos());
        }

        let M :isize = (ntaps - 1) / 2;
        let fw_t0 = 2.0 * PI * cutoff_freq / sampling_freq;

        let mut taps = vec![0.0; ntaps as usize];

        // compute the tap values
        for n in -M..=M {
            if n == 0 {
                taps[(n+M) as usize] = fw_t0 / PI * window[(n+M) as usize];
            } else {
                taps[(n+M) as usize] = (n as f32 * fw_t0).sin() / (n as f32 * PI) * window[(n+M) as usize];
            }
        }

        // find the factor to normalize the gain, fmax.
        // For low-pass, gain @ zero freq = 1.0

        let mut fmax = taps[0 + M as usize];

        for n in 1..= M {
            fmax += 2.0 * taps[(n + M) as usize];
        }

        let gain = gain / fmax;	// normalize

        for i in 0..ntaps as usize {
            taps[i] *= gain;
        }

        return taps;
    }

    pub fn fir_filter(samples: &[Complex32], taps: &[f32], decimation: usize) -> Vec<Complex32> {
        let mut ret = Vec::<Complex32>::with_capacity(samples.len());

        let stop = (samples.len() / decimation) * decimation;

        for n in (0..stop).step_by(decimation) {
            let mut real: f64 = 0.0;
            let mut img: f64 = 0.0;

            let start = if n >= taps.len() { n - (taps.len()-1) } else { 0 };

            for k in start..=n {
                real += samples[k].re as f64 * taps[n-k] as f64;
                img += samples[k].im as f64 * taps[n-k] as f64;
            }

            ret.push(Complex32::new(real as f32, img as f32));
        }

        ret
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
    fn test_read() {
        let mut in_file = BufReader::new(OpenOptions::new().read(true).create(false).open("ones_low_pass.dat").expect("Cannot open post_low_pass.dat file"));

        loop {
            let f = in_file.read_f32::<LE>().unwrap();

            println!("{:0.016}", f);
        }
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

    #[test]
    fn test_generate_low_pass_taps() {
        let taps = QuadratureDemodulator::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);

        assert_eq!(*taps.first().unwrap(), 0.00000526322);
        assert_eq!(*taps.last().unwrap(), 0.00000526322);
    }

    #[test]
    fn test_low_pass() {
        let input = vec![Complex32::new(1.0, 1.0); 100];

        let taps = QuadratureDemodulator::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let output = QuadratureDemodulator::fir_filter(&input, &taps, 1);

        assert_eq!(output.len(), 100);

        assert_eq!(output[0].re, 0.00000526322);
        assert_eq!(output[0].im, 0.00000526322);

        assert_eq!(output[1].re, 0.000009232372);
        assert_eq!(output[1].im, 0.000009232372);

        assert_eq!(output[2].re, 0.000011889521);
        assert_eq!(output[2].im, 0.000011889521);

        assert_eq!(output[3].re, 0.000013222045);
        assert_eq!(output[3].im, 0.000013222045);

        assert_eq!(output[4].re, 0.000013222048);
        assert_eq!(output[4].im, 0.000013222048);

        assert_eq!(output[99].re, 0.00010319796);
        assert_eq!(output[99].im, 0.00010319796);

    }

    #[test]
    fn test_low_pass_decimation() {
        let input = vec![Complex32::new(1.0, 1.0); 100];

        let taps = QuadratureDemodulator::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let output = QuadratureDemodulator::fir_filter(&input, &taps, 20);

        println!("TAPS: {}", taps.len());

        assert_eq!(output.len(), 5);

        // these were taken from running 100 (1.0, 1.0) numbers through the GNURadio low-pass filter
        assert_eq!(output[0].re, 0.00000526322);
        assert_eq!(output[0].im, 0.00000526322);

        assert_eq!(output[1].re, -0.00015483372);
        assert_eq!(output[1].im, -0.00015483372);

        assert_eq!(output[2].re, -0.0005668219);
        assert_eq!(output[2].im, -0.0005668219);

        assert_eq!(output[3].re, -0.00065905956);
        assert_eq!(output[3].im, -0.00065905956);

        assert_eq!(output[4].re, -0.00025988425);
        assert_eq!(output[4].im, -0.00025988425);
    }

    #[test]
    fn test_low_pass_large_input() {
        let input = vec![Complex32::new(1.0, 1.0); 2500];

        let taps = QuadratureDemodulator::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let output = QuadratureDemodulator::fir_filter(&input, &taps, 1);

        println!("TAPS: {}", taps.len());

        assert_eq!(output.len(), 2500);

        // these were taken from running 100 (1.0, 1.0) numbers through the GNURadio low-pass filter
        assert_eq!(output[0].re, 0.00000526322);
        assert_eq!(output[0].im, 0.00000526322);

        assert_eq!(output[1].re, 0.000009232372);
        assert_eq!(output[1].im, 0.000009232372);

        assert_eq!(output[2].re, 0.000011889521);
        assert_eq!(output[2].im, 0.000011889521);

        assert_eq!(output[3].re, 0.000013222045);
        assert_eq!(output[3].im, 0.000013222045);

        assert_eq!(output[4].re, 0.000013222048);
        assert_eq!(output[4].im, 0.000013222048);

        assert_eq!(output[2496].re, 1.0000006);
        assert_eq!(output[2496].im, 1.0000006);

        assert_eq!(output[2497].re, 1.0000006);
        assert_eq!(output[2497].im, 1.0000006);

        assert_eq!(output[2498].re, 1.0000006);
        assert_eq!(output[2498].im, 1.0000006);

        assert_eq!(output[2499].re, 1.0000006);
        assert_eq!(output[2499].im, 1.0000006);
    }

    #[test]
    fn test_low_pass_large_input_decimation() {
        let input = vec![Complex32::new(1.0, 1.0); 2500];

        let taps = QuadratureDemodulator::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let output = QuadratureDemodulator::fir_filter(&input, &taps, 103);

        println!("COMPUTED OUT LEN: {}", input.len() / 103);

        assert_eq!(output.len(), 24);

        // these were taken from running 100 (1.0, 1.0) numbers through the GNURadio low-pass filter
        assert_eq!(output[0].re, 0.00000526322);
        assert_eq!(output[0].im, 0.00000526322);

        assert_eq!(output[1].re, 0.0001204816);
        assert_eq!(output[1].im, 0.0001204816);

        assert_eq!(output[2].re, 0.00039262543);
        assert_eq!(output[2].im, 0.00039262543);

        assert_eq!(output[3].re, 0.00082254934);
        assert_eq!(output[3].im, 0.00082254934);

        assert_eq!(output[4].re, 0.0013695889);
        assert_eq!(output[4].im, 0.0013695889);

        assert_eq!(output[19].re, 1.0027326);
        assert_eq!(output[19].im, 1.0027326);

        assert_eq!(output[20].re, 1.0017551);
        assert_eq!(output[20].im, 1.0017551);

        assert_eq!(output[21].re, 1.0011041);
        assert_eq!(output[21].im, 1.0011041);

        assert_eq!(output[22].re, 1.0007218);
        assert_eq!(output[22].im, 1.0007218);

        assert_eq!(output[23].re, 1.0005318);
        assert_eq!(output[23].im, 1.0005318);
    }
}
