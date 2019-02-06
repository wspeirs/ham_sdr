use num::complex::Complex32;
use std::f32::consts::PI;


pub struct Filter {
    taps: Vec<f32>
}

impl Filter {
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

    pub fn new(taps: &[f32]) -> Filter {
        let mut t = Vec::<f32>::with_capacity(taps.len());
        t.extend_from_slice(taps);

        return Filter { taps: t }
    }

    pub fn filter(&self, decimation: usize, input: &[Complex32]) -> Vec<Complex32> {
        let mut ret = Vec::<Complex32>::with_capacity(input.len());

        let stop = (input.len() / decimation) * decimation;

        for n in (0..stop).step_by(decimation) {
            let mut real: f64 = 0.0;
            let mut img: f64 = 0.0;

            let start = if n >= self.taps.len() { n - (self.taps.len()-1) } else { 0 };

            for k in start..=n {
                real += input[k].re as f64 * self.taps[n-k] as f64;
                img += input[k].im as f64 * self.taps[n-k] as f64;
            }

            ret.push(Complex32::new(real as f32, img as f32));
        }

        ret
    }
}


#[cfg(test)]
mod test {
    use super::*;
    use num::complex::Complex32;

    #[test]
    fn test_generate_low_pass_taps() {
        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);

        assert_eq!(*taps.first().unwrap(), 0.00000526322);
        assert_eq!(*taps.last().unwrap(), 0.00000526322);
    }

    #[test]
    fn test_low_pass() {
        let input = vec![Complex32::new(1.0, 1.0); 100];

        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let filter = Filter::new(&taps);
        let output = filter.filter(1, &input);

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

        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let filter = Filter::new(&taps);
        let output = filter.filter(1, &input);

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

        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let filter = Filter::new(&taps);
        let output = filter.filter(1, &input);

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

        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let filter = Filter::new(&taps);
        let output = filter.filter(1, &input);

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