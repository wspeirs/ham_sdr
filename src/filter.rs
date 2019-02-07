use num::complex::{Complex32, Complex64};
use std::f64::consts::PI;


pub struct Filter {
    taps: Vec<f32>,
    output: Vec<Complex32>  // we keep this around so we don't have to re-allocate every time
}

impl Filter {
    /// Generates the taps for a low-pass filter
    /// Translated from https://github.com/gnuradio/gnuradio/blob/v3.7.9.3/gr-filter/lib/firdes.cc#L92
    pub fn generate_low_pass_taps(gain :f64, sampling_freq :f64, cutoff_freq :f64, transition_width :f64) -> Vec<f32> {
        // perform some sanity checks
        assert!(sampling_freq > 0.0, format!("sampling_freq ({}) < 0", sampling_freq));
        assert!(cutoff_freq > 0.0, format!("cutoff_freq ({}) <= 0", cutoff_freq));
        assert!(cutoff_freq <= sampling_freq / 2.0, format!("cutoff_freq ({}) > {}", cutoff_freq, sampling_freq/2.0));
        assert!(transition_width > 0.0, format!("transition_width <= 0"));

        // we're using a Hamming window
        const MAX_ATTENUATION :f64 = 53.0;

        let ntaps = (MAX_ATTENUATION * sampling_freq / (22.0 * transition_width)).floor() as isize;
        let ntaps  = if (ntaps & 1) == 0 { ntaps + 1 } else { ntaps };

        // construct the truncated ideal impulse response
        // [sin(x)/x for the low pass case]

        let mut window :Vec<f32> = Vec::with_capacity(ntaps as usize);
        let M = (ntaps - 1) as f64;

        // compute the window values
        for n in 0..ntaps {
//            println!("{:0.20} - {:0.20} * (({:0.20}) / {:0.20}).cos()", 0.54, 0.46, (2.0 * PI * n as f64), M);
//            println!("{:0.20}", 2.0_f64 * PI_64 * n as f64);
//            println!("({:0.20}).cos() = {:0.20}", (2.0_f64 * PI * n as f64) / M as f64, ((2.0_f64 * PI * n as f64) / M as f64).cos());
//            println!("{:0.20}", 0.54 - 0.46 * ((2.0_f64 * PI * n as f64) / M as f64).cos());
            window.push((0.54 - 0.46 * ((2.0 * PI * n as f64) / M).cos()) as f32);
        }

        let M :isize = (ntaps - 1) / 2;
        let fw_t0 = 2.0 * PI * cutoff_freq / sampling_freq;

        let mut taps :Vec<f32> = vec![0.0; ntaps as usize];

        // compute the tap values
        for n in -M..=M {
            if n == 0 {
                taps[(n+M) as usize] = (fw_t0 / PI * window[(n+M) as usize] as f64) as f32;
            } else {
                taps[(n+M) as usize] = ((n as f64 * fw_t0).sin() / (n as f64 * PI) * window[(n+M) as usize] as f64) as f32;
            }
        }

        // find the factor to normalize the gain, fmax.
        // For low-pass, gain @ zero freq = 1.0

        let mut fmax = taps[0 + M as usize] as f64;

        for n in 1..= M {
            fmax += 2.0 * taps[(n + M) as usize] as f64;
        }

        let gain = gain / fmax;	// normalize

        for i in 0..ntaps as usize {
            taps[i] = (taps[i] as f64 * gain) as f32;
        }

        return taps;
    }

    /// Construct a new filter using taps.len() as the size for the output buffer
    pub fn new(taps: &[f32]) -> Filter {
        Filter::with_capacity(taps, taps.len())
    }

    /// Construct a new filter using the given capacity for the output buffer
    pub fn with_capacity(taps: &[f32], capacity: usize) -> Filter {
        let mut r_taps = Vec::<f32>::with_capacity(taps.len());

        // we want to reverse the taps so it's easier to manage in the filter code
        taps.iter().rev().for_each(|t| r_taps.push(*t));

        Filter {
            taps: r_taps,
            output: vec![Complex32::new(0.0, 0.0); capacity]
        }
    }

    fn dot_product(input: &[Complex32], taps: &[f32]) -> Complex32 {
        assert_eq!(input.len(), taps.len());

        let f = |acc :(f32,f32), (i, t) :(&Complex32, &f32)| (i.re.mul_add(*t, acc.0), i.im.mul_add(*t, acc.1));

        let (r,i) = input.iter()
            .zip(taps.iter())
            .fold((0.0, 0.0), f );

        return Complex32::new(r,i);
    }

    pub fn filter(&mut self, decimation: usize, input: &[Complex32]) -> &[Complex32] {
        let stop = (input.len() / decimation) * decimation;

        // ensure we have enough space in our internal buffer for the result
        if self.output.len() < stop {
            println!("REALLOC: {} -> {}", self.output.len(), stop);
            self.output.resize(stop, Complex32::new(0.0, 0.0));
        }

        let mut output_index = 0;

        // compute the convolution
        for n in (0..stop).step_by(decimation) {
            let i_start = if n >= self.taps.len() { n - (self.taps.len()-1) } else { 0 };
            let t_start = if n >= self.taps.len()-1 { 0 } else { (self.taps.len()-1) - n};

            let c = Filter::dot_product(&input[i_start..=n], &self.taps[t_start..]);

            self.output[output_index] = c;
            output_index += 1;
        }

        &self.output[0..output_index]
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
    fn test_generate_low_pass_taps() {
        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let mut taps_file = BufReader::new(OpenOptions::new().read(true).create(false).open("taps.dat").expect("Cannot open taps.dat file"));
        let mut my_file = BufWriter::new(OpenOptions::new().write(true).create(true).truncate(true).open("my_taps.dat").expect("Cannot open phase file"));

        taps.iter().for_each(|t| my_file.write_f32::<LE>(*t).unwrap());

        let mut i = 0;

        loop {
            let ftap = taps_file.read_f32::<LE>();

            if ftap.is_err() {
                break;
            }

            let ftap = ftap.unwrap();

            assert_eq!(taps[i], ftap);

            i += 1;
        }
    }

    #[test]
    fn test_low_pass_iq() {
        let mut iq_file = BufReader::new(OpenOptions::new().read(true).create(false).open("iq_10000.dat").expect("Cannot open iq.dat file"));
        let mut input = Vec::<Complex32>::new();

        loop {
            let r = iq_file.read_f32::<LE>();

            if r.is_err() {
                break;
            }

            let r = r.unwrap();
            let i = iq_file.read_f32::<LE>().unwrap();

            input.push(Complex32::new(r, i));
        }

        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let mut filter = Filter::new(&taps);
        let output = filter.filter(20, &input);

//        assert_eq!(output.len(), 100);

        assert_eq!(output[0].re, 0.00000468754569737939);
        assert_eq!(output[0].im, -0.00000111020824533625);

        assert_eq!(output[1].re, -0.00011307443492114544);
        assert_eq!(output[1].im, 0.00004673069997807033);

        assert_eq!(output[2].re, -0.00041445155511610210);
        assert_eq!(output[2].im, 0.00015367753803730011);

        assert_eq!(output[97].re, 1.00110352039337158203);
        assert_eq!(output[97].im, 1.00110352039337158203);

        assert_eq!(output[98].re, 1.00072109699249267578);
        assert_eq!(output[98].im, 1.00072109699249267578);

        assert_eq!(output[99].re, 1.00053107738494873047);
        assert_eq!(output[99].im, 1.00053107738494873047);
    }

    #[test]
    fn test_low_pass_ones() {
        let mut in_file = BufReader::new(OpenOptions::new().read(true).create(false).open("ones_low_pass.dat").expect("Cannot open file"));
        let mut file_output = Vec::<Complex32>::new();

        loop {
            let r = in_file.read_f32::<LE>();

            if r.is_err() {
                break;
            }

            let r = r.unwrap();
            let i = in_file.read_f32::<LE>().unwrap();

            file_output.push(Complex32::new(r, i));
        }

        let input = vec![Complex32::new(1.0, 1.0); 100];
        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let mut filter = Filter::new(&taps);
        let output = filter.filter(1, &input);

        assert_eq!(output.len(), 100);

        for i in 0..100 {
            assert_eq!(output[i].re as f32, file_output[i].re, "Mismatch at {}: {:0.20} != {:0.20}", i, output[i].re, file_output[i].re);
            assert_eq!(output[i].im as f32, file_output[i].im, "Mismatch at {}: {:0.20} != {:0.20}", i, output[i].im, file_output[i].im);
            println!("{:0.20} {:0.20}", output[i].re, file_output[i].re)
        }

//        assert_eq!(output[0].re, 0.00000526320945937186);
//        assert_eq!(output[0].im, 0.00000526320945937186);
//
//        assert_eq!(output[1].re, 0.00012048177450196818);
//        assert_eq!(output[1].im, 0.00012048177450196818);
//
//        assert_eq!(output[2].re, 0.00039262371137738228);
//        assert_eq!(output[2].im, 0.00039262371137738228);
//
//        assert_eq!(output[97].re, 1.00110352039337158203);
//        assert_eq!(output[97].im, 1.00110352039337158203);
//
//        assert_eq!(output[98].re, 1.00072109699249267578);
//        assert_eq!(output[98].im, 1.00072109699249267578);
//
//        assert_eq!(output[99].re, 1.00053107738494873047);
//        assert_eq!(output[99].im, 1.00053107738494873047);
    }

    #[test]
    fn test_low_pass_decimation() {
        let input = vec![Complex32::new(1.0, 1.0); 100];

        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let mut filter = Filter::new(&taps);
        let output = filter.filter(20, &input);

        println!("TAPS: {}", taps.len());

        assert_eq!(output.len(), 5);

        // these were taken from running 100 (1.0, 1.0) numbers through the GNURadio low-pass filter
        assert_eq!(output[0].re, 0.00000526320945937186);
        assert_eq!(output[0].im, 0.00000526320945937186);

        assert_eq!(output[1].re, -0.00015483508468605578);
        assert_eq!(output[1].im, -0.00015483508468605578);

        assert_eq!(output[2].re, -0.0005668227);
        assert_eq!(output[2].im, -0.0005668227);

        assert_eq!(output[3].re, -0.00065905834);
        assert_eq!(output[3].im, -0.00065905834);

        assert_eq!(output[4].re, -0.00025988230481743813);
        assert_eq!(output[4].im, -0.00025988230481743813);
    }

    #[test]
    fn test_low_pass_large_input() {
        let input = vec![Complex32::new(1.0, 1.0); 2500];

        let taps = Filter::generate_low_pass_taps(1.0, 10e6, 100e3, 10e3);
        let mut filter = Filter::new(&taps);
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
        let mut filter = Filter::new(&taps);
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