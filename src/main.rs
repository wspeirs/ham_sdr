#[macro_use]
extern crate log;

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;
use std::thread::sleep;
use std::io::{Write, stdout, BufWriter};
use std::fs::{File, OpenOptions};

use byteorder::{LE, WriteBytesExt};

use std::f32::consts::PI;

use num::complex::Complex32;

use sdr::fir::FIR;
use sdr::fm::FMDemod;

use rs_libhackrf::hackrf::HackRF;
use rs_libhackrf::error::Error;

const FREQ :u64 = 95_900_000; // set to 95.9MHz
const SAMPLE_RATE :u32 = 10_000_000;

fn main() -> Result<(), Error> {
//    simple_logger::init_with_level(log::Level::Debug).unwrap();
//    simple_logger::init_with_level(log::Level::Info).unwrap();

    let mut hrf = HackRF::new().unwrap();
    let mut dev = hrf.open_device(0).expect("Error opening device; not plugged in?");

    dev.set_vga_gain(20)?;
    dev.set_lna_gain(20)?;
    dev.set_amp_enable(false);

    dev.set_antenna_enable(true);

    dev.set_freq(FREQ)?;
    dev.set_sample_rate(SAMPLE_RATE as f64)?;

    let mut test_file = BufWriter::new(OpenOptions::new().write(true).create(true).open("test.dat").expect("Cannot open phase file"));

    let mut low_pass_fir :FIR<Complex32> = FIR::lowpass(2409, 0.02); // cut off / (sample/2)
    let mut decimation_fir = FIR::new(low_pass_fir.taps(), 20, 1);
    let mut resample_fir = FIR::resampler(4103, 500, 96);

    let mut fm_demod = FMDemod::new();

    dev.start_rx(|buff| {
        let mut std_out = stdout();

//        buff.iter().for_each(|b| {
//            test_file.write_u8(*b);
//        });
//
//        test_file.flush();

        // convert from 8-bit IQ to complex values
        let buff = buff.chunks(2).map(|chunk| {
            Complex32::new(chunk[0] as f32 / 128.0, chunk[1] as f32 / 128.0)
        }).collect::<Vec<_>>();

        // low-pass filter, and decimation
        let buff = decimation_fir.process(&buff);

//        buff.iter().for_each(|c| {
//            test_file.write_f32::<LE>(c.re);
//            test_file.write_f32::<LE>(c.im);
//        });
//
//        test_file.flush();

        // demodulation
        let res = fm_demod.process(&buff);

//        res.iter().for_each(|f| {
//            test_file.write_f32::<LE>(*f);
//        });
//
//        test_file.flush();

        // re-sample
        let res = resample_fir.process(&res);

        res.iter().for_each(|r| {
//            std_out.write_f32::<LE>(*r);
            test_file.write_f32::<LE>(*r);
        });

//        std_out.flush();
        test_file.flush();

        Error::SUCCESS
    })?;

    sleep(Duration::from_secs(3));

    Ok(())
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_zip() {
        let buff = (0..20).collect::<Vec<u8>>();

        let i = buff.iter().step_by(2).map(|n| *n);
        let q = buff.iter().skip(1).step_by(2).map(|n| *n);

        for (i,q) in i.zip(q) {
            println!("I: {} Q: {}", i, q);
        }
    }

}

