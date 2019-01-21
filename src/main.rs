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

use rs_libhackrf::hackrf::HackRF;
use rs_libhackrf::error::Error;

mod fm;

use fm::FmDemod;

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

    let mut demod = Rc::new(RefCell::new(FmDemod::new(75000, SAMPLE_RATE)));
    let mut demod_clone = demod.clone();

    let mut mag_file = BufWriter::new(OpenOptions::new().write(true).create(true).open("my_mag.dat").expect("Cannot open mag file"));
    let mut phase_file = BufWriter::new(OpenOptions::new().write(true).create(true).open("my_phase.dat").expect("Cannot open phase file"));
    let mut test_file = BufWriter::new(OpenOptions::new().write(true).create(true).open("my_test.dat").expect("Cannot open phase file"));

//    let mut mag_file = Rc::new(RefCell::new(OpenOptions::new().write(true).create_new(true).open("my_mag.dat")));
//    let mut phase_file = Rc::new(RefCell::new(OpenOptions::new().write(true).create_new(true).open("my_phase.dat")));
//    let mut mag_file_clone = mag_file.clone();
//    let mut phase_file = phase_file.clone();

    dev.start_rx(|buff| {
        let mut demod = demod_clone.borrow_mut();
        let mut std_out = stdout();

//        buff.iter().for_each(|b| {
//            test_file.write_u8(*b);
//        });
//
//        test_file.flush();

        let buff = buff.chunks(2).map(|chunk| {
            Complex32::new(chunk[0] as f32 / 128.0, chunk[1] as f32 / 128.0)
        }).collect::<Vec<_>>();


        buff.iter().for_each(|c| {
            test_file.write_f32::<LE>(c.re);
            test_file.write_f32::<LE>(c.im);
        });

        test_file.flush();

//        buff.iter().for_each(|c| {
//            mag_file.write_f32::<BigEndian>(c.re);
//            phase_file.write_f32::<BigEndian>(c.im);
//        });
//
//        mag_file.flush();
//        phase_file.flush();

//        let buff = buff.chunks(8).map(|chunk| {
//            vec![
//                Complex32::new((chunk[0] as f32) - 127.5, (chunk[1] as f32) - 127.5),
//                Complex32::new(((255 - chunk[3]) as f32) - 127.5, (chunk[2] as f32) - 127.5),
//                Complex32::new(((255 - chunk[4]) as f32) - 127.5, ((255 - chunk[5]) as f32) - 127.5),
//                Complex32::new((chunk[7] as f32) - 127.5, ((255 - chunk[6]) as f32) - 127.5)
//            ]
//        }).flatten().collect::<Vec<_>>();
//
//        // sum up the values in chunks of 6; downsampling
//        let buff = buff.chunks(6).map(|chunk| {
//            chunk.into_iter().sum()
//        });
//
//        // do the demodulation
//        let mut buff = buff
//            .map(|c| demod.feed(c))
//            .map(|f| (f / PI * (1<<14) as f32) as i16)
//            .collect::<Vec<_>>()
//        ;
//
//        // deemph filter???
//
//        // low pass
//        let i2 :isize = 0;
//        let fast :isize = 170000; // rate_out
//        let slow :isize = 32000;  // rate_out2
//        let mut now_lpr :isize = 0;
//        let mut pre_lpr_index :isize = 0;
//        let mut ret_buff = Vec::new();
//
//        for res in buff {
//            now_lpr += res as isize;
//            pre_lpr_index += slow;
//
//            if pre_lpr_index < fast {
//                continue
//            }
//
//            ret_buff.push((now_lpr / (fast/slow)) as i16);
//            now_lpr = 0;
//            pre_lpr_index -= fast;
//        }
//
//        ret_buff.into_iter().for_each(|v| {
//            std_out.write_i16::<BigEndian>(v);
//        });

        Error::SUCCESS
//        Error::INVALID_PARAM(String::new())
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

