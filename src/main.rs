#[macro_use]
extern crate log;

use std::time::Duration;

use libusb::{Context, DeviceHandle};
use libusb::Error as USBError;

mod rtlsdr;

/// Goes through the devices until it finds a match for the product_id and vendor_id, then opens it
fn open_device(context: &Context, pid: u16, vid: u16) -> Result<DeviceHandle, USBError> {
    for mut device in context.devices().unwrap().iter() {
        let device_desc = device.device_descriptor().unwrap();

        debug!("DEVICE: {:#06x}:{:#06x}", device_desc.product_id(), device_desc.vendor_id());

        if device_desc.product_id() != pid || device_desc.vendor_id() != vid {
            continue;
        }

        return device.open();
    }

    return Err(USBError::NoDevice);
}

fn main() -> Result<(), USBError> {
    simple_logger::init_with_level(log::Level::Debug).unwrap();
//    simple_logger::init_with_level(log::Level::Info).unwrap();

    let mut context = libusb::Context::new()?;

    // open the SDR
    let mut device_handle = open_device(&context, 0x2838, 0x0bda)?;

    // claim the interface
    device_handle.claim_interface(0)?;

    // reset the USB device just in case
    device_handle.reset();

    //
    // RTL-SDR specific code
    //
    rtlsdr::init_baseband(&mut device_handle).expect("Error setting the baseband");


    Ok(())
}

