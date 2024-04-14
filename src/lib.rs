#![warn(unused_crate_dependencies, unreachable_pub)]
#![deny(unused_must_use, rust_2018_idioms)]

use alloy_primitives::{hex, Address, FixedBytes};
use byteorder::{BigEndian, ByteOrder, LittleEndian};
use console::Term;
use fs4::FileExt;
use ocl::{Buffer, Context, Device, MemFlags, Platform, ProQue, Program, Queue};
use rand::{thread_rng, Rng};
use separator::Separatable;
use std::fmt::Write as _;
use std::fs::{File, OpenOptions};
use std::io::prelude::*;
use std::time::{SystemTime, UNIX_EPOCH};
use terminal_size::{terminal_size, Height};

mod reward;
pub use reward::Reward;

static KERNEL_SRC: &str = include_str!("./kernels/keccak256.cl");

pub struct Config {
    pub factory: Address,
    pub owner: Address,
    pub init_code_hash: FixedBytes<32>,
    pub work_size: u32,
    pub gpu_device: u8,
    pub max_create3_nonce: u8,
    pub leading_zeroes_threshold: Option<u8>,
    pub total_zeroes_threshold: Option<u8>,
    pub output_file: String,
}

/// Given a Config object with a factory address, a caller address, a keccak-256
/// hash of the contract initialization code, and a device ID, search for salts
/// using OpenCL that will enable the factory contract to deploy a contract to a
/// gas-efficient address via CREATE3. This method also takes threshold values
/// for both leading zero bytes and total zero bytes - any address that does not
/// meet or exceed the threshold will not be returned. Default threshold values
/// are three leading zeroes or five total zeroes.
///
/// The 32-byte salt is constructed as follows:
///   - the 20-byte calling address (to prevent frontrunning)
///   - a random 4-byte segment (to prevent collisions with other runs)
///   - a 4-byte segment unique to each work group running in parallel
///   - a 4-byte nonce segment (incrementally stepped through during the run)
///
/// When a salt that will result in the creation of a gas-efficient contract
/// address is found, it will be appended to `efficient_addresses.txt` along
/// with the resultant address and the "value" (i.e. approximate rarity) of the
/// resultant address.
///
/// This method is still highly experimental and could almost certainly use
/// further optimization - contributions are more than welcome!
pub fn gpu(config: Config) -> ocl::Result<()> {
    println!(
        "Setting up experimental OpenCL miner using device {}...",
        config.gpu_device
    );

    // (create if necessary) and open a file where found salts will be written
    let file = output_file(&config.output_file);

    // create object for computing rewards (relative rarity) for a given address
    let rewards = Reward::new();

    // track how many addresses have been found and information about them
    let mut found: u64 = 0;
    let mut found_list: Vec<String> = vec![];

    // set up a controller for terminal output
    let term = Term::stdout();

    // set up a platform to use
    let platform = Platform::new(ocl::core::default_platform()?);

    // set up the device to use
    let device = Device::by_idx_wrap(platform, config.gpu_device as usize)?;

    // set up the context to use
    let context = Context::builder()
        .platform(platform)
        .devices(device)
        .build()?;

    // set up the program to use
    let program = Program::builder()
        .devices(device)
        .src(mk_kernel_src(&config))
        .build(&context)?;

    // set up the queue to use
    let queue = Queue::new(&context, device, None)?;

    let work_size = config.work_size;
    // set up the "proqueue" (or amalgamation of various elements) to use
    let ocl_pq = ProQue::new(context, queue, program, Some(work_size));
    let work_factor = (work_size as u128) / 1_000_000;

    // create a random number generator
    let mut rng = thread_rng();

    // determine the start time
    let start_time: f64 = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64();

    // set up variables for tracking performance
    let mut rate: f64 = 0.0;
    let mut cumulative_nonce: u64 = 0;

    // the previous timestamp of printing to the terminal
    let mut previous_time: f64 = 0.0;

    // the last work duration in milliseconds
    let mut work_duration_millis: u64 = 0;

    // begin searching for addresses
    loop {
        // construct the 4-byte message to hash, leaving last 8 of salt empty
        let salt = FixedBytes::<4>::random();

        // build a corresponding buffer for passing the message to the kernel
        let message_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(4)
            .copy_host_slice(&salt[..])
            .build()?;

        // reset nonce & create a buffer to view it in little-endian
        // for more uniformly distributed nonces, we shall initialize it to a random value
        let mut nonce: [u32; 1] = rng.gen();
        let mut view_buf = [0; 8];

        // build a corresponding buffer for passing the nonce to the kernel
        let mut nonce_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(1)
            .copy_host_slice(&nonce)
            .build()?;

        // establish a buffer for nonces that result in desired addresses
        let mut solutions: Vec<u64> = vec![0; 2];
        let solutions_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().write_only())
            .len(solutions.len())
            .copy_host_slice(&solutions)
            .build()?;

        // repeatedly enqueue kernel to search for new addresses
        loop {
            // build the kernel and define the type of each buffer
            let kern = ocl_pq
                .kernel_builder("hashMessage")
                .arg_named("message", None::<&Buffer<u8>>)
                .arg_named("nonce", None::<&Buffer<u32>>)
                .arg_named("solutions", None::<&Buffer<u64>>)
                .build()?;

            // set each buffer
            kern.set_arg("message", Some(&message_buffer))?;
            kern.set_arg("nonce", Some(&nonce_buffer))?;
            kern.set_arg("solutions", &solutions_buffer)?;

            // enqueue the kernel
            unsafe { kern.enq()? };

            // calculate the current time
            let mut now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            let current_time = now.as_secs() as f64;

            // we don't want to print too fast
            let print_output = current_time - previous_time > 0.99;
            previous_time = current_time;

            // clear the terminal screen
            if print_output {
                term.clear_screen()?;

                // get the total runtime and parse into hours : minutes : seconds
                let total_runtime = current_time - start_time;
                let total_runtime_hrs = total_runtime as u64 / 3600;
                let total_runtime_mins = (total_runtime as u64 - total_runtime_hrs * 3600) / 60;
                let total_runtime_secs = total_runtime
                    - (total_runtime_hrs * 3600) as f64
                    - (total_runtime_mins * 60) as f64;

                // determine the number of attempts being made per second
                let work_rate: u128 =
                    work_factor * cumulative_nonce as u128 * config.max_create3_nonce as u128;
                if total_runtime > 0.0 {
                    rate = 1.0 / total_runtime;
                }

                // fill the buffer for viewing the properly-formatted nonce
                LittleEndian::write_u64(&mut view_buf, (nonce[0] as u64) << 32);

                // calculate the terminal height, defaulting to a height of ten rows
                let height = terminal_size().map(|(_w, Height(h))| h).unwrap_or(10);

                // display information about the total runtime and work size
                term.write_line(&format!(
                    "total runtime: {}:{:02}:{:02} ({} cycles)\t\t\t\
                     work size per cycle: {}",
                    total_runtime_hrs,
                    total_runtime_mins,
                    total_runtime_secs,
                    cumulative_nonce,
                    work_size.separated_string(),
                ))?;

                // display information about the attempt rate and found solutions
                term.write_line(&format!(
                    "rate: {:.2} million attempts per second\t\t\t\
                     total found this run: {}",
                    work_rate as f64 * rate,
                    found
                ))?;

                // display information about the current search criteria
                term.write_line(&format!(
                    "current search space: {}xxxxxxxx{:08x}\t\t\
                     threshold: {:?} leading or {:?} total zeroes",
                    hex::encode(salt),
                    BigEndian::read_u64(&view_buf),
                    config.leading_zeroes_threshold,
                    config.total_zeroes_threshold
                ))?;

                // display recently found solutions based on terminal height
                let rows = if height < 5 { 1 } else { height as usize - 4 };
                let last_rows: Vec<String> = found_list.iter().cloned().rev().take(rows).collect();
                let ordered: Vec<String> = last_rows.iter().cloned().rev().collect();
                let recently_found = &ordered.join("\n");
                term.write_line(recently_found)?;
            }

            // increment the cumulative nonce (does not reset after a match)
            cumulative_nonce += 1;

            // record the start time of the work
            let work_start_time_millis = now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000;

            // sleep for 98% of the previous work duration to conserve CPU
            if work_duration_millis != 0 {
                std::thread::sleep(std::time::Duration::from_millis(
                    work_duration_millis * 980 / 1000,
                ));
            }

            // read the solutions from the device
            solutions_buffer.read(&mut solutions).enq()?;

            // record the end time of the work and compute how long the work took
            now = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
            work_duration_millis = (now.as_secs() * 1000 + now.subsec_nanos() as u64 / 1000000)
                - work_start_time_millis;

            // if at least one solution is found, end the loop
            if solutions[0] != 0 {
                break;
            }

            // if no solution has yet been found, increment the nonce
            nonce[0] += 1;

            // update the nonce buffer with the incremented nonce value
            nonce_buffer = Buffer::builder()
                .queue(ocl_pq.queue().clone())
                .flags(MemFlags::new().read_write())
                .len(1)
                .copy_host_slice(&nonce)
                .build()?;
        }

        // iterate over each solution, first converting to a fixed array

        if solutions[0] != 0 {
            let create1_nonce = solutions[1];
            let create2_nonce = solutions[0].to_le_bytes();
            let mut create2_salt = [0u8; 32];
            create2_salt[0..20].copy_from_slice(&config.owner[..]);
            create2_salt[20..24].copy_from_slice(&salt[..]);
            create2_salt[24..32].copy_from_slice(&create2_nonce);
            let deployer = config
                .factory
                .create2(&create2_salt, &config.init_code_hash);
            let address = deployer.create(create1_nonce.into());

            // count total and leading zero bytes
            let mut total = 0;
            let mut leading = 0;
            for (i, &b) in address.iter().enumerate() {
                if b == 0 {
                    total += 1;
                } else if leading == 0 {
                    // set leading on finding non-zero byte
                    leading = i;
                }
            }

            let key = leading * 20 + total;
            let reward = rewards.get(&key).unwrap_or("0");
            let output = format!(
                "0x{} ({}) => {} => {}",
                hex::encode(create2_salt),
                create1_nonce - 1,
                address,
                reward
            );

            let show = format!("{output} ({leading} / {total})");
            found_list.push(show.to_string());

            file.lock_exclusive().expect("Couldn't lock file.");

            writeln!(&file, "{output}")
                .unwrap_or_else(|_| panic!("Couldn't write to `{}` file.", config.output_file));

            file.unlock().expect("Couldn't unlock file.");
            found += 1;
        }
    }
}

#[track_caller]
fn output_file(path: &str) -> File {
    OpenOptions::new()
        .append(true)
        .create(true)
        .read(true)
        .open(path)
        .unwrap_or_else(|_| panic!("Could not create or open `{}` file.", path))
}

/// Creates the OpenCL kernel source code by populating the template with the
/// values from the Config object.
fn mk_kernel_src(config: &Config) -> String {
    let mut src = String::with_capacity(2048 + KERNEL_SRC.len());

    let factory = config.factory.iter();
    let owner = config.owner.iter();
    let hash = config.init_code_hash.iter();
    let hash = hash.enumerate().map(|(i, x)| (i + 52, x));
    for (i, x) in factory.chain(owner).enumerate().chain(hash) {
        writeln!(src, "#define S_{} {}u", i + 1, x).unwrap();
    }

    let lz = config.leading_zeroes_threshold.unwrap_or(0);
    writeln!(src, "#define LEADING_ZEROES {lz}").unwrap();
    let tz = config.total_zeroes_threshold.unwrap_or(0);
    writeln!(src, "#define TOTAL_ZEROES {tz}").unwrap();

    let condition = match (
        config.leading_zeroes_threshold,
        config.total_zeroes_threshold,
    ) {
        (Some(_), Some(_)) => "hasLeading(digest) || hasTotal(digest)",
        (Some(_), None) => "hasLeading(digest)",
        (None, Some(_)) => "hasTotal(digest)",
        (None, None) => unreachable!(),
    };
    writeln!(src, "#define SUCCESS_CONDITION() {}", condition).unwrap();

    writeln!(src, "#define MAX_NONCE {}u", config.max_create3_nonce).unwrap();

    src.push_str(KERNEL_SRC);

    src
}
