use std::io::{self, Write};

use crate::reader::{FIRSTIDX, MAGICINTS};
use crate::{padding, Magic};

/// XDR padding bytes.
const ZERO_PAD: [u8; 3] = [0; 3];

#[derive(Default)]
struct EncodeState {
    /// Number of pending bits in lastbyte (0-7).
    lastbits: usize,
    /// Pending bits waiting to be written (stored in low positions).
    lastbyte: u8,
}

/// Encode `nbits` bits from `value` into the buffer.
/// Bits are written MSB-first to match the decoder's expectations.
#[inline]
fn encodebits(buf: &mut Vec<u8>, state: &mut EncodeState, value: u32, nbits: usize) {
    if nbits == 0 {
        return;
    }

    // Combine pending bits with new bits.
    // state.lastbyte has state.lastbits pending bits in low positions.
    let total_bits = state.lastbits + nbits;
    let pending = ((state.lastbyte as u64) << nbits) | (value as u64);

    // Write complete bytes (MSB first).
    let mut remaining_bits = total_bits;
    while remaining_bits >= 8 {
        let shift = remaining_bits - 8;
        let byte = (pending >> shift) as u8;
        buf.push(byte);
        remaining_bits -= 8;
    }

    // Store leftover bits for next call.
    state.lastbits = remaining_bits;
    state.lastbyte = (pending & ((1u64 << remaining_bits) - 1)) as u8;
}

/// Encode a full byte, combining with pending bits.
#[inline]
fn encodebyte(buf: &mut Vec<u8>, state: &mut EncodeState, byte: u8) {
    encodebits(buf, state, byte as u32, 8);
}

/// Flush any remaining pending bits as a final byte.
fn flush_bits(buf: &mut Vec<u8>, state: &mut EncodeState) {
    if state.lastbits > 0 {
        // Pad the pending bits to fill a byte (MSB-aligned).
        buf.push(state.lastbyte << (8 - state.lastbits));
        state.lastbits = 0;
        state.lastbyte = 0;
    }
}

/// Pack three integers into a u32 using the size encoding.
/// Uses wrapping arithmetic since intermediate products may overflow.
#[inline]
const fn pack_into_u32(nums: [i32; 3], sizes: [u32; 3]) -> u32 {
    let sz = sizes[2];
    let szy = sizes[1].wrapping_mul(sz);
    (nums[0] as u32)
        .wrapping_mul(szy)
        .wrapping_add((nums[1] as u32).wrapping_mul(sz))
        .wrapping_add(nums[2] as u32)
}

/// Pack three integers into a u64 using the size encoding.
/// Uses wrapping arithmetic for consistency with u32 version.
#[inline]
const fn pack_into_u64(nums: [i32; 3], sizes: [u32; 3]) -> u64 {
    let sz = sizes[2] as u64;
    let szy = (sizes[1] as u64).wrapping_mul(sz);
    (nums[0] as u64)
        .wrapping_mul(szy)
        .wrapping_add((nums[1] as u64).wrapping_mul(sz))
        .wrapping_add(nums[2] as u64)
}

fn encodeints(
    buf: &mut Vec<u8>,
    state: &mut EncodeState,
    nbits: u32,
    sizes: [u32; 3],
    nums: [i32; 3],
) {
    if nbits <= 32 {
        let packed = pack_into_u32(nums, sizes);
        // Write bytes LSB first (matching how decodeints reads them).
        let mut nbytes = 0;
        let mut bits_left = nbits;
        while bits_left >= 8 {
            encodebyte(buf, state, (packed >> (8 * nbytes)) as u8);
            nbytes += 1;
            bits_left -= 8;
        }
        if bits_left > 0 {
            encodebits(
                buf,
                state,
                (packed >> (8 * nbytes)) & ((1 << bits_left) - 1),
                bits_left as usize,
            );
        }
        return;
    }

    if nbits <= 64 {
        let packed = pack_into_u64(nums, sizes);
        let mut nbytes = 0;
        let mut bits_left = nbits;
        while bits_left >= 8 {
            encodebyte(buf, state, (packed >> (8 * nbytes)) as u8);
            nbytes += 1;
            bits_left -= 8;
        }
        if bits_left > 0 {
            encodebits(
                buf,
                state,
                ((packed >> (8 * nbytes)) & ((1 << bits_left) - 1)) as u32,
                bits_left as usize,
            );
        }
        return;
    }

    // For very large nbits, use the byte array method (inverse of decodeints).
    let mut bytes = [0u8; 32];
    let mut nbytes: usize = 0;

    // Pack nums[2], nums[1], nums[0] into bytes (reverse order of unpacking).
    let mut carry = nums[2] as u32;
    for (i, byte) in bytes.iter_mut().enumerate() {
        *byte = (carry & 0xff) as u8;
        carry >>= 8;
        if carry == 0 && i >= nbytes {
            nbytes = i + 1;
            break;
        }
    }

    // Multiply by sizes[2] and add nums[1].
    let mut temp_carry = 0u64;
    for byte in bytes.iter_mut().take(nbytes) {
        temp_carry += *byte as u64 * sizes[2] as u64;
        *byte = (temp_carry & 0xff) as u8;
        temp_carry >>= 8;
    }
    while temp_carry > 0 {
        bytes[nbytes] = (temp_carry & 0xff) as u8;
        temp_carry >>= 8;
        nbytes += 1;
    }

    // Add nums[1].
    temp_carry = nums[1] as u64;
    for byte in bytes.iter_mut().take(nbytes.max(4)) {
        temp_carry += *byte as u64;
        *byte = (temp_carry & 0xff) as u8;
        temp_carry >>= 8;
    }
    while temp_carry > 0 {
        bytes[nbytes] = (temp_carry & 0xff) as u8;
        temp_carry >>= 8;
        nbytes += 1;
    }

    // Multiply by sizes[1] and add nums[0].
    temp_carry = 0;
    for byte in bytes.iter_mut().take(nbytes) {
        temp_carry += *byte as u64 * sizes[1] as u64;
        *byte = (temp_carry & 0xff) as u8;
        temp_carry >>= 8;
    }
    while temp_carry > 0 {
        bytes[nbytes] = (temp_carry & 0xff) as u8;
        temp_carry >>= 8;
        nbytes += 1;
    }

    temp_carry = nums[0] as u64;
    for byte in bytes.iter_mut().take(nbytes.max(4)) {
        temp_carry += *byte as u64;
        *byte = (temp_carry & 0xff) as u8;
        temp_carry >>= 8;
    }
    while temp_carry > 0 {
        bytes[nbytes] = (temp_carry & 0xff) as u8;
        temp_carry >>= 8;
        nbytes += 1;
    }

    // Now write the bytes.
    let mut bits_left = nbits;
    let mut byte_idx = 0;
    while bits_left >= 8 {
        encodebyte(buf, state, bytes[byte_idx]);
        byte_idx += 1;
        bits_left -= 8;
    }
    if bits_left > 0 {
        encodebits(
            buf,
            state,
            bytes[byte_idx] as u32 & ((1 << bits_left) - 1),
            bits_left as usize,
        );
    }
}

/// Returns the number of bits needed to represent `size`.
#[inline]
const fn sizeofint(size: u32) -> u32 {
    if size == 0 {
        0
    } else {
        u32::BITS - size.leading_zeros()
    }
}

fn sizeofints(sizes: [u32; 3]) -> u32 {
    let mut nbytes = 1usize;
    let mut bytes = [0u8; 32];
    bytes[0] = 1;
    let mut nbits = 0;

    for size in sizes {
        let mut tmp = 0u32;
        let mut bytecount = 0;
        while bytecount < nbytes {
            tmp += bytes[bytecount] as u32 * size;
            bytes[bytecount] = (tmp & 0xff) as u8;
            tmp >>= 8;
            bytecount += 1;
        }
        while tmp != 0 {
            bytes[bytecount] = (tmp & 0xff) as u8;
            bytecount += 1;
            tmp >>= 8;
        }
        nbytes = bytecount;
    }

    nbytes -= 1;
    let mut num = 1u32;
    while bytes[nbytes] as u32 >= num {
        nbits += 1;
        num *= 2;
    }

    nbytes as u32 * 8 + nbits
}

fn calc_sizeint(
    minint: [i32; 3],
    maxint: [i32; 3],
    sizeint: &mut [u32; 3],
    bitsizeint: &mut [u32; 3],
) -> u32 {
    for ((size, &min), &max) in sizeint.iter_mut().zip(&minint).zip(&maxint) {
        *size = (max - min) as u32 + 1;
    }
    bitsizeint.fill(0);

    // Check if one of the sizes is too big to be multiplied.
    if sizeint.iter().any(|&s| s > 0x00ff_ffff) {
        for (bits, &size) in bitsizeint.iter_mut().zip(sizeint.iter()) {
            *bits = sizeofint(size);
        }
        return 0; // Flags use of large sizes.
    }

    sizeofints(*sizeint)
}

/// Write compressed positions to the writer.
/// Returns the number of compressed bytes written.
///
/// # Errors
/// Returns an error if writing to the underlying writer fails.
///
/// # Panics
/// Panics if `positions.len()` is not divisible by 3.
pub fn write_compressed_positions<W: Write>(
    writer: &mut W,
    positions: &[f32],
    precision: f32,
    magic: Magic,
) -> io::Result<usize> {
    assert_eq!(positions.len() % 3, 0);

    let to_int = |f: f32| (f * precision).round() as i32;
    let mut int_coords: Vec<[i32; 3]> = positions
        .chunks_exact(3)
        .map(|p| [to_int(p[0]), to_int(p[1]), to_int(p[2])])
        .collect();

    let (minint, maxint) = calc_bounds(&int_coords);
    let smallidx = find_initial_smallidx(&int_coords);

    write_prelude(writer, &minint, &maxint, smallidx)?;

    let mut sizeint = [0u32; 3];
    let mut bitsizeint = [0u32; 3];
    let bitsize = calc_sizeint(minint, maxint, &mut sizeint, &mut bitsizeint);

    let mut compressed = Vec::new();
    let mut state = EncodeState::default();

    encode_coordinates(
        &mut compressed,
        &mut state,
        &mut int_coords,
        minint,
        bitsize,
        &sizeint,
        &bitsizeint,
        smallidx,
    );

    flush_bits(&mut compressed, &mut state);
    write_compressed_data(writer, &compressed, magic)?;

    Ok(compressed.len())
}

/// Encode coordinates with run-length compression and water swap.
#[allow(clippy::too_many_arguments)]
fn encode_coordinates(
    buf: &mut Vec<u8>,
    state: &mut EncodeState,
    coords: &mut [[i32; 3]],
    minint: [i32; 3],
    bitsize: u32,
    sizeint: &[u32; 3],
    bitsizeint: &[u32; 3],
    mut smallidx: usize,
) {
    let mut idx = 0usize;
    let mut prevrun: i32 = -1;
    let lastidx = MAGICINTS.len().saturating_sub(1);
    let maxidx = lastidx.min(smallidx + 8);
    let minidx = maxidx.saturating_sub(8);

    let mut smaller = MAGICINTS[smallidx.saturating_sub(1).max(FIRSTIDX)] / 2;
    let mut small = MAGICINTS[smallidx] / 2;
    let mut sizesmall = [MAGICINTS[smallidx] as u32; 3];
    let larger = MAGICINTS[maxidx] / 2;
    let mut prevcoord = [0; 3];

    while idx < coords.len() {
        let mut is_small = false;
        let mut is_smaller = if idx >= 1 {
            if smallidx < maxidx
                && (coords[idx][0] - prevcoord[0]).abs() < larger
                && (coords[idx][1] - prevcoord[1]).abs() < larger
                && (coords[idx][2] - prevcoord[2]).abs() < larger
            {
                1
            } else if smallidx > minidx {
                -1
            } else {
                0
            }
        } else {
            0
        };

        if idx + 1 < coords.len()
            && (coords[idx][0] - coords[idx + 1][0]).abs() < small
            && (coords[idx][1] - coords[idx + 1][1]).abs() < small
            && (coords[idx][2] - coords[idx + 1][2]).abs() < small
        {
            coords.swap(idx, idx + 1);
            is_small = true;
        }

        let coord = coords[idx];
        encode_full_coord(buf, state, coord, minint, bitsize, sizeint, bitsizeint);
        prevcoord = coord;
        idx += 1;

        let mut run: i32 = 0;
        if !is_small && is_smaller == -1 {
            is_smaller = 0;
        }

        let mut tmpcoord = [0i32; 24];
        while is_small && run < 8 * 3 && idx < coords.len() {
            let next = coords[idx];
            if is_smaller == -1 {
                let dx = (next[0] - prevcoord[0]) as i64;
                let dy = (next[1] - prevcoord[1]) as i64;
                let dz = (next[2] - prevcoord[2]) as i64;
                let dsq = dx * dx + dy * dy + dz * dz;
                let threshold = (smaller as i64) * (smaller as i64);
                if dsq >= threshold {
                    is_smaller = 0;
                }
            }

            tmpcoord[run as usize] = next[0] - prevcoord[0] + small;
            tmpcoord[run as usize + 1] = next[1] - prevcoord[1] + small;
            tmpcoord[run as usize + 2] = next[2] - prevcoord[2] + small;
            run += 3;
            prevcoord = next;
            idx += 1;

            is_small = false;
            if idx < coords.len()
                && (coords[idx][0] - prevcoord[0]).abs() < small
                && (coords[idx][1] - prevcoord[1]).abs() < small
                && (coords[idx][2] - prevcoord[2]).abs() < small
            {
                is_small = true;
            }
        }

        if run != prevrun || is_smaller != 0 {
            prevrun = run;
            encodebits(buf, state, 1, 1);
            let run_value = (run + is_smaller + 1) as u32;
            encodebits(buf, state, run_value, 5);
        } else {
            encodebits(buf, state, 0, 1);
        }

        let mut k = 0;
        while k < run {
            let chunk = [
                tmpcoord[k as usize],
                tmpcoord[k as usize + 1],
                tmpcoord[k as usize + 2],
            ];
            encodeints(buf, state, smallidx as u32, sizesmall, chunk);
            k += 3;
        }

        if is_smaller != 0 {
            if is_smaller < 0 {
                smallidx = smallidx.saturating_sub(1);
                small = smaller;
                smaller = if smallidx > FIRSTIDX {
                    MAGICINTS[smallidx - 1] / 2
                } else {
                    0
                };
            } else {
                smallidx = (smallidx + 1).min(lastidx);
                smaller = small;
                small = MAGICINTS[smallidx] / 2;
            }
            sizesmall.fill(MAGICINTS[smallidx] as u32);
        }
    }
}

fn calc_bounds(int_coords: &[[i32; 3]]) -> ([i32; 3], [i32; 3]) {
    int_coords.iter().fold(
        ([i32::MAX; 3], [i32::MIN; 3]),
        |(mut min, mut max), coord| {
            for (i, &c) in coord.iter().enumerate() {
                min[i] = min[i].min(c);
                max[i] = max[i].max(c);
            }
            (min, max)
        },
    )
}

/// Find initial smallidx based on minimum delta between adjacent coordinates.
/// This matches the original C implementation which uses mindiff (sum of absolute
/// differences between consecutive coordinate triplets).
fn find_initial_smallidx(int_coords: &[[i32; 3]]) -> usize {
    let mindiff = int_coords
        .windows(2)
        .map(|w| {
            (w[0][0] - w[1][0]).abs() + (w[0][1] - w[1][1]).abs() + (w[0][2] - w[1][2]).abs()
        })
        .min()
        .unwrap_or(0);

    // Find first index where MAGICINTS[i] >= mindiff.
    MAGICINTS[FIRSTIDX..]
        .iter()
        .position(|&m| m >= mindiff)
        .map_or(MAGICINTS.len() - 1, |pos| FIRSTIDX + pos)
}

fn write_prelude<W: Write>(
    writer: &mut W,
    minint: &[i32; 3],
    maxint: &[i32; 3],
    smallidx: usize,
) -> io::Result<()> {
    for &v in minint.iter().chain(maxint) {
        writer.write_all(&v.to_be_bytes())?;
    }
    writer.write_all(&(smallidx as u32).to_be_bytes())
}

fn encode_full_coord(
    buf: &mut Vec<u8>,
    state: &mut EncodeState,
    coord: [i32; 3],
    minint: [i32; 3],
    bitsize: u32,
    sizeint: &[u32; 3],
    bitsizeint: &[u32; 3],
) {
    let relative = [
        coord[0] - minint[0],
        coord[1] - minint[1],
        coord[2] - minint[2],
    ];
    if bitsize == 0 {
        encodebits(buf, state, relative[0] as u32, bitsizeint[0] as usize);
        encodebits(buf, state, relative[1] as u32, bitsizeint[1] as usize);
        encodebits(buf, state, relative[2] as u32, bitsizeint[2] as usize);
    } else {
        encodeints(buf, state, bitsize, *sizeint, relative);
    }
}

fn write_compressed_data<W: Write>(
    writer: &mut W,
    compressed: &[u8],
    magic: Magic,
) -> io::Result<()> {
    let nbytes = compressed.len();
    match magic {
        Magic::Xtc1995 => writer.write_all(&(nbytes as u32).to_be_bytes())?,
        Magic::Xtc2023 => writer.write_all(&(nbytes as u64).to_be_bytes())?,
    }
    writer.write_all(compressed)?;
    let pad = padding(nbytes);
    if pad > 0 {
        writer.write_all(&ZERO_PAD[..pad])?;
    }
    Ok(())
}
