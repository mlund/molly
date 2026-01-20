mod common;

use std::io::{self, Cursor};

use common::trajectories;
use molly::{Frame, XTCReader, XTCWriter};

fn assert_frames_eq(original: &[Frame], roundtrip: &[Frame]) {
    assert_eq!(
        original.len(),
        roundtrip.len(),
        "frame count mismatch: original {} vs roundtrip {}",
        original.len(),
        roundtrip.len()
    );

    for (i, (orig, rt)) in original.iter().zip(roundtrip.iter()).enumerate() {
        assert_eq!(orig.step, rt.step, "frame {i}: step mismatch");
        assert_eq!(orig.time, rt.time, "frame {i}: time mismatch");
        assert_eq!(orig.boxvec, rt.boxvec, "frame {i}: boxvec mismatch");
        assert_eq!(
            orig.positions.len(),
            rt.positions.len(),
            "frame {i}: position count mismatch"
        );

        for (j, (op, rp)) in orig.positions.iter().zip(rt.positions.iter()).enumerate() {
            let diff = (op - rp).abs();
            assert!(
                diff < 1e-5,
                "frame {i}, position {j}: value mismatch (original {op}, roundtrip {rp}, diff {diff})"
            );
        }
    }
}

macro_rules! roundtrip_test {
    ($name:ident, $path:expr) => {
        #[test]
        fn $name() -> io::Result<()> {
            let frames = XTCReader::open($path)?.read_all_frames()?;
            let mut buf = Vec::new();
            {
                let mut writer = XTCWriter::new(Cursor::new(&mut buf));
                for frame in frames.iter() {
                    writer.write_frame(frame)?;
                }
            }
            let roundtrip = XTCReader::new(Cursor::new(&buf)).read_all_frames()?;
            assert_frames_eq(&frames, &roundtrip);
            Ok(())
        }
    };
}

roundtrip_test!(roundtrip_adk, trajectories::ADK);
roundtrip_test!(roundtrip_aux, trajectories::AUX);
roundtrip_test!(roundtrip_cob, trajectories::COB);
roundtrip_test!(roundtrip_smol, trajectories::SMOL);
roundtrip_test!(roundtrip_ten, trajectories::TEN);
roundtrip_test!(roundtrip_xyz, trajectories::XYZ);
roundtrip_test!(roundtrip_delinyah, trajectories::DELINYAH);

fn encode_size(path: &str) -> io::Result<usize> {
    let frames = XTCReader::open(path)?.read_all_frames()?;
    let mut buf = Vec::new();
    {
        let mut writer = XTCWriter::new(std::io::Cursor::new(&mut buf));
        for frame in frames.iter() {
            writer.write_frame(frame)?;
        }
    }
    Ok(buf.len())
}

#[test]
#[ignore]
fn report_compressed_sizes() -> io::Result<()> {
    let inputs = [
        ("ADK", trajectories::ADK),
        ("AUX", trajectories::AUX),
        ("COB", trajectories::COB),
        ("SMOL", trajectories::SMOL),
        ("TEN", trajectories::TEN),
        ("XYZ", trajectories::XYZ),
        ("DELINYAH", trajectories::DELINYAH),
    ];
    for (name, path) in inputs {
        let size = encode_size(path)?;
        println!("{name}: {size} bytes");
    }
    Ok(())
}
