// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{BufReader, Read, Seek},
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::bail;
use clap::Parser;
use drupes::pass_one;
use drupes::pass_three;
use drupes::pass_two;
use drupes::summarize;
use rayon::prelude::*;

/// Finds duplicate files and optionally deletes them.
///
/// This program recursively analyzes one or more paths and tries to find files
/// that appear in multiple places, possibly with different names, but have the
/// exact same content. This can happen, for example, if you restore a
/// collection of backups from different dates, which is the case that motivated
/// the author.
#[derive(Parser)]
struct Drupes {
    /// Also consider empty files, which will report all empty files except one
    /// as duplicate; by default, empty files are ignored, because this is
    /// rarely what you actually want.
    #[clap(short, long)]
    empty: bool,

    /// Don't print the first filename in a set of duplicates, so that all the
    /// printed filenames are files to consider removing.
    #[clap(short('f'), long)]
    omit_first: bool,

    /// Instead of listing duplicates, print a summary of what was found.
    #[clap(short('m'), long)]
    summarize: bool,

    /// Engages "paranoid mode" and performs byte-for-byte comparisons of files,
    /// in case you've found the first real-world BLAKE3 hash collision (please
    /// publish it if so)
    #[clap(short, long)]
    paranoid: bool,

    /// Try to delete all duplicates but one, skipping any files that cannot be
    /// deleted for whatever reason.
    #[clap(long)]
    delete: bool,

    /// Enable additional output about what the program is doing.
    #[clap(short, long)]
    verbose: bool,

    /// List of directories to search, recursively, for duplicate files; if
    /// omitted, the current directory is searched.
    roots: Vec<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let start = Instant::now();

    let mut args = Drupes::parse();

    if args.roots.is_empty() {
        // Search the current directory by default.
        args.roots.push(".".into());
    }

    let mut paths: BTreeMap<u64, Vec<PathBuf>> =
        pass_one(args.roots, args.verbose, args.empty, start)?;

    if args.verbose {
        eprintln!(
            "{:?} pass one complete, found {} size-groups",
            start.elapsed(),
            paths.len()
        );
    }

    // Drop all file size groups that contain no duplicates (have only one
    // member).
    //
    // This saves about 10% of runtime.
    paths.retain(|_size, paths| paths.len() > 1);

    if args.verbose {
        eprintln!("...of which {} had more than one member", paths.len());
    }

    // PASS TWO
    //
    // We've reduced the data set to files whose sizes are not unique. This pass
    // takes those files and hashes the first `PREHASH_SIZE` bytes of each. If
    // two files have different hashes for the first `PREHASH_SIZE` bytes, they
    // cannot possibly be duplicates, so we can use this to avoid reading the
    // full contents of files.
    //
    // This is a significant performance improvement for directories of large
    // files like photos or videos (~50%).
    //
    // This is constructed as a Rayon pipeline because (1) I find it reasonably
    // clear this way once I got used to it and (2) it's by far the
    // easiest-to-reach "go faster button."
    let hashed_files: HashMap<blake3::Hash, Vec<&Path>> = pass_two(&paths);

    let unique_prehash_groups = hashed_files.len();

    if args.verbose {
        eprintln!(
            "{:?} pass two complete, found {unique_prehash_groups} \
            unique first blocks",
            start.elapsed()
        );
        let dupesets = hashed_files
            .values()
            .filter(|paths| paths.len() > 1)
            .count();
        eprintln!("...of which {dupesets} are present in more than one file");
        let dupes = hashed_files
            .values()
            .map(|paths| paths.len().saturating_sub(1))
            .sum::<usize>();
        eprintln!("...for a total of {dupes} possibly redundant files");
    }

    // PASS THREE
    //
    // For any files whose first `PREHASH_SIZE` bytes match at least one other
    // file, hash the entire contents to scan for differences later on.
    let mut hashed_files = pass_three(hashed_files);
    if args.verbose {
        eprintln!(
            "{:?} pass three complete, generating results",
            start.elapsed()
        );
    }

    if args.paranoid {
        // Given our map of collated hash-groups from the previous step, let's
        // check our work.
        //
        // This takes each hash-group containing at least two paths and reads
        // the contents of each file, comparing them to one another. The files
        // are not kept in memory, so this works fine on very large files
        // (keeping files in memory is the operating system's job).
        //
        // Note that if this ever finds anything, it is **almost certainly** a
        // bug in this program. If it isn't a bug in this program, it's probably
        // a file being modified out from under us. BLAKE3 is
        // collision-resistant, and finding two files with the same length, same
        // BLAKE3 hash, and different contents would be a newsworthy event. It's
        // certainly possible, but rather unlikely.
        eprintln!("paranoid mode: verifying file contents");
        hashed_files
            .par_iter()
            .filter(|(_, files)| files.len() > 1)
            .try_for_each(|(_, files)| {
                // Arbitrarily choose the first file in each group as a
                // "representative."
                let first = &files[0];
                let first_f = File::open(first)?;
                let first_meta = first_f.metadata()?;
                let mut first_f = BufReader::new(first_f);

                // Compare it to every other file in the group, one at a time.
                for other in &files[1..] {
                    // ...starting from the beginning of the first file, please.
                    first_f.rewind()?;

                    let other_f = File::open(other)?;
                    let other_meta = other_f.metadata()?;
                    let mut other_f = BufReader::new(other_f);

                    // This provides some _very basic_ protection against files
                    // being modified while this program is running, but in
                    // general, this program is not written with that situation
                    // in mind.
                    if first_meta.len() != other_meta.len() {
                        bail!(
                            "files no longer have same length:\n{}\n{}",
                            first.display(),
                            other.display()
                        );
                    }

                    // Read one byte at a time from each file, comparing each
                    // byte. Single byte reads are the easiest thing to
                    // implement, and are reasonably fast because BufReader
                    // converts them into larger reads under the hood. No need
                    // to reimplement the standard library!
                    let mut buf1 = [0u8];
                    let mut buf2 = [0u8];
                    for _ in 0..first_meta.len() {
                        first_f.read_exact(&mut buf1)?;
                        other_f.read_exact(&mut buf2)?;
                        if buf1 != buf2 {
                            bail!(
                                "files differ (blake3 collision found?):\n{}\n{}",
                                first.display(),
                                other.display()
                            );
                        }
                    }
                }
                Ok(())
            })?;
        eprintln!("files really are duplicates");
    }

    if args.summarize {
        summarize(unique_prehash_groups, &paths, &hashed_files)?;
    } else {
        // Print filenames of each duplicate-group.
        for files in hashed_files.values_mut() {
            if files.len() > 1 {
                // Our files have arrived in a nondeterministic order due to our
                // use of concurrency. Let's fix that.
                files.sort();

                let mut files = files.iter();
                // Implement the omit-first flag by skipping:
                if args.omit_first {
                    files.next();
                }

                for f in files {
                    println!("{}", f.display());
                }
                if !args.omit_first {
                    println!();
                }
            }
        }
    }

    if args.delete {
        // The scary delete mode!
        for files in hashed_files.values() {
            if files.len() > 1 {
                for f in &files[1..] {
                    println!("deleting: {}", f.display());
                    if let Err(e) = std::fs::remove_file(f) {
                        eprintln!("error deleting {}: {e}", f.display());
                    }
                }
            }
        }
    }

    Ok(())
}
