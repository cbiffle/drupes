use anyhow::Context;
use jwalk::WalkDir;
use rayon::prelude::*;
use size::Size;
use std::io::Seek;
use std::{
    collections::{BTreeMap, HashMap},
    fs::File,
    io::{ErrorKind, Read},
    path::{Path, PathBuf},
    time::Instant,
};

const PREHASH_SIZE: usize = 4 * 1024;

/// PASS ONE
///
/// Traverse the requested parts of the filesystem, collating files by size
/// (i.e. producing a map with file sizes as keys, and lists of files as
/// values).
///
/// Any value in the map with more than one path represents a "file size
/// group," which is a potential duplicate group. On the other hand, any
/// value in the map containing only _one_ path need not be considered
/// further.
///
/// We do this because, generally speaking, getting the size of a file is
/// much cheaper than reading its contents, and in practice file sizes are
/// _relatively_ unique.
pub fn pass_one(
    roots: Vec<PathBuf>,
    verbose: bool,
    args_empty: bool,
    start: Instant,
) -> anyhow::Result<BTreeMap<u64, Vec<PathBuf>>, anyhow::Error> {
    let mut paths: BTreeMap<u64, Vec<PathBuf>> = BTreeMap::new();
    for root in &roots {
        if verbose {
            eprintln!("{:?} starting walk of {}", start.elapsed(), root.display());
        }

        for entry in WalkDir::new(root) {
            let entry =
                entry.with_context(|| format!("problem reading dirent in {}", root.display()))?;
            let meta = entry.metadata().with_context(|| {
                format!("problem getting metadata for {}", entry.path().display())
            })?;
            if meta.is_file() && (meta.len() > 0 || args_empty) {
                paths
                    .entry(meta.len())
                    .or_default()
                    .push(entry.path().to_owned());
            }
        }
    }
    Ok(paths)
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
pub fn pass_two(paths: &BTreeMap<u64, Vec<PathBuf>>) -> HashMap<blake3::Hash, Vec<&Path>> {
    paths
        .par_iter()
        // Flatten the map into a list of paths to hash, discarding the size
        // information.
        .flat_map(|(_size, paths)| paths)
        // Hash each path, producing a (path, hash) pair. Note that this can
        // fail to access the filesystem.
        //
        // We use `map_with` here to allocate exactly one I/O buffer per backing
        // Rayon thread, instead of one per closure, because I'm neurotic.
        .map_with(vec![0u8; PREHASH_SIZE], |buf, path| {
            let mut f =
                File::open(path).with_context(|| format!("unable to open: {}", path.display()))?;

            // Read up to `PREHASH_SIZE` bytes, or fewer if the file is shorter
            // than that. (It's odd that there's no operation for this in the
            // standard library.)
            let mut total = 0;
            while total < buf.len() {
                match f.read(&mut buf[total..]) {
                    Ok(0) => break,
                    Ok(n) => total += n,
                    Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                    Err(e) => {
                        return Err(e).context(format!("unable to read path: {}", path.display()))
                    }
                }
            }
            // Hash the first chunk of the file.
            Ok((blake3::hash(buf), path))
        })
        // Squawk about any reads that failed, and remove them from further
        // consideration.
        .filter_map(|result| match result {
            Ok(data) => Some(data),
            Err(e) => {
                eprintln!("{e:?}");
                None
            }
        })
        // Take the stream of (hash, path) pairs and collate them by hash,
        // producing "hash groups."
        //
        // Rayon's fold is a little surprising: this produces, not a single map,
        // but a _stream_ of maps, because (roughly speaking) each thread
        // calculates its own.
        //
        // Many hash-groups will only contain one path, and will be filtered out
        // below. Any group containing multiple paths needs to be hashed more
        // fully in the next pass.
        .fold(
            HashMap::<blake3::Hash, Vec<&Path>>::new,
            |mut map, (hash, path)| {
                map.entry(hash).or_default().push(path);
                map
            },
        )
        // Collapse the stream of hashmaps into one, merging hash groups as
        // required.
        .reduce(HashMap::new, |mut a, b| {
            for (k, v) in b {
                a.entry(k).or_default().extend(v);
            }
            a
        })
}

/// PASS THREE
///
/// For any files whose first `PREHASH_SIZE` bytes match at least one other
/// file, hash the entire contents to scan for differences later on.
pub fn pass_three(
    hashed_files: HashMap<blake3::Hash, Vec<&Path>>,
) -> HashMap<blake3::Hash, Vec<&Path>> {
    hashed_files
        .into_par_iter()
        // Ignore groups with only one member.
        .filter(|(_, paths)| paths.len() > 1)
        // Flatten the `prehash => vec of paths` map to a stream of `prehash,
        // path` pairs. Since the prehash has no (straightforward) relation to
        // the hash of the overall file, we don't need to maintain the group
        // structure.
        //
        // We do, however, forward the prehash value on, so that we can use it
        // for keying below.
        .flat_map(|(hash, paths)| paths.into_par_iter().map(move |p| (hash, p)))
        // Hash the tail of each file to produce `(path, hash)` pairs. Note that
        // this can fail to access the filesystem (again).
        //
        // This takes the prehash as input, and uses it as the key for a keyed
        // hash of the rest of the file. This is important for correctness: if
        // we just hashed the tail end of every file, we could detect two files
        // as "identical" even if their first `PREHASH_SIZE` bytes differed! By
        // incorporating the prehash as key we chain the two hashes and prevent
        // this.
        //
        // For files smaller than `PREHASH_SIZE`, we immediately finalize the
        // keyed hash without reading anything.
        .map(|(prehash, path)| {
            let mut f =
                File::open(path).with_context(|| format!("unable to open: {}", path.display()))?;
            let mut hasher = blake3::Hasher::new_keyed(prehash.as_bytes());

            // Small files have already been completely hashed. Skip them.
            if f.metadata()?.len() > PREHASH_SIZE as u64 {
                f.seek(std::io::SeekFrom::Start(PREHASH_SIZE as u64))?;
                hasher.update_reader(f)?;
            }
            Ok::<_, anyhow::Error>((hasher.finalize(), path))
        })
        // Squawk about any reads that failed, and remove them from further
        // consideration.
        .filter_map(|result| match result {
            Ok(data) => Some(data),
            Err(e) => {
                eprintln!("{e}");
                None
            }
        })
        // Collect groups of (path, hash) pairs and collate them by hash. This
        // is identical to the end of Pass Two.
        .fold(HashMap::<_, Vec<&Path>>::new, |mut map, (hash, path)| {
            map.entry(hash).or_default().push(path);
            map
        })
        // Collapse the stream of hashmaps into one, merging hash groups as
        // required. This is also identical to the end of Pass Two.
        .reduce(HashMap::new, |mut a, b| {
            for (k, v) in b {
                a.entry(k).or_default().extend(v);
            }
            a
        })
}

pub fn summarize(
    unique_prehash_groups: usize,
    paths: &BTreeMap<u64, Vec<PathBuf>>,
    hashed_files: &HashMap<blake3::Hash, Vec<&Path>>,
) -> anyhow::Result<()> {
    // Work out some statistics, instead of printing filenames.

    // How many unique size classes did we discover in the first pass?
    let unique_size_classes = paths.len();
    // How many files did we find in our recursive scan?
    let total_files_checked = paths.values().map(|v| v.len()).sum::<usize>();

    // How many hash-groups containing duplicates did we discover?
    let set_count = hashed_files
        .values()
        .filter(|files| files.len() > 1)
        .count();
    // And how many duplicates, beyond the first in each group, did we find?
    let dupe_count = hashed_files
        .values()
        .filter_map(|files| files.len().checked_sub(1))
        .sum::<usize>();
    // How large are the duplicates on disk?
    let dupe_size = hashed_files
        .values()
        .filter(|files| files.len() > 1)
        .try_fold(0, |sum, files| {
            std::fs::metadata(files[0]).map(|meta| sum + meta.len() * (files.len() as u64 - 1))
        })?;
    // Convenient unit formatting:
    let dupe_size = Size::from_bytes(dupe_size);

    println!(
        "{dupe_count} duplicate files (in {set_count} sets), \
            occupying {dupe_size}"
    );
    println!(
        "checked {total_files_checked} files in \
            {unique_size_classes} size classes"
    );
    println!("prehashing identified {unique_prehash_groups} groups");
    Ok(())
}
