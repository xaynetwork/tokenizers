//!
//! This module defines helpers to allow optional Rayon usage.
//!

// use rayon::iter::IterBridge;
// use rayon::prelude::*;
// use rayon_cond::CondIterator;

pub const ENV_VARIABLE: &str = "TOKENIZERS_PARALLELISM";

// Reading/Writing this variable should always happen on the main thread
static mut USED_PARALLELISM: bool = false;

/// Check if the TOKENIZERS_PARALLELISM env variable has been explicitly set
pub fn is_parallelism_configured() -> bool {
    std::env::var(ENV_VARIABLE).is_ok()
}

/// Check if at some point we used a parallel iterator
pub fn has_parallelism_been_used() -> bool {
    unsafe { USED_PARALLELISM }
}

/// Get the currently set value for `TOKENIZERS_PARALLELISM` env variable
pub fn get_parallelism() -> bool {
    match std::env::var(ENV_VARIABLE) {
        Ok(mut v) => {
            v.make_ascii_lowercase();
            !matches!(v.as_ref(), "" | "off" | "false" | "f" | "no" | "n" | "0")
        }
        Err(_) => true, // If we couldn't get the variable, we use the default
    }
}

/// Set the value for `TOKENIZERS_PARALLELISM` for the current process
pub fn set_parallelism(val: bool) {
    std::env::set_var(ENV_VARIABLE, if val { "true" } else { "false" })
}

/// Allows to convert into an iterator that can be executed either parallelly or serially.
///
/// The choice is made according to the currently set `TOKENIZERS_PARALLELISM` environment variable.
/// This variable can have one of the following values
///   - False => "" (empty value), "false", "f", "off", "no", "n", "0"
///   - True => Any other value
///
pub trait MaybeParallelIterator<T, S>
where
    S: Iterator<Item = T>,
{
    /// Convert ourself in a CondIterator, that will be executed either in parallel or serially,
    /// based solely on the `TOKENIZERS_PARALLELISM` environment variable
    fn into_maybe_par_iter(self) -> S;
    /// Convert ourself in a CondIterator, that will be executed either in parallel or serially,
    /// based on both the `TOKENIZERS_PARALLELISM` environment variable and the provided bool.
    /// Both must be true to run with parallelism activated.
    fn into_maybe_par_iter_cond(self, _cond: bool) -> S;
}

impl<T, S, I> MaybeParallelIterator<T, S> for I
where
    I: IntoIterator<IntoIter = S, Item = S::Item>,
    S: Iterator<Item = T>,
{
    fn into_maybe_par_iter(self) -> S {
        self.into_iter()
    }

    fn into_maybe_par_iter_cond(self, _cond: bool) -> S {
        self.into_maybe_par_iter()
    }
}

/// Shared reference version of MaybeParallelIterator, works the same but returns an iterator
/// over references, does not consume self
pub trait MaybeParallelRefIterator<'data, T, S>
where
    S: Iterator<Item = T>,
    T: 'data,
{
    fn maybe_par_iter(&'data self) -> S;
    fn maybe_par_iter_cond(&'data self, _cond: bool) -> S;
}

impl<'data, T, S, I: 'data + ?Sized> MaybeParallelRefIterator<'data, T, S> for I
where
    &'data I: MaybeParallelIterator<T, S>,
    S: Iterator<Item = T>,
    T: 'data,
{
    fn maybe_par_iter(&'data self) -> S {
        self.into_maybe_par_iter()
    }

    fn maybe_par_iter_cond(&'data self, cond: bool) -> S {
        self.into_maybe_par_iter_cond(cond)
    }
}

/// Exclusive reference version of MaybeParallelIterator, works the same but returns an iterator
/// over mutable references, does not consume self
pub trait MaybeParallelRefMutIterator<'data, T, S>
where
    S: Iterator<Item = T>,
    T: 'data,
{
    fn maybe_par_iter_mut(&'data mut self) -> S;
    fn maybe_par_iter_mut_cond(&'data mut self, cond: bool) -> S;
}

impl<'data, T, S, I: 'data + ?Sized> MaybeParallelRefMutIterator<'data, T, S> for I
where
    &'data mut I: MaybeParallelIterator<T, S>,
    S: Iterator<Item = T>,
    T: 'data,
{
    fn maybe_par_iter_mut(&'data mut self) -> S {
        self.into_maybe_par_iter()
    }

    fn maybe_par_iter_mut_cond(&'data mut self, cond: bool) -> S {
        self.into_maybe_par_iter_cond(cond)
    }
}

/// Converts any serial iterator into a CondIterator, that can either run parallelly or serially.
pub trait MaybeParallelBridge<T, S>
where
    S: Iterator<Item = T> + Send,
    T: Send,
{
    fn maybe_par_bridge(self) -> S;
    fn maybe_par_bridge_cond(self, cond: bool) -> S;
}

impl<T, S> MaybeParallelBridge<T, S> for S
where
    S: Iterator<Item = T> + Send,
    T: Send,
{
    fn maybe_par_bridge(self) -> S {
        self
    }

    fn maybe_par_bridge_cond(self, _cond: bool) -> S {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maybe_parallel_iterator() {
        let mut v = vec![1u32, 2, 3, 4, 5, 6];

        assert_eq!(v.maybe_par_iter().sum::<u32>(), 21);
        assert_eq!(
            v.maybe_par_iter_mut()
                .map(|v| {
                    *v *= 2;
                    *v
                })
                .sum::<u32>(),
            42
        );
        assert_eq!(v.maybe_par_iter().sum::<u32>(), 42);
        assert_eq!(v.into_maybe_par_iter().sum::<u32>(), 42);
    }
}
