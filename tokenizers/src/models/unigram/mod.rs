//! [Unigram](https://arxiv.org/abs/1804.10959) model.
mod lattice;
mod model;
mod serialization;
#[cfg(feature = "trainer")]
mod trainer;
mod trie;

pub use lattice::*;
pub use model::*;
#[cfg(feature = "trainer")]
pub use trainer::*;
