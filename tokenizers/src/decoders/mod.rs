#[cfg(not(feature = "bert"))]
pub mod bpe;
pub mod wordpiece;

// Re-export these as decoders
#[cfg(not(feature = "bert"))]
pub use super::{pre_tokenizers::byte_level, pre_tokenizers::metaspace};

use serde::{Deserialize, Serialize};

#[cfg(not(feature = "bert"))]
use crate::{
    decoders::bpe::BPEDecoder, pre_tokenizers::byte_level::ByteLevel,
    pre_tokenizers::metaspace::Metaspace,
};
use crate::{decoders::wordpiece::WordPiece, Decoder, Result};

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum DecoderWrapper {
    #[cfg(not(feature = "bert"))]
    BPE(BPEDecoder),
    #[cfg(not(feature = "bert"))]
    ByteLevel(ByteLevel),
    WordPiece(WordPiece),
    #[cfg(not(feature = "bert"))]
    Metaspace(Metaspace),
}

impl Decoder for DecoderWrapper {
    fn decode(&self, tokens: Vec<String>) -> Result<String> {
        match self {
            #[cfg(not(feature = "bert"))]
            DecoderWrapper::BPE(bpe) => bpe.decode(tokens),
            #[cfg(not(feature = "bert"))]
            DecoderWrapper::ByteLevel(bl) => bl.decode(tokens),
            #[cfg(not(feature = "bert"))]
            DecoderWrapper::Metaspace(ms) => ms.decode(tokens),
            DecoderWrapper::WordPiece(wp) => wp.decode(tokens),
        }
    }
}

#[cfg(not(feature = "bert"))]
impl_enum_from!(BPEDecoder, DecoderWrapper, BPE);
#[cfg(not(feature = "bert"))]
impl_enum_from!(ByteLevel, DecoderWrapper, ByteLevel);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Metaspace, DecoderWrapper, Metaspace);
impl_enum_from!(WordPiece, DecoderWrapper, WordPiece);
