pub mod bert;
#[cfg(not(feature = "bert"))]
pub mod precompiled;
#[cfg(not(feature = "bert"))]
pub mod replace;
#[cfg(not(feature = "bert"))]
pub mod strip;
#[cfg(not(feature = "bert"))]
pub mod unicode;
#[cfg(not(feature = "bert"))]
pub mod utils;

pub use crate::normalizers::bert::BertNormalizer;
#[cfg(not(feature = "bert"))]
pub use crate::normalizers::{
    precompiled::Precompiled,
    replace::Replace,
    strip::{Strip, StripAccents},
    unicode::{Nmt, NFC, NFD, NFKC, NFKD},
    utils::{Lowercase, Sequence},
};

use serde::{Deserialize, Serialize};

use crate::{NormalizedString, Normalizer};

/// Wrapper for known Normalizers.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum NormalizerWrapper {
    BertNormalizer(BertNormalizer),
    #[cfg(not(feature = "bert"))]
    StripNormalizer(Strip),
    #[cfg(not(feature = "bert"))]
    StripAccents(StripAccents),
    #[cfg(not(feature = "bert"))]
    NFC(NFC),
    #[cfg(not(feature = "bert"))]
    NFD(NFD),
    #[cfg(not(feature = "bert"))]
    NFKC(NFKC),
    #[cfg(not(feature = "bert"))]
    NFKD(NFKD),
    #[cfg(not(feature = "bert"))]
    Sequence(Sequence),
    #[cfg(not(feature = "bert"))]
    Lowercase(Lowercase),
    #[cfg(not(feature = "bert"))]
    Nmt(Nmt),
    #[cfg(not(feature = "bert"))]
    Precompiled(Precompiled),
    #[cfg(not(feature = "bert"))]
    Replace(Replace),
}

impl Normalizer for NormalizerWrapper {
    fn normalize(&self, normalized: &mut NormalizedString) -> crate::Result<()> {
        match self {
            NormalizerWrapper::BertNormalizer(bn) => bn.normalize(normalized),
            #[cfg(not(feature = "bert"))]
            NormalizerWrapper::StripNormalizer(sn) => sn.normalize(normalized),
            #[cfg(not(feature = "bert"))]
            NormalizerWrapper::StripAccents(sn) => sn.normalize(normalized),
            #[cfg(not(feature = "bert"))]
            NormalizerWrapper::NFC(nfc) => nfc.normalize(normalized),
            #[cfg(not(feature = "bert"))]
            NormalizerWrapper::NFD(nfd) => nfd.normalize(normalized),
            #[cfg(not(feature = "bert"))]
            NormalizerWrapper::NFKC(nfkc) => nfkc.normalize(normalized),
            #[cfg(not(feature = "bert"))]
            NormalizerWrapper::NFKD(nfkd) => nfkd.normalize(normalized),
            #[cfg(not(feature = "bert"))]
            NormalizerWrapper::Sequence(sequence) => sequence.normalize(normalized),
            #[cfg(not(feature = "bert"))]
            NormalizerWrapper::Lowercase(lc) => lc.normalize(normalized),
            #[cfg(not(feature = "bert"))]
            NormalizerWrapper::Nmt(lc) => lc.normalize(normalized),
            #[cfg(not(feature = "bert"))]
            NormalizerWrapper::Precompiled(lc) => lc.normalize(normalized),
            #[cfg(not(feature = "bert"))]
            NormalizerWrapper::Replace(lc) => lc.normalize(normalized),
        }
    }
}

impl_enum_from!(BertNormalizer, NormalizerWrapper, BertNormalizer);
#[cfg(not(feature = "bert"))]
impl_enum_from!(NFKD, NormalizerWrapper, NFKD);
#[cfg(not(feature = "bert"))]
impl_enum_from!(NFKC, NormalizerWrapper, NFKC);
#[cfg(not(feature = "bert"))]
impl_enum_from!(NFC, NormalizerWrapper, NFC);
#[cfg(not(feature = "bert"))]
impl_enum_from!(NFD, NormalizerWrapper, NFD);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Strip, NormalizerWrapper, StripNormalizer);
#[cfg(not(feature = "bert"))]
impl_enum_from!(StripAccents, NormalizerWrapper, StripAccents);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Sequence, NormalizerWrapper, Sequence);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Lowercase, NormalizerWrapper, Lowercase);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Nmt, NormalizerWrapper, Nmt);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Precompiled, NormalizerWrapper, Precompiled);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Replace, NormalizerWrapper, Replace);
