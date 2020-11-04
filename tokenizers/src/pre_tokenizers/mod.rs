pub mod bert;
#[cfg(not(feature = "bert"))]
pub mod byte_level;
#[cfg(not(feature = "bert"))]
pub mod delimiter;
#[cfg(not(feature = "bert"))]
pub mod digits;
#[cfg(not(feature = "bert"))]
pub mod metaspace;
#[cfg(not(feature = "bert"))]
pub mod punctuation;
#[cfg(not(feature = "bert"))]
pub mod sequence;
#[cfg(not(feature = "bert"))]
pub mod unicode_scripts;
#[cfg(not(feature = "bert"))]
pub mod whitespace;

use serde::{Deserialize, Serialize};

#[cfg(not(feature = "bert"))]
use crate::pre_tokenizers::{
    byte_level::ByteLevel,
    delimiter::CharDelimiterSplit,
    digits::Digits,
    metaspace::Metaspace,
    punctuation::Punctuation,
    sequence::Sequence,
    unicode_scripts::UnicodeScripts,
    whitespace::{Whitespace, WhitespaceSplit},
};
use crate::{pre_tokenizers::bert::BertPreTokenizer, PreTokenizedString, PreTokenizer};

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(untagged)]
pub enum PreTokenizerWrapper {
    BertPreTokenizer(BertPreTokenizer),
    #[cfg(not(feature = "bert"))]
    ByteLevel(ByteLevel),
    #[cfg(not(feature = "bert"))]
    Delimiter(CharDelimiterSplit),
    #[cfg(not(feature = "bert"))]
    Metaspace(Metaspace),
    #[cfg(not(feature = "bert"))]
    Whitespace(Whitespace),
    #[cfg(not(feature = "bert"))]
    Sequence(Sequence),
    #[cfg(not(feature = "bert"))]
    Punctuation(Punctuation),
    #[cfg(not(feature = "bert"))]
    WhitespaceSplit(WhitespaceSplit),
    #[cfg(not(feature = "bert"))]
    Digits(Digits),
    #[cfg(not(feature = "bert"))]
    UnicodeScripts(UnicodeScripts),
}

impl PreTokenizer for PreTokenizerWrapper {
    fn pre_tokenize(&self, normalized: &mut PreTokenizedString) -> crate::Result<()> {
        match self {
            PreTokenizerWrapper::BertPreTokenizer(bpt) => bpt.pre_tokenize(normalized),
            #[cfg(not(feature = "bert"))]
            PreTokenizerWrapper::ByteLevel(bpt) => bpt.pre_tokenize(normalized),
            #[cfg(not(feature = "bert"))]
            PreTokenizerWrapper::Delimiter(dpt) => dpt.pre_tokenize(normalized),
            #[cfg(not(feature = "bert"))]
            PreTokenizerWrapper::Metaspace(mspt) => mspt.pre_tokenize(normalized),
            #[cfg(not(feature = "bert"))]
            PreTokenizerWrapper::Whitespace(wspt) => wspt.pre_tokenize(normalized),
            #[cfg(not(feature = "bert"))]
            PreTokenizerWrapper::Punctuation(tok) => tok.pre_tokenize(normalized),
            #[cfg(not(feature = "bert"))]
            PreTokenizerWrapper::Sequence(tok) => tok.pre_tokenize(normalized),
            #[cfg(not(feature = "bert"))]
            PreTokenizerWrapper::WhitespaceSplit(wspt) => wspt.pre_tokenize(normalized),
            #[cfg(not(feature = "bert"))]
            PreTokenizerWrapper::Digits(wspt) => wspt.pre_tokenize(normalized),
            #[cfg(not(feature = "bert"))]
            PreTokenizerWrapper::UnicodeScripts(us) => us.pre_tokenize(normalized),
        }
    }
}

impl_enum_from!(BertPreTokenizer, PreTokenizerWrapper, BertPreTokenizer);
#[cfg(not(feature = "bert"))]
impl_enum_from!(ByteLevel, PreTokenizerWrapper, ByteLevel);
#[cfg(not(feature = "bert"))]
impl_enum_from!(CharDelimiterSplit, PreTokenizerWrapper, Delimiter);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Whitespace, PreTokenizerWrapper, Whitespace);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Punctuation, PreTokenizerWrapper, Punctuation);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Sequence, PreTokenizerWrapper, Sequence);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Metaspace, PreTokenizerWrapper, Metaspace);
#[cfg(not(feature = "bert"))]
impl_enum_from!(WhitespaceSplit, PreTokenizerWrapper, WhitespaceSplit);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Digits, PreTokenizerWrapper, Digits);
#[cfg(not(feature = "bert"))]
impl_enum_from!(UnicodeScripts, PreTokenizerWrapper, UnicodeScripts);
