//! Popular tokenizer models.

#[cfg(not(feature = "bert"))]
pub mod bpe;
#[cfg(not(feature = "bert"))]
pub mod unigram;
#[cfg(not(feature = "bert"))]
pub mod wordlevel;
pub mod wordpiece;

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize, Serializer};

use crate::{models::wordpiece::WordPiece, Model, Result, Token};
#[cfg(not(feature = "bert"))]
use crate::{
    models::{
        bpe::{BpeTrainer, BPE},
        unigram::{Unigram, UnigramTrainer},
        wordlevel::WordLevel,
        wordpiece::WordPieceTrainer,
    },
    AddedToken, Trainer,
};

/// Wraps a vocab mapping (ID -> token) to a struct that will be serialized in order
/// of token ID, smallest to largest.
struct OrderedVocabIter<'a> {
    vocab_r: &'a HashMap<u32, String>,
}

impl<'a> OrderedVocabIter<'a> {
    fn new(vocab_r: &'a HashMap<u32, String>) -> Self {
        Self { vocab_r }
    }
}

impl<'a> Serialize for OrderedVocabIter<'a> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let iter = (0u32..(self.vocab_r.len() as u32)).map(|i| (&self.vocab_r[&i], i));
        serializer.collect_map(iter)
    }
}

#[derive(Deserialize, Serialize, Debug, PartialEq, Clone)]
#[serde(untagged)]
pub enum ModelWrapper {
    WordPiece(WordPiece),
    #[cfg(not(feature = "bert"))]
    BPE(BPE),
    #[cfg(not(feature = "bert"))]
    WordLevel(WordLevel),
    #[cfg(not(feature = "bert"))]
    Unigram(Unigram),
}

#[cfg(not(feature = "bert"))]
impl_enum_from!(WordLevel, ModelWrapper, WordLevel);
impl_enum_from!(WordPiece, ModelWrapper, WordPiece);
#[cfg(not(feature = "bert"))]
impl_enum_from!(BPE, ModelWrapper, BPE);
#[cfg(not(feature = "bert"))]
impl_enum_from!(Unigram, ModelWrapper, Unigram);

impl Model for ModelWrapper {
    fn tokenize(&self, tokens: &str) -> Result<Vec<Token>> {
        use ModelWrapper::*;
        match self {
            #[cfg(not(feature = "bert"))]
            WordLevel(t) => t.tokenize(tokens),
            WordPiece(t) => t.tokenize(tokens),
            #[cfg(not(feature = "bert"))]
            BPE(t) => t.tokenize(tokens),
            #[cfg(not(feature = "bert"))]
            Unigram(t) => t.tokenize(tokens),
        }
    }

    fn token_to_id(&self, token: &str) -> Option<u32> {
        use ModelWrapper::*;
        match self {
            #[cfg(not(feature = "bert"))]
            WordLevel(t) => t.token_to_id(token),
            WordPiece(t) => t.token_to_id(token),
            #[cfg(not(feature = "bert"))]
            BPE(t) => t.token_to_id(token),
            #[cfg(not(feature = "bert"))]
            Unigram(t) => t.token_to_id(token),
        }
    }

    fn id_to_token(&self, id: u32) -> Option<&str> {
        use ModelWrapper::*;
        match self {
            #[cfg(not(feature = "bert"))]
            WordLevel(t) => t.id_to_token(id),
            WordPiece(t) => t.id_to_token(id),
            #[cfg(not(feature = "bert"))]
            BPE(t) => t.id_to_token(id),
            #[cfg(not(feature = "bert"))]
            Unigram(t) => t.id_to_token(id),
        }
    }

    fn get_vocab(&self) -> &HashMap<String, u32> {
        use ModelWrapper::*;
        match self {
            #[cfg(not(feature = "bert"))]
            WordLevel(t) => t.get_vocab(),
            WordPiece(t) => t.get_vocab(),
            #[cfg(not(feature = "bert"))]
            BPE(t) => t.get_vocab(),
            #[cfg(not(feature = "bert"))]
            Unigram(t) => t.get_vocab(),
        }
    }

    fn get_vocab_size(&self) -> usize {
        use ModelWrapper::*;
        match self {
            #[cfg(not(feature = "bert"))]
            WordLevel(t) => t.get_vocab_size(),
            WordPiece(t) => t.get_vocab_size(),
            #[cfg(not(feature = "bert"))]
            BPE(t) => t.get_vocab_size(),
            #[cfg(not(feature = "bert"))]
            Unigram(t) => t.get_vocab_size(),
        }
    }

    fn save(&self, folder: &Path, name: Option<&str>) -> Result<Vec<PathBuf>> {
        use ModelWrapper::*;
        match self {
            #[cfg(not(feature = "bert"))]
            WordLevel(t) => t.save(folder, name),
            WordPiece(t) => t.save(folder, name),
            #[cfg(not(feature = "bert"))]
            BPE(t) => t.save(folder, name),
            #[cfg(not(feature = "bert"))]
            Unigram(t) => t.save(folder, name),
        }
    }
}

#[cfg(not(feature = "bert"))]
pub enum TrainerWrapper {
    BpeTrainer(BpeTrainer),
    WordPieceTrainer(WordPieceTrainer),
    UnigramTrainer(UnigramTrainer),
}

#[cfg(not(feature = "bert"))]
impl Trainer for TrainerWrapper {
    type Model = ModelWrapper;

    fn should_show_progress(&self) -> bool {
        match self {
            TrainerWrapper::BpeTrainer(bpe) => bpe.should_show_progress(),
            TrainerWrapper::WordPieceTrainer(wpt) => wpt.should_show_progress(),
            TrainerWrapper::UnigramTrainer(wpt) => wpt.should_show_progress(),
        }
    }

    fn train(&self, words: HashMap<String, u32>) -> Result<(Self::Model, Vec<AddedToken>)> {
        match self {
            TrainerWrapper::BpeTrainer(bpe) => bpe.train(words).map(|(m, t)| (m.into(), t)),
            TrainerWrapper::WordPieceTrainer(wpt) => wpt.train(words).map(|(m, t)| (m.into(), t)),
            TrainerWrapper::UnigramTrainer(wpt) => wpt.train(words).map(|(m, t)| (m.into(), t)),
        }
    }

    fn process_tokens(&self, words: &mut HashMap<String, u32>, tokens: Vec<String>) {
        match self {
            TrainerWrapper::BpeTrainer(bpe) => bpe.process_tokens(words, tokens),
            TrainerWrapper::WordPieceTrainer(wpt) => wpt.process_tokens(words, tokens),
            TrainerWrapper::UnigramTrainer(wpt) => wpt.process_tokens(words, tokens),
        }
    }
}

#[cfg(not(feature = "bert"))]
impl_enum_from!(BpeTrainer, TrainerWrapper, BpeTrainer);
#[cfg(not(feature = "bert"))]
impl_enum_from!(WordPieceTrainer, TrainerWrapper, WordPieceTrainer);
#[cfg(not(feature = "bert"))]
impl_enum_from!(UnigramTrainer, TrainerWrapper, UnigramTrainer);
