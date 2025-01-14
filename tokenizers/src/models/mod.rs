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
        wordlevel::{WordLevel, WordLevelTrainer},
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
    #[cfg(not(feature = "bert"))]
    BPE(BPE),
    // WordPiece must stay before WordLevel here for deserialization (for retrocompatibility
    // with the versions not including the "type"), since WordLevel is a subset of WordPiece
    WordPiece(WordPiece),
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
    #[cfg(not(feature = "bert"))]
    type Trainer = TrainerWrapper;

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

    fn id_to_token(&self, id: u32) -> Option<String> {
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

    fn get_vocab(&self) -> HashMap<String, u32> {
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

    #[cfg(not(feature = "bert"))]
    fn get_trainer(&self) -> Self::Trainer {
        use ModelWrapper::*;
        match self {
            #[cfg(not(feature = "bert"))]
            WordLevel(t) => t.get_trainer().into(),
            WordPiece(t) => t.get_trainer().into(),
            #[cfg(not(feature = "bert"))]
            BPE(t) => t.get_trainer().into(),
            #[cfg(not(feature = "bert"))]
            Unigram(t) => t.get_trainer().into(),
        }
    }
}

#[cfg(not(feature = "bert"))]
pub enum TrainerWrapper {
    BpeTrainer(BpeTrainer),
    WordPieceTrainer(WordPieceTrainer),
    WordLevelTrainer(WordLevelTrainer),
    UnigramTrainer(UnigramTrainer),
}

#[cfg(not(feature = "bert"))]
impl Trainer for TrainerWrapper {
    type Model = ModelWrapper;

    fn should_show_progress(&self) -> bool {
        match self {
            TrainerWrapper::BpeTrainer(bpe) => bpe.should_show_progress(),
            TrainerWrapper::WordPieceTrainer(wpt) => wpt.should_show_progress(),
            TrainerWrapper::WordLevelTrainer(wpt) => wpt.should_show_progress(),
            TrainerWrapper::UnigramTrainer(wpt) => wpt.should_show_progress(),
        }
    }

    fn train(&self, model: &mut ModelWrapper) -> Result<Vec<AddedToken>> {
        match self {
            TrainerWrapper::BpeTrainer(t) => match model {
                ModelWrapper::BPE(bpe) => t.train(bpe),
                _ => Err("BpeTrainer can only train a BPE".into()),
            },
            TrainerWrapper::WordPieceTrainer(t) => match model {
                ModelWrapper::WordPiece(wp) => t.train(wp),
                _ => Err("WordPieceTrainer can only train a WordPiece".into()),
            },
            TrainerWrapper::WordLevelTrainer(t) => match model {
                ModelWrapper::WordLevel(wl) => t.train(wl),
                _ => Err("WordLevelTrainer can only train a WordLevel".into()),
            },
            TrainerWrapper::UnigramTrainer(t) => match model {
                ModelWrapper::Unigram(u) => t.train(u),
                _ => Err("UnigramTrainer can only train a Unigram".into()),
            },
        }
    }

    fn feed<I, S, F>(&mut self, iterator: I, process: F) -> Result<()>
    where
        I: Iterator<Item = S> + Send,
        S: AsRef<str> + Send,
        F: Fn(&str) -> Result<Vec<String>> + Sync,
    {
        match self {
            TrainerWrapper::BpeTrainer(bpe) => bpe.feed(iterator, process),
            TrainerWrapper::WordPieceTrainer(wpt) => wpt.feed(iterator, process),
            TrainerWrapper::WordLevelTrainer(wpt) => wpt.feed(iterator, process),
            TrainerWrapper::UnigramTrainer(wpt) => wpt.feed(iterator, process),
        }
    }
}

#[cfg(not(feature = "bert"))]
impl_enum_from!(BpeTrainer, TrainerWrapper, BpeTrainer);
#[cfg(not(feature = "bert"))]
impl_enum_from!(WordPieceTrainer, TrainerWrapper, WordPieceTrainer);
#[cfg(not(feature = "bert"))]
impl_enum_from!(UnigramTrainer, TrainerWrapper, UnigramTrainer);
#[cfg(not(feature = "bert"))]
impl_enum_from!(WordLevelTrainer, TrainerWrapper, WordLevelTrainer);

#[cfg(all(test, not(feature = "bert")))]
mod tests {
    use super::*;

    #[test]
    fn trainer_wrapper_train_model_wrapper() {
        let trainer = TrainerWrapper::BpeTrainer(BpeTrainer::default());
        let mut model = ModelWrapper::Unigram(Unigram::default());

        let result = trainer.train(&mut model);
        assert!(result.is_err());
    }
}
