pub mod bert;
#[cfg(not(feature = "bert"))]
pub mod roberta;
#[cfg(not(feature = "bert"))]
pub mod template;

// Re-export these as processors
#[cfg(not(feature = "bert"))]
pub use super::pre_tokenizers::byte_level;

use serde::{Deserialize, Serialize};

#[cfg(not(feature = "bert"))]
use crate::{
    pre_tokenizers::byte_level::ByteLevel,
    processors::{roberta::RobertaProcessing, template::TemplateProcessing},
};
use crate::{processors::bert::BertProcessing, Encoding, PostProcessor, Result};

#[derive(Serialize, Deserialize, PartialEq, Debug, Clone)]
#[serde(untagged)]
pub enum PostProcessorWrapper {
    // Roberta must be before Bert for deserialization (serde does not validate tags)
    #[cfg(not(feature = "bert"))]
    Roberta(RobertaProcessing),
    Bert(BertProcessing),
    #[cfg(not(feature = "bert"))]
    ByteLevel(ByteLevel),
    #[cfg(not(feature = "bert"))]
    Template(TemplateProcessing),
}

impl PostProcessor for PostProcessorWrapper {
    fn added_tokens(&self, is_pair: bool) -> usize {
        match self {
            PostProcessorWrapper::Bert(bert) => bert.added_tokens(is_pair),
            #[cfg(not(feature = "bert"))]
            PostProcessorWrapper::ByteLevel(bl) => bl.added_tokens(is_pair),
            #[cfg(not(feature = "bert"))]
            PostProcessorWrapper::Roberta(roberta) => roberta.added_tokens(is_pair),
            #[cfg(not(feature = "bert"))]
            PostProcessorWrapper::Template(template) => template.added_tokens(is_pair),
        }
    }

    fn process(
        &self,
        encoding: Encoding,
        pair_encoding: Option<Encoding>,
        add_special_tokens: bool,
    ) -> Result<Encoding> {
        match self {
            PostProcessorWrapper::Bert(bert) => {
                bert.process(encoding, pair_encoding, add_special_tokens)
            }
            #[cfg(not(feature = "bert"))]
            PostProcessorWrapper::ByteLevel(bl) => {
                bl.process(encoding, pair_encoding, add_special_tokens)
            }
            #[cfg(not(feature = "bert"))]
            PostProcessorWrapper::Roberta(roberta) => {
                roberta.process(encoding, pair_encoding, add_special_tokens)
            }
            #[cfg(not(feature = "bert"))]
            PostProcessorWrapper::Template(template) => {
                template.process(encoding, pair_encoding, add_special_tokens)
            }
        }
    }
}

impl_enum_from!(BertProcessing, PostProcessorWrapper, Bert);
#[cfg(not(feature = "bert"))]
impl_enum_from!(ByteLevel, PostProcessorWrapper, ByteLevel);
#[cfg(not(feature = "bert"))]
impl_enum_from!(RobertaProcessing, PostProcessorWrapper, Roberta);
#[cfg(not(feature = "bert"))]
impl_enum_from!(TemplateProcessing, PostProcessorWrapper, Template);

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "bert"))]
    use super::*;

    #[cfg(not(feature = "bert"))]
    #[test]
    fn deserialize_bert_roberta_correctly() {
        let roberta = RobertaProcessing::default();
        let roberta_r = r#"{
            "type":"RobertaProcessing",
            "sep":["</s>",2],
            "cls":["<s>",0],
            "trim_offsets":true,
            "add_prefix_space":true
        }"#
        .replace(char::is_whitespace, "");
        assert_eq!(serde_json::to_string(&roberta).unwrap(), roberta_r);
        assert_eq!(
            serde_json::from_str::<PostProcessorWrapper>(&roberta_r).unwrap(),
            PostProcessorWrapper::Roberta(roberta)
        );

        let bert = BertProcessing::default();
        let bert_r = r#"{"type":"BertProcessing","sep":["[SEP]",102],"cls":["[CLS]",101]}"#;
        assert_eq!(serde_json::to_string(&bert).unwrap(), bert_r);
        assert_eq!(
            serde_json::from_str::<PostProcessorWrapper>(bert_r).unwrap(),
            PostProcessorWrapper::Bert(bert)
        );
    }
}
