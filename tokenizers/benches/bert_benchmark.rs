#[macro_use]
extern crate criterion;

mod common;

use std::{
    fs::File,
    io::{BufRead, BufReader},
    path::Path,
};

use criterion::Criterion;
use tokenizers::{
    decoders, models::wordpiece::WordPiece, normalizers::BertNormalizer,
    pre_tokenizers::bert::BertPreTokenizer, processors::bert::BertProcessing, EncodeInput, Model,
    TokenizerImpl,
};
#[cfg(not(feature = "bert"))]
use tokenizers::{
    decoders::DecoderWrapper, models::wordpiece::WordPieceTrainerBuilder,
    normalizers::NormalizerWrapper, pre_tokenizers::whitespace::Whitespace,
    processors::PostProcessorWrapper,
};

#[cfg(not(feature = "bert"))]
use common::iter_bench_train;
use common::{iter_bench_encode, iter_bench_encode_batch};

static BATCH_SIZE: usize = 1_000;

type BertTokenizer = TokenizerImpl<
    WordPiece,
    BertNormalizer,
    BertPreTokenizer,
    BertProcessing,
    decoders::wordpiece::WordPiece,
>;

/// Resembling the BertTokenizer implementation from the Python bindings.
fn create_bert_tokenizer(wp: WordPiece) -> BertTokenizer {
    let sep_id = *wp.get_vocab().get("[SEP]").unwrap();
    let cls_id = *wp.get_vocab().get("[CLS]").unwrap();
    let mut tokenizer = TokenizerImpl::new(wp);
    tokenizer.with_pre_tokenizer(BertPreTokenizer);
    tokenizer.with_normalizer(BertNormalizer::default());
    tokenizer.with_decoder(decoders::wordpiece::WordPiece::default());
    tokenizer.with_post_processor(BertProcessing::new(
        ("[SEP]".to_string(), sep_id),
        ("[CLS]".to_string(), cls_id),
    ));
    tokenizer
}

pub fn bench_bert(c: &mut Criterion) {
    let wp = WordPiece::from_file("data/bert-base-uncased-vocab.txt")
        .build()
        .unwrap();
    let tokenizer = create_bert_tokenizer(wp);
    let mut lines: Vec<EncodeInput> = vec![];
    let mut batches: Vec<Vec<EncodeInput>> = vec![vec![]];
    for line in BufReader::new(File::open(Path::new("data/big.txt")).unwrap()).lines() {
        let line: EncodeInput = line.unwrap().into();
        lines.push(line.clone());
        if batches.last().unwrap().len() >= BATCH_SIZE {
            batches.push(vec![]);
        }
        batches.last_mut().unwrap().push(line);
    }

    c.bench_function("WordPiece BERT encode", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, &tokenizer, &lines))
    });

    c.bench_function("WordPiece BERT encode batch", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, &tokenizer, &batches))
    });
}

#[cfg(not(feature = "bert"))]
fn bench_train(c: &mut Criterion) {
    let mut trainer = WordPieceTrainerBuilder::default()
        .show_progress(false)
        .build();
    type Tok = TokenizerImpl<
        WordPiece,
        NormalizerWrapper,
        Whitespace,
        PostProcessorWrapper,
        DecoderWrapper,
    >;
    let mut tokenizer = Tok::new(WordPiece::default());
    tokenizer.with_pre_tokenizer(Whitespace::default());
    c.bench_function("WordPiece Train vocabulary (small)", |b| {
        b.iter_custom(|iters| {
            iter_bench_train(
                iters,
                &mut tokenizer,
                &mut trainer,
                vec!["data/small.txt".to_string()],
            )
        })
    });

    let mut tokenizer = Tok::new(WordPiece::default());
    tokenizer.with_pre_tokenizer(Whitespace::default());
    c.bench_function("WordPiece Train vocabulary (big)", |b| {
        b.iter_custom(|iters| {
            iter_bench_train(
                iters,
                &mut tokenizer,
                &mut trainer,
                vec!["data/big.txt".to_string()],
            )
        })
    });
}

criterion_group! {
    name = bert_benches;
    config = Criterion::default().sample_size(20);
    targets = bench_bert
}

#[cfg(not(feature = "bert"))]
criterion_group! {
    name = benches_train;
    config = Criterion::default().sample_size(10);
    targets = bench_train
}

#[cfg(not(feature = "bert"))]
criterion_main!(bert_benches, benches_train);
#[cfg(feature = "bert")]
criterion_main!(bert_benches);
