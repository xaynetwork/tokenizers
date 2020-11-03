#[macro_use]
extern crate criterion;

mod common;

use std::{
    fs::File,
    io::{BufRead, BufReader},
    ops::Deref,
    path::Path,
};

#[cfg(feature = "trainer")]
use common::iter_bench_train;
use common::{iter_bench_encode, iter_bench_encode_batch};
use criterion::Criterion;
#[cfg(feature = "trainer")]
use tokenizers::{
    models::bpe::BpeTrainerBuilder, models::TrainerWrapper, pre_tokenizers::whitespace::Whitespace,
};
use tokenizers::{
    models::bpe::BPE,
    pre_tokenizers::byte_level::ByteLevel,
    tokenizer::{AddedToken, EncodeInput},
    Tokenizer,
};

static BATCH_SIZE: usize = 1_000;

fn create_gpt2_tokenizer(bpe: BPE) -> Tokenizer {
    let mut tokenizer = Tokenizer::new(bpe);
    tokenizer.with_pre_tokenizer(ByteLevel::default());
    tokenizer.with_decoder(ByteLevel::default());
    tokenizer.add_tokens(&[AddedToken::from("ing", false).single_word(false)]);
    tokenizer.add_special_tokens(&[AddedToken::from("[ENT]", true).single_word(true)]);
    tokenizer
}

fn bench_gpt2(c: &mut Criterion) {
    let bpe = BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .build()
        .unwrap();
    let tokenizer = create_gpt2_tokenizer(bpe);
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

    c.bench_function("BPE GPT2 encode", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, tokenizer.deref(), &lines))
    });

    c.bench_function("BPE GPT2 encode batch", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, tokenizer.deref(), &batches))
    });

    let bpe = BPE::from_file("data/gpt2-vocab.json", "data/gpt2-merges.txt")
        .cache_capacity(0)
        .build()
        .unwrap();
    let tokenizer = create_gpt2_tokenizer(bpe);

    c.bench_function("BPE GPT2 encode, no cache", |b| {
        b.iter_custom(|iters| iter_bench_encode(iters, &tokenizer, &lines))
    });

    c.bench_function("BPE GPT2 encode batch, no cache", |b| {
        b.iter_custom(|iters| iter_bench_encode_batch(iters, &tokenizer, &batches))
    });
}

#[cfg(feature = "trainer")]
fn bench_train(c: &mut Criterion) {
    #[cfg(feature = "progressbar")]
    let trainer: TrainerWrapper = BpeTrainerBuilder::default()
        .show_progress(false)
        .build()
        .into();
    #[cfg(not(feature = "progressbar"))]
    let trainer: TrainerWrapper = BpeTrainerBuilder::default().build().into();
    let mut tokenizer = Tokenizer::new(BPE::default()).into_inner();
    tokenizer.with_pre_tokenizer(Whitespace::default());
    c.bench_function("BPE Train vocabulary (small)", |b| {
        b.iter_custom(|iters| {
            iter_bench_train(
                iters,
                &mut tokenizer,
                &trainer,
                vec!["data/small.txt".to_string()],
            )
        })
    });

    let mut tokenizer = Tokenizer::new(BPE::default()).into_inner();
    tokenizer.with_pre_tokenizer(Whitespace::default());
    c.bench_function("BPE Train vocabulary (big)", |b| {
        b.iter_custom(|iters| {
            iter_bench_train(
                iters,
                &mut tokenizer,
                &trainer,
                vec!["data/big.txt".to_string()],
            )
        })
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(20);
    targets = bench_gpt2
}
#[cfg(feature = "trainer")]
criterion_group! {
    name = benches_train;
    config = Criterion::default().sample_size(10);
    targets = bench_train
}
#[cfg(not(feature = "trainer"))]
criterion_main!(benches);
#[cfg(feature = "trainer")]
criterion_main!(benches, benches_train);
