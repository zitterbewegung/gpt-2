#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf

import model, sample, encoder

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=1,
    temperature=1,
    top_k=0,
    top_p=0
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    :top_p=0.0 : Float value controlling diversity. Implements nucleus sampling,
     overriding top_k if set to a value > 0. A good setting is 0.9.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 2
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)

        while True:
            raw_text = input("Model prompt >>> ")
            while not raw_text:
                print('Prompt should not be empty!')
                raw_text = input("Model prompt >>> ")
            raw_text = raw_text.replace('\\n', '\n')
            print(repr(raw_text))
            context_tokens = enc.encode(raw_text)
            generated = 0
            for text in generate_poetry(prompt=raw_text, enc=enc, output=output, context=context, nsamples=1, batch_size=batch_size, sess=sess):
                #print("=" * 40 + " SAMPLE " + str(generated) + " " + "=" * 40)
                print(text)
                generated += 1
            print("=" * 80)

def generate_result(prompt, enc, output, context, nsamples=1, batch_size=1, sess=tf.get_default_session()):
    context_tokens = enc.encode(prompt)
    for _ in range(nsamples // batch_size):
        out = sess.run(output, feed_dict={
            context: [context_tokens for _ in range(batch_size)]
        })[:, len(context_tokens):]
        for i in range(batch_size):
            text = enc.decode(out[i])
            yield text

def generate_line_no_e(prompt, **kws):
    result = prompt
    current = ''
    while '\n' not in current:
        for text in generate_result(prompt + current, **kws):
            if 'e' not in text:
                current += text
    return prompt + current

def generate_line_no_e(prompt, **kws):
    result = prompt
    current = ''
    while '\n' not in current:
        for text in generate_result(prompt + current, **kws):
            if 'e' in text:
                print(repr((prompt + current + text).splitlines()[-1]))
                current = ''
            else:
                current += text
    return prompt + current


def generate_line(prompt, **kws):
    result = prompt
    current = ''
    while '\n' not in current:
        for text in generate_result(prompt + current, **kws):
            current += text
    return prompt + current

import pronouncing as pro

def last_word(text):
  while len(text) > 0 and not str.isalnum(text[-1]):
    text = text[0:-1]
  return text.split()[-1]

def generated_poem(prompt):
  if prompt.endswith('\n'):
    return prompt
  else:
    return '\n'.join(prompt.splitlines()[0:-1]) + '\n'

def generate_poem(prompt, **kws):
  rhymes = [x for x in pro.rhymes(last_word(generated_poem(prompt)))] #if 'e' not in x]
  print(repr(rhymes))
  while True:
    attempt = generate_line(prompt, **kws)
    print(repr(attempt.splitlines()[-1]))
    if last_word(attempt) in rhymes:
      return attempt

def generate_poetry(prompt, **kws):
  if len(generated_poem(prompt).strip().splitlines()) == 1 and not prompt.endswith('\n'):
    prompt = generate_line(prompt, **kws)
    print(repr(prompt))
  if len(generated_poem(prompt).strip().splitlines()) % 2 == 0:
    attempt = generate_line(prompt, **kws)
    yield attempt
  else:
    text = generate_poem(prompt, **kws)
    yield text

if __name__ == '__main__':
    fire.Fire(interact_model)
