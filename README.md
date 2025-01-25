# ⚒️ `kyzel` - local coding AI in your image

_Oh no, another fancy yet useless AI project..._

Bear with me, this one is more than a thin wrapper around Anthropic/OpenAI.

We're going to *make running local models compelling enough that you'd actually want to do so* - and not just because you can't use cloud models for privacy reasons.

## Concretely?

For users:
- [X] **Streamline high-interactivity tasks**: ask a local LLM to work with your files, analyze a dataset, iteratively debug a piece of code, figure out the idiosyncracies of a REST API, etc.
- [X] **Built for the real world**: the overarching goal is to make this practical enough that you'd actually want to use it for day-to-day tasks.
- [X] **Efficiency at the forefront**: we do our best to save your time, but also your GPU's: we adopt token-efficient strategies wherever possible.
- [X] **Privacy first**: Your data never leaves your device, unless you explicitly opt into using cloud models or sharing data with us in order to improve our models.
- [X] **Python-first**: LLMs are best at writing Python over other languages, so that's what we're focusing on
- [ ] **Plug-and-play**: download and run one of our fine-tuned models on your machine in minutes, with minimal setup.
- [ ] **The more you use it, the better it gets**: collect private interaction data as you go and fine-tune your model locally with it.

For researchers / ML engineers:
- [X] **On-device fine-tuning with [Unsloth](https://unsloth.ai/)**: none of this would be possible without their excellent work.
  - [X] **SFT with QLoRA**
  - [ ] **Low-data RL with DPO**
- [X] **Powered by `Phi4`** on larger GPUs (24GB VRAM), **`Qwen 2.5 Coder 3B`** on smaller ones (8GB VRAM). Designed to be repurposed with minimal for other/new models.
- [ ] **Didactical approach**: detailed writeups and emphasis on clean, minimal, robust code.

## Lingo
- A *conversation* is a sequence of (ChatML) messages. It's a flat stream, or list.
- An *event* is something that happened
  - Examples: the user sent a message to the model; the model sent a message to the user; the model thought to itself; the model requested execution of some code; the execution yielded some result...
  - One could consider less obvious events: the user *preferred* such-and-such generation over this other one; the user reworded the model's output
  - Importantly, the user can choose to roll back to an earlier point in the conversation (not necessarily perfectly, but at least roll back the chat history), and they do so through relevant events.
- A *session* contains many events. It's also a list, but it implicitly can branch out into many conversation paths, potential or actual.

## Repository structure

- `src`: Python code
- `data`: training/validation data
- `run`: runtime files

## `llmexport` exclude rules for core logic only

If using [`llmexport`](https://github.com/yberreby/llmexport), and wanting to focus on core logic, you can exclude display logic, sample data, tests, etc. with the following command:

```
llmexport --stdout * -i '*/ipy/*' -i 'data/*' -i '*test*' -i 'nb/*'
```

## Quickstart

Prerequisites:
- Modern NVIDIA GPU, preferably 24 GB VRAM (RTX 4090), at least 8GB VRAM (RTX 4060).
- Linux (tested on Arch Linux)

In a shell:
```
# Create venv, install dependencies. This will take a while.
./setup.sh

# Convert notebooks
uv run jupytext --to ipynb nb/*

# Run tests
uv run pytest

# Run Jupyter
uv run jupyter-lab .
```



## The story

I've seen local LLMs get better and better for a while now. Yet, somehow, I never found myself having any use for them. With each new release that fits on my hardware, I would fire up [`ollama`](https://github.com/ollama/ollama), ask some test questions, then go back to doing real work with much smarter (read: useful) proprietary models - usually Claude 3.5 Sonnet.

But then, every once in a while, I'd find myself debugging or exploring something with Claude, and I'd think "this would be so much simpler if I could _let the model do some exploration on its own_". Try some things, see what happens, and then report back. In many tasks, *context is key* - intelligence can only get you so far. A common way to get around this is to paste as much context as possible, but _gathering_ the appropriate context can be time-consuming when it is not just lines of code / documentation - think specific system configuration, behavior of some new API, etc. Yet, it's not necessarily that _complex_ to do - it's often a rather mechanical back-and-forth with the environment. Some intelligence/knowledge is required, of course, but it's not necessarily the bottleneck.

One could just load up a lot of money into their favorite AI cloud provider, use of the many fancy "AI coding agent" projects out there, and get rolling. However, this feels unsatisfactory. It also quickly gets ridiculously expensive. A single-person non-trivial project is easily 50-100k tokens. As of 2025-01-22, that's $0.15 - $0.30 per message with Claude 3.5 Sonnet (without prompt caching), so you could spend $1.5 - $3.0 in _10-30 seconds_ with a tight, (semi-)automated feedback loop that takes your code as context and iterates in a REPL. This is not sustainable for the vast majority of people.

Even without taking your whole codebase as context, steering a LLM purely in-context, with prompt engineering, in order to make it adhere to a specific response format/style/approach can require inputting a significant amount of tokens (and paying for them, with money or local compute). Seems like a waste. **What works super well for "style" / formatting / specific interaction patterns again? LoRA and the likes**. Too bad many cloud models can't be fine-tuned this way, and when they do, it's with a proprietary API that might go away at any time.

In January 2025, [Phi4 was released on HuggingFace](https://huggingface.co/microsoft/phi-4) (thanks, Microsoft!). It's a 14-billion parameter LLM with a out-of-the-box context length of 16k tokens. A quantized version can run in as little as 11GB of VRAM.

I made a few throwaway POCs with it, hooking it up to an IPython console with manual validation (in my shell) or fully-autonomous execution (in a Docker container), and found it impressively capable for its size and speed. It does paint itself into corners, get confused, make simple mistakes, etc, but even zero-shot, it holds promise.

Most of the issues I encountered with Phi4 could be attributed to, in descending order of prevalence:
- **Incorrect formatting**: fixable with inference-time constrained generation / fine-tuning to teach it correct formatting / just retrying with a non-negligible temperature.
- **Error cascading**: making a wrong assumption / decision early on, then getting confused by it later on. In my experience, this happens even to the best LLMs - confusion must be rooted out as early as possible.
- **Lack of context**: simply not having the full picture, because of only being able to get a few thousand tokens as inputs, or because it needs to be explicitly given a crucial piece of information. LLMs aren't magic - they know what they learned and what they are given as input, nothing more.
- **Lack of intelligence**: some tasks are simply too hard for a 14B-model to figure out. In those cases, guidance from a skilled human being and/or a frontier LLM is needed.

These seem... surprisingly fixable?

Note that a lot _could_ be done without any fine-tuning, but spending hours trying to get a model to properly follow instructions is rather frustrating (speaking from experience), and **wastes precious input tokens**. It's also rather model-specific. A good finetuning pipeline is more easily repurposable than a "prompt engineering pipeline" (read: sparkling string concatenation).

I had only tried finetuning a LLM once, with a short hacking session back in the LLaMA 1 days (eternity ago!). This seemed like a great excuse to see how tooling, fine-tuning methods and models have evolved since then!


## But don't you need tens of thousands of examples to fine-tune LLMs?

Why would you?

The models we consider were **pretrained** on trillions of tokens and **instruct-tuned** extensively. They can _already_ be steered rather decently without fine-tuning, with just a system prompt and care for staying close to the original data distribution. We're just going to _improve upon this baseline_. We're not starting from scratch.

We're not trying to inject fundamentally new knowledge into the model either, just to _repurpose_ existing knowledge contained within its pretrained weights. This means we shouldn't need that many steps / trainable parameters / examples.

For our purposes, **data quality > quantity**.

We're starting with _one_ handwritten example: [`c0.xml`](./data/c0.xml). Yes, one (1). This _one_ example already gives a useful training signal. We'll iterate from there.

In LoRA, the original pretrained weights are frozen and neatly separated from the low-rank weight delta that we will be learning. In the limit, as this delta tends to zero (w.r.t. some matrix norm), we get back the original pretrained model - our baseline, which is not useless. On the other hand, if the delta is large (or happens to destabilize crucial features - which is harder to characterize), it's easy to get gibberish back. Somewhere in between, there should be a sweet spot - close enough to the original model, but slightly adjusted to _better_ fit our task. Not perfectly, just better. That's already something.

## Making data repurposable

When fine-tuning LLMs, *formatting matters*. Same content, laid out differently = vastly different perplexity, training signal, performance.

For this reason, we quickly design a simple at-rest data storage format that easily allows repurposing samples. A conversation, or session, is a sequence of "events". An event can be emitted by the user or by the model.

We use a custom XML schema to cleanly delineate:
- **Human-written messages**: e.g. `find my most-backlinked org-roam nodes`, `no, don't use this library, use XXX`
- **Model "thoughts"**, which serve as communication for the user + CoT for the model itself: e.g. `Loading a SBERT model. This tends to be an expensive operation, so we'll time it.`
- **Code** that the model wants to run immediately, getting back its output: e.g. `import torch; torch.cuda.is_available()`
- **Execution output / result from the said code**: e.g. `True`, `ImportError: No module named 'torch'`

This this, we can use any format we want for the model's input/output: code could be wrapped in triple backticks, in XML tags, in-between special tokens, etc. This is especially useful for handling new model releases - if we collectively switch away from ChatML, no problem: just write a new conversion routine, and token streams conforming to `<standard of the day>` can be generated on-the-fly.
