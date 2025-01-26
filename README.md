# ⚒️ `kyzel` - local-first REPL assistant

_kyzel (/ˈkɪzəl/): mispronunciation of "chisel"._

**Hook a pretrained SoTA LLM running on your hardware to your IPython REPL, and let it learn from you.**

![demo](./docs/demo.png)

- Ask questions and let a local LLM write and run Python code to answer them.
  - **Inference runs on your hardware**: no Internet connectivity required unless you opt-in.
  - **Code runs on your machine, with your environment**: use it to explore a new library, debug a piece of code, etc.
  - **No untrusted code execution**: approve code fragments in 1-2 keystrokes.
  - **Dissociated LLM inference and execution environment**: if you have a laptop and a GPU workstation/server, you can run the inference server on the workstation (`src.server`) while executing the generated code on your laptop.
- **Interject** to steer the model (your guidance is precious!), or **stay silent** and let it decide what to do next by itself when the path ahead is clear.
- Easily **collect private, on-device data** from your interactions with the model.
- **Locally fine-tune** the model on this data with QLoRA on a single GPU, in minutes.


## Quickstart

Prerequisites:
- [`uv`](https://docs.astral.sh/uv/) for Python dependency management
- Modern NVIDIA GPU (tested on RTX 4090)
- Linux (tested on Arch Linux)

Initial setup:
```zsh
# Create venv, install dependencies. This will take a while - we have large binary dependencies.
./setup.sh

# Convert Jupytext notebooks to .ipynb
uv run jupytext --to ipynb nb/*.py

# Run tests
uv run -m pytest
```

To **start the inference server** on a machine with a NVIDIA GPU and at least 12GB of VRAM:
```
# This will download a pretrained model on first run, then start a tiny, unauthenticated REST API on 127.0.0.1:8000.
# Don't expose it publicly, use SSH tunneling.
# This API cannot execute any code, only generate tokens.
uv run -m src.server
```

Then, on the code execution machine (can be same as inference machine, or different), **start a Jupyter server**:
```zsh
uv run jupyter-lab .
```

Then start playing around with the notebooks in the `nb/` directory.


## Behind this project

_This project is a work-in-progress. It is not yet production-ready, but it is functional._

I'm a neuro-AI researcher. This is a **solo, hobby project** that I hack around on in my free time, started because I wanted to see what the SoTA of local LLMs could do by now, at an inference/training speed that I can tolerate and on hardware that isn't prohibitively expensive.

That being said, you should generally expect **high code quality**. I've done related POCs before (unpublished), which were somewhat-functional and got me where I wanted to be, but the code was _atrocious_ spaghetti cobbled together with Claude 3.5 Sonnet in 2-3 hour hacking sessions. In contrast, **I want this codebase to be a joy to read and develop**. Everything should be clearly organized and mostly self-documenting. Please shame me in issues if this is not the case (though in those instances, I am likely aware already, and it's on my TODO list).

However, you may encounter dependency issues, quirks, regressions, etc. Sorry for those; limited bandwidth. Please also open an issue in that case, or better yet, a pull request.

The LLM world moves _fast_, details are _exceedingly important_ (woops, you had an extra trailing newline? here is your _huge perplexity increase_!), and it can be hard to find full working examples. I hope this repo can be a step toward that.

And who knows - maybe it'll get to the point where it's user-friendly enough that you'd care even if ML engineering isn't your thing :)


## Juicy bits implemented so far

Note: all notebooks are stored using [Jupytext](https://github.com/mwouts/jupytext); you must **convert them to .ipynb** before running them in Jupyter! To do so, run `uv run jupytext --to ipynb nb/*.py`.

- **Constrained generation** with a custom [LogitsProcessor](https://huggingface.co/docs/transformers/v4.48.0/en/internal/generation_utils#transformers.LogitsProcessor):
  - Ensures that even without any fine-tuning, LLM outputs always contain a `<thought>` tag (for it to do CoT), an `<action>` tag (for it to describe its next action at a high level), and a Python code block to be executed in the REPL upon user approval.
  - This can be **surprisingly non-trivial to get right**, given the existence of multiple tokenizations of a given piece of text, the presence of undesirable trailing characters after an expected token, etc.
  - See implementation: [`src.generate.constrain`](./src/generate/constrain/__init__.py_)
  - Upon generation, model outputs are parsed to extract structured data.
- **Model-agnostic events**:
  - **Events** ([`src.types.events`](./src/types/events.py)) are an abstraction over assistant/user actions, decoupled from formatting/tokenization concerns.
  - **Sessions** ([`src.types.session`](./src/types/session.py)) are collections of events that happened over a given user-AI interaction.
  - Events, and by extension **sessions**, can easily be converted to/from XML for storage and sharing: see [`src.persist`](./src/persist/).
- **Rich, colorful display in Jupyter**: custom HTML formatting ([`src.display.html`](./src/display/html)) in order to make working with event streams / sessions a joy.
- **Human-in-the-loop data collection**:
  - Base or fine-tuned models can be used interactively with the `nb/interactive` notebook (still rather primitive, but generally functional).
  - Satisfactory interaction sessions can be saved to XML from the notebook.
  - The plaintext XML format enables easy manual correction of small missteps in the text editor of your choice.
  - Once a few sessions have been collected, they can be **flattened into token streams** and **converted to a HuggingFace [`Dataset`](https://huggingface.co/docs/datasets/en/package_reference/main_classes#datasets.Dataset)** using the `xml_to_dataset` notebook.
- **Quantized Phi-4 fine-tuning with Unsloth**
  - `nb/train` allows for low-data fine-tuning of Phi-4 on the collected data with **as little as 10-13 GB of VRAM** thanks to [Unsloth](https://unsloth.ai/).


## Making data repurposable

When fine-tuning LLMs, *formatting matters*. Same content, laid out differently = vastly different perplexity, training signal, performance.

For this reason, we have a simple at-rest data storage format that easily allows saved sessions to be repurposed.

We use a custom XML schema to cleanly delineate semantically-distinct events, such as:
- **Human-written messages**: e.g. `find my most-backlinked org-roam nodes`, `no, don't use this library, use XXX`
- **Model "thoughts"**, which serve as communication for the user + CoT for the model itself: e.g. `Loading a SBERT model. This tends to be an expensive operation, so we'll time it.`
- **Code** that the model wants to run immediately, getting back its output: e.g. `import torch; torch.cuda.is_available()`
- **Execution output / result from the said code**: e.g. `True`, `ImportError: No module named 'torch'`

With this, we can use any format we want for the model's input/output: code could be wrapped in triple backticks, in XML tags, in-between special tokens, etc. This is especially useful for handling new model releases - if we collectively switch away from ChatML, no problem: just write a new conversion routine, and token streams conforming to `<standard of the day>` can be generated on the fly.

## FAQ

## But don't you need tens of thousands of examples to fine-tune LLMs?

Not necessarily.

The models we consider were **pretrained** on trillions of tokens and **instruct-tuned** extensively. They can _already_ be steered rather decently without fine-tuning, with just a system prompt and care for staying close to the original data distribution. We're just going to _improve upon this baseline_. We're not starting from scratch.

We're not trying to inject fundamentally new knowledge into the model either, just to _repurpose_ existing knowledge contained within its pretrained weights. This means we shouldn't need that many steps / trainable parameters / examples.

For our purposes, **data quality > quantity**. Even one high-quality sample, used carefully, can provide a useful learning signal.

In LoRA, the original pretrained weights are frozen and neatly separated from the low-rank weight delta that we will be learning. In the limit, as this delta tends to zero (w.r.t. some matrix norm), we get back the original pretrained model - our baseline, which is not useless (given its existing abilities + system prompt + our constrained generation pipeline). On the other hand, if the delta is large (or happens to destabilize crucial features - which is harder to characterize), it's easy to get gibberish back. Somewhere in between, there should be a sweet spot - close enough to the original model, but slightly adjusted to _better_ fit our task. Not perfectly, just better. That's already something.

### Why use a custom output grammar/format instead of JSON tool use?

Output formats involving deep nesting, escaping, quoting, etc, are not great for LLMs, unless they were specifically trained to handle them. Tool use models _are_ often trained for JSON, but these tend to be biased toward minimizing the number of tool calls, which we don't want here - the model should "feel free" to do many REPL roundtrips, prioritizing correctness and knowledge discovery.

This formatting strategy is not a radical idea. Consider **CodeAct** by Apple AI (see [paper](https://arxiv.org/abs/2402.01030), [dataset](https://huggingface.co/datasets/xingyaoww/code-act?row=1)): the model thinks in plaintext and outputs code within `<execute>` XML tags.


### Why not just use a cloud model?

- Your data leaves your machine - no-go for many usecases.
- Can get expensive quickly.
- Can't fully customize the grammar.
- Often, cannot fine-tune, or have to do so through an idiosyncratic proprietary API.

That being said, local models running on 1 GPU won't be at frontier-level intelligence.

As such, it makes sense to have the option to request help from a cloud model when faced with a complex task - but only when needed (to reduce costs), and with the user's explicit consent. Not implemented for now, but I had it in a previous POC, and it worked very well.


## Original story (written early into the project)

I've seen local LLMs get better and better for a while now. Yet, somehow, I never found myself having any use for them. With each new release that fits on my hardware, I would fire up [`ollama`](https://github.com/ollama/ollama), ask some test questions, then go back to doing real work with much smarter (read: useful) proprietary models - usually Claude 3.5 Sonnet.

But then, every once in a while, I'd find myself debugging or exploring something with Claude, and I'd think "this would be so much simpler if I could _let the model do some exploration on its own_". Try some things, see what happens, and then report back. In many tasks, *context is key* - intelligence can only get you so far. A common way to get around this is to paste as much context as possible, but _gathering_ the appropriate context can be time-consuming when it is not just lines of code / documentation - think specific system configuration, behavior of some new API, etc. Yet, it's not necessarily that _complex_ to do - it's often a rather mechanical back-and-forth with the environment. Some intelligence/knowledge is required, of course, but it's not necessarily the bottleneck.

One could just load up a lot of money into their favorite AI cloud provider, use of the many fancy "AI coding agent" projects out there, and get rolling. However, this feels unsatisfactory. It also quickly gets prohibitively expensive. A single-person non-trivial project is easily 50-100k tokens. As of 2025-01-22, that's $0.15 - $0.30 per message with Claude 3.5 Sonnet (without prompt caching), so you could spend $1.5 - $3.0 in _10-30 seconds_ with a tight, (semi-)automated feedback loop that takes your code as context and iterates in a REPL. This is not sustainable for the vast majority of people.

Even without taking your whole codebase as context, steering a LLM purely in-context, with prompt engineering, in order to make it adhere to a specific response format/style/approach can require inputting a significant amount of tokens (and paying for them, with money or local compute). Seems like a waste. **What works super well for "style" / formatting / specific interaction patterns again? LoRA and the likes**. Too bad many cloud models can't be fine-tuned this way, and when they do, it's with a proprietary API that might go away at any time.

In January 2025, [Phi4 was released on HuggingFace](https://huggingface.co/microsoft/phi-4) (thanks, Microsoft!). It's a 14-billion-parameter LLM with a out-of-the-box context length of 16k tokens. A quantized version can run in as little as 11GB of VRAM.

I made a few throwaway POCs with it, hooking it up to an IPython console with manual validation (in my shell) or fully-autonomous execution (in a Docker container), and found it impressively capable for its size and speed. It does paint itself into corners, get confused, make simple mistakes, etc, but even zero-shot, it holds promise.

Most of the issues I encountered with Phi4 could be attributed to, in descending order of prevalence:
- **Incorrect formatting**: fixable with inference-time constrained generation / fine-tuning to teach it correct formatting / just retrying with a non-negligible temperature.
- **Error cascading**: making a wrong assumption / decision early on, then getting confused by it later on. In my experience, this happens even to the best LLMs - confusion must be rooted out as early as possible.
- **Lack of context**: simply not having the full picture, because of only being able to get a few thousand tokens as inputs, or because it needs to be explicitly given a crucial piece of information. LLMs aren't magic - they know what they learned and what they are given as input, nothing more.
- **Lack of intelligence**: some tasks are simply too hard for a 14B-model to figure out. In those cases, guidance from a skilled human being and/or a frontier LLM is needed.

These seem... surprisingly fixable?

Note that a lot _could_ be done without any fine-tuning, but spending hours trying to get a model to properly follow instructions is rather frustrating (speaking from experience), and **wastes precious input tokens**. It's also rather model-specific. A good finetuning pipeline is more easily repurposable than a "prompt engineering pipeline" (read: sparkling string concatenation).

I had only tried finetuning a LLM once, with a short hacking session back in the LLaMA 1 days (eternity ago!). This seemed like a great excuse to see how tooling, fine-tuning methods and models have evolved since then!


## License

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.
