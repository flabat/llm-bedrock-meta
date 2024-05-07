# llm-bedrock-meta

[![PyPI](https://img.shields.io/pypi/v/llm-bedrock-meta.svg)](https://pypi.org/project/llm-bedrock-meta/)
[![Changelog](https://img.shields.io/github/v/release/flabat/llm-bedrock-meta?include_prereleases&label=changelog)](https://github.com/flabat/llm-bedrock-meta/releases)
[![Tests](https://github.com/flabat/llm-bedrock-meta/workflows/Test/badge.svg)](https://github.com/flabat/llm-bedrock-meta/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/flabat/llm-bedrock-meta/blob/main/LICENSE)

Plugin for [LLM](https://llm.datasette.io/) adding support for Meta LLama 2 and 3 models in Amazon Bedrock

## Installation

Install this plugin in the same environment as LLM. From the current directory
```bash
llm install llm-bedrock-meta
```
## Configuration

You will need to specify AWS Configuration with the normal boto3 and environment variables.

For example, to use the region `us-west-2` and AWS credentials under the `personal` profile, set the environment variables

```bash
export AWS_DEFAULT_REGION=us-west-2
export AWS_PROFILE=personal
```

## Usage

This plugin adds models called `bedrock-llama2-13b`, `bedrock-llama2-70b`, `bedrock-llama3-8b-instruct`, and `bedrock-llama3-70b-instruct`. You can also use it with the aliases like `bl2` or `bl2-70`.

You can query them like this:

```bash
llm -m bl3-70-i "Ten great names for a new space station"
```

```bash
llm -m bl3-70-i "Ten great names for a new space station"
```

You can also chat with the model:

```bash
llm chat -m bl3-70-i
```

## Options

- `-o max_gen_len 1024`, default 2_048: The maximum number of tokens to generate before stopping.
- `-o verbose 1`, default 0: Output more verbose logging.
- `-o temperature 0.8`, default 0.6: Use a lower value to decrease randomness in the response.
- `top_p`, default 0.9: Use a lower value to ignore less probable options. Set to 0 or 1.0 to disable.

Use like this:
```bash
llm -m bl3-70-i -o max_gen_len 20 "Sing me the alphabet"
 Ah, ah, ah! Here's the alphabet song for you:


A, B, C, D, E, F, G,
H, I, J, K, L, M, N, O,
P, Q, R, S, T, U, V, W,
X, Y, Z,
Now I know my ABCs,
Next time won't you sing with me?


Hope that brought a smile to your face!
```

