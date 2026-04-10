# NIAH Benchmark: gemma4 E4B Q4_K_M

Needle-In-A-Haystack retrieval test comparing ggml-org (upstream) and
unsloth quantizations of `gemma-4-E4B-it` at Q4_K_M.

## Setup

- **Hardware**: Mac Mini M4, 16 GB unified memory
- **Server**: llama-cpp-turboquant via immortal, 32K context
- **Date**: 2026-04-10
- **Models**:
  - `ggml-org/gemma-4-E4B-it-GGUF:Q4_K_M` (4.62 GiB)
  - `unsloth/gemma-4-E4B-it-GGUF:Q4_K_M` (4.62 GiB, imatrix-calibrated)
- **Test**: `lctl niah --depth <D> --context <C>`
  - Needle: "The best thing to do in San Francisco is eat a
    sandwich and sit in Dolores Park on a sunny day."
  - Filler: 8 rotating paragraphs on unrelated topics
  - Scoring: response must contain "sandwich", "dolores park", "sunny"
- **Runs**: 3 per configuration (5 depths x 4 context sizes = 60 total per model)

## Results

Pass rate out of 3 runs per configuration.

| Depth | Context | ggml-org | unsloth |
|-------|---------|----------|---------|
| 0%    | 512     | 3/3      | 2/3     |
| 0%    | 1024    | 3/3      | 3/3     |
| 0%    | 2048    | 1/3      | 0/3     |
| 0%    | 4096    | 0/3      | 3/3     |
| 25%   | 512     | 3/3      | 3/3     |
| 25%   | 1024    | 3/3      | 2/3     |
| 25%   | 2048    | 3/3      | 3/3     |
| 25%   | 4096    | 3/3      | 3/3     |
| 50%   | 512     | 3/3      | 3/3     |
| 50%   | 1024    | 3/3      | 3/3     |
| 50%   | 2048    | 2/3      | 3/3     |
| 50%   | 4096    | 3/3      | 3/3     |
| 75%   | 512     | 3/3      | 2/3     |
| 75%   | 1024    | 3/3      | 0/3     |
| 75%   | 2048    | 0/3      | 3/3     |
| 75%   | 4096    | 1/3      | 2/3     |
| 100%  | 512     | 3/3      | 2/3     |
| 100%  | 1024    | 1/3      | 0/3     |
| 100%  | 2048    | 0/3      | 0/3     |
| 100%  | 4096    | 1/3      | 0/3     |

### Totals

ggml-org 42/60 (70%) vs unsloth 40/60 (67%)

## Analysis

- Both quantizations perform similarly overall. The 3-percentage-point
  difference is within noise for 60 trials.
- Both degrade at high needle depths (75-100%), where the needle sits
  near the end of the prompt far from the retrieval question. This is a
  known weak spot for smaller models.
- Depth 0% (needle at the very start) also shows failures at longer
  contexts for both models, likely due to the "lost in the middle"
  phenomenon where models attend more to the beginning and end of the
  prompt but struggle with specific positions at scale.
- The 25-50% depth range is the sweet spot for both, with near-perfect
  retrieval up to 4096 tokens.
- Failure modes differ: ggml-org tends to give short wrong answers,
  while unsloth more often returns empty responses.

## Conclusion

No meaningful quality difference between the two quantizations for
retrieval tasks. The ggml-org build is recommended as the default since
it is the upstream/canonical source and has marginally more consistent
behavior at edge cases.
