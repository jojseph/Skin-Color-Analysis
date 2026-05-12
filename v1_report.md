# 📊 Project Report on Large Language Models (LLMs)
## Agarthan Skin Tone Analyzer (ASTA)

> **Reference:** MIT 6.S191 Introduction to Deep Learning (2025) — Large Language Models (Google)

---

## 1. Project Overview

The **Agarthan Skin Tone Analyzer (ASTA)** is a web application that determines a user's **seasonal colour type** — Spring, Summer, Autumn, or Winter — from a portrait photograph. Given a season classification, the app returns a personalised:

- Colour palette (6 hex values)
- Outfit colour recommendations and colours to avoid
- Makeup tips (foundation, blush, lips, eyes)

The application accepts input via file upload, live camera capture, and a curated sample gallery. The entire classification and recommendation pipeline is handled by a single **Vision Language Model (VLM)** call — no separate rule-based system or custom-trained model is required.

---

## 2. Background: From RNNs to Transformers

To understand why modern LLMs are built the way they are, it helps to understand what came before them.

### Recurrent Neural Networks (RNNs)

Early sequence models processed text **one token at a time**, passing a hidden state forward at each step. While elegant, this design had two critical weaknesses:

- **Vanishing gradients** — information from early tokens was diluted or lost as the sequence length grew, making it hard to learn long-range dependencies
- **Sequential bottleneck** — each step depended on the previous one, so computation could not be parallelised; training on long sequences was slow

LSTMs and GRUs improved the memory problem but did not solve the parallelisation issue.

### The Attention Breakthrough

In the landmark 2017 paper *"Attention Is All You Need"* (Vaswani et al., Google), the authors asked: *what if instead of passing information step by step, we let every token attend directly to every other token in the sequence simultaneously?*

This idea — **self-attention** — became the foundation of the Transformer architecture and, subsequently, all modern LLMs.

Key properties of the Transformer that addressed RNN limitations:
- **Parallelisable** — all tokens are processed simultaneously; training is dramatically faster on GPUs/TPUs
- **Long-range dependencies** — any two tokens can interact directly, regardless of their distance in the sequence

---

## 3. The Transformer Architecture

The Transformer is the architectural backbone of every modern LLM including Gemini. It is composed of stacked **Transformer blocks**, each containing two main components: a **Multi-Head Self-Attention** layer and a **Feed-Forward Network**. Both are wrapped with residual connections and layer normalisation.

![The Transformer encoder–decoder architecture (Vaswani et al., 2017)](https://upload.wikimedia.org/wikipedia/commons/8/8f/The-Transformer-model-architecture.png)

*Figure 1 — The Transformer encoder–decoder architecture (Vaswani et al., 2017, "Attention Is All You Need"). Modern decoder-only LLMs like Gemini use only the right half (decoder stack) with a causal mask.*

### 3.1 Tokenisation

Before any processing, raw text is converted into discrete integer tokens using a **tokeniser**. Modern LLMs use **Byte Pair Encoding (BPE)** or similar subword algorithms:

1. Start with a large text corpus
2. Iteratively merge the most frequent adjacent byte pairs into a single new token
3. Repeat until the desired vocabulary size is reached (typically 32k–100k tokens)

For example, the word `"skateboarding"` might be tokenised as `["skate", "boarding"]` — two known subwords — rather than character by character. This balances vocabulary size with the ability to represent rare or novel words.

Each token is then mapped to a dense vector of fixed dimension *d* called an **embedding** — the model's learned numerical representation of that token.

### 3.2 Self-Attention: Queries, Keys, and Values

Self-attention is the mechanism that lets every token look at every other token and decide how much to attend to each one. For an input sequence of *n* tokens, each represented as an embedding vector, the attention layer computes three new matrices from learned linear projections:

| Matrix | Symbol | Role |
|---|---|---|
| **Query** | **Q** | "What am I looking for?" |
| **Key** | **K** | "What do I contain / advertise?" |
| **Value** | **V** | "What information do I carry?" |

The attention score between token *i* and token *j* is computed as the dot product of token *i*'s query with token *j*'s key, scaled by √*d_k* to prevent very large values:

```python
# Scaled Dot-Product Attention (conceptual)
import numpy as np

def scaled_dot_product_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / np.sqrt(d_k)  # (n, n) attention matrix
    weights = softmax(scores, dim=-1)                  # normalise to sum to 1
    return weights @ V                                 # weighted sum of values
```

![Scaled Dot-Product Attention block diagram](https://upload.wikimedia.org/wikipedia/commons/thumb/1/1b/Transformer%2C_attention_block_diagram.png/500px-Transformer%2C_attention_block_diagram.png)

The output for each token is a **weighted sum of all value vectors**, where the weights reflect how relevant each other token is. A token will attend strongly to tokens whose keys match its query — effectively routing information through the sequence based on semantic relevance rather than positional proximity.

**Masking:** In decoder-only models (like GPT, and the generation component of Gemini), a **causal mask** prevents each token from attending to future tokens. This preserves the autoregressive property needed for text generation.

### 3.3 Multi-Head Attention

Rather than computing a single attention function, the Transformer computes **h parallel attention heads**, each with its own Q, K, V projection matrices:

- Each head can learn to focus on a **different type of relationship** — one head might track syntactic agreement, another coreference, another semantic similarity
- The outputs of all heads are concatenated and linearly projected back to dimension *d*

This is analogous to using multiple convolutional filters in a CNN — each filter detects a different feature.

![Multi-Head Attention: h independent attention heads run in parallel, outputs concatenated then projected](https://upload.wikimedia.org/wikipedia/commons/thumb/d/d2/Multiheaded_attention%2C_block_diagram.png/500px-Multiheaded_attention%2C_block_diagram.png)

### 3.4 Positional Encoding

Self-attention is **permutation-invariant** — it treats the input as a set, not a sequence. To give the model information about token order, a **positional encoding** is added to each token embedding before the first layer.

Modern LLMs (including Gemini) use **Rotary Position Embeddings (RoPE)**, which encode absolute and relative position information directly into the Q and K matrices. This allows the model to generalise to longer sequences than it was trained on.

### 3.5 Feed-Forward Network & Residual Connections

After the attention layer, each token's representation passes through a position-wise **Feed-Forward Network (FFN)** — two linear layers with a non-linearity (typically GeLU) in between. This is where the model stores most of its "factual knowledge" according to mechanistic interpretability research.

**Residual connections** (skip connections) add the input of each sub-layer to its output. Combined with **layer normalisation**, this prevents gradient degradation during training and allows very deep stacks of Transformer blocks.

```python
# A single Transformer block (simplified)
class TransformerBlock(nn.Module):
    def forward(self, x):
        # Multi-Head Self-Attention with residual
        x = x + self.attention(self.norm1(x))
        # Feed-Forward Network with residual
        x = x + self.ffn(self.norm2(x))
        return x
```

---

## 4. LLM Training Pipeline

Training a production LLM like Gemini involves three distinct phases, each building on the previous.

### Phase 1: Pre-training

The model is trained on a **massive corpus** of text (web pages, books, code, scientific papers) using **next-token prediction** as the objective:

> *Given tokens 1…n, predict token n+1.*

This simple objective, applied at web scale, forces the model to develop deep representations of language, world knowledge, reasoning, and common sense.

Pre-training is the most compute-intensive phase. Gemini was trained on trillions of tokens across text and multimodal data using Google's TPU infrastructure.

### Phase 2: Supervised Fine-Tuning (SFT)

The pre-trained model is a next-token predictor, not an assistant. SFT adapts it to follow instructions by fine-tuning on a curated dataset of **prompt → ideal response** pairs, written or curated by human annotators.

This teaches the model the format and style of being helpful, while the pre-trained knowledge base remains largely intact.

*"SFT is like teaching a very knowledgeable person how to communicate clearly."*

### Phase 3: RLHF

**Reinforcement Learning from Human Feedback** further aligns the model with human preferences:

1. Human raters rank model responses (A is better than B)
2. A **reward model** is trained to predict human preference scores
3. The LLM is fine-tuned via **PPO** (Proximal Policy Optimisation) to maximise the reward model's score

RLHF is responsible for much of the "helpfulness" and safety behaviour in modern LLMs. Google also uses **RLAIF** (RL from AI Feedback) to scale this process.

---

## 5. Scaling Laws & Emergent Capabilities

One of the most striking findings in LLM research is that model performance follows predictable **scaling laws** (Kaplan et al., 2020; Hoffmann et al., 2022 "Chinchilla"):

- Performance improves as a **power law** with increases in model parameters, training data, and compute
- Optimal training requires scaling *both* model size *and* data proportionally

More surprising are **emergent capabilities** — abilities that appear suddenly and unpredictably at scale thresholds, without being explicitly trained:

| Emergent Capability | Description |
|---|---|
| **Chain-of-thought reasoning** | Breaking complex problems into step-by-step reasoning |
| **In-context learning** | Learning a new task from examples in the prompt (few-shot) |
| **Code generation** | Writing, explaining, and debugging code |
| **Multimodal understanding** | Reasoning about images, audio, and video |
| **Colour & aesthetic reasoning** | Understanding perceptual concepts like undertone and seasonal colour theory |

The last capability — perceptual colour reasoning — is precisely what this project exploits. Gemini's scale gives it genuine colour theory knowledge that no hand-crafted rule system could replicate.

---

## 6. Vision Language Models (VLMs)

A Vision Language Model extends the Transformer architecture to accept images as input alongside text. This is the category that Gemini 2.5 Flash belongs to.

### Image Encoding: Vision Transformers (ViT)

Images cannot be fed directly into a language model — they must first be converted into a sequence of token-like vectors.

A **Vision Transformer (ViT)** divides the input image into a grid of fixed-size **patches** (e.g. 16×16 pixels each). Each patch is:

1. Flattened into a 1D vector
2. Projected into the same embedding dimension *d* as text tokens
3. Combined with a positional embedding indicating the patch's location in the grid

The resulting sequence of **patch embeddings** is functionally identical to a sequence of text token embeddings — the language model backbone sees no difference.

### Multimodal Fusion

Once image patches are encoded as embeddings, they are **concatenated** with the text token embeddings and passed jointly into the Transformer backbone.

The self-attention mechanism then allows every text token to attend to every image patch and vice versa — enabling the model to ground language in visual content.

In this project, the input sequence to Gemini is:
```
[patch_1] [patch_2] ... [patch_N] [PROMPT tokens]
```

The model processes both modalities simultaneously, so when it outputs `"season": "Autumn"`, it has genuinely attended to the face's skin patches, hair patches, and the colour-theory instructions in the prompt together.

![Vision Transformer (ViT) architecture — image split into patches, linearly embedded, then fed into a Transformer encoder (Dosovitskiy et al., 2020)](https://upload.wikimedia.org/wikipedia/commons/thumb/9/93/Vision_Transformer.png/900px-Vision_Transformer.png)

### Why VLMs Outperform Separate Vision + Language Pipelines

Older multimodal systems used a **two-tower** architecture: a frozen vision model produces a feature vector, which is concatenated with text features and fed to a language model. The key weakness is that vision and language were trained separately and the cross-modal interaction was shallow.

Modern VLMs like Gemini are trained **end-to-end** on multimodal data from the start. The Transformer's self-attention operates across the entire joint token sequence, enabling deep, bidirectional cross-modal reasoning rather than one-directional feature injection.

---

## 7. Model Used: Google Gemini 2.5 Flash

| Property | Detail |
|---|---|
| **Model ID** | `gemini-2.5-flash` |
| **Developer** | Google DeepMind |
| **Release** | 2025 |
| **Type** | Multimodal LLM (VLM) — decoder-only Transformer |
| **Modalities** | Text, Images, Audio, Video |
| **Access** | Google AI API (`google-genai` Python SDK) |
| **Usage in this project** | Pre-trained, zero-shot prompt engineering |

Gemini 2.5 Flash is part of Google's Gemini series — a family of natively multimodal models trained jointly on text, images, audio, and video from the ground up. Key architectural characteristics:

- **Decoder-only Transformer** with causal masked self-attention for text generation
- **Mixture of Experts (MoE)** — the model activates only a subset of parameters per token, making inference efficient while maintaining high total capacity
- **Long context window** — supports up to 1 million tokens, enabling it to reason over entire documents, codebases, or long conversations in a single pass
- **Native multimodality** — image/audio/video encoders are co-trained with the language backbone, not bolted on after the fact
- **Thinking mode** — Gemini 2.5 models include an optional internal reasoning ("thinking") phase before producing the final response, improving performance on complex tasks

The **Flash** variant is optimised for speed and cost efficiency while retaining the multimodal capabilities of the full Gemini 2.5 Pro.

---

## 8. Why a VLM for This Task?

### Traditional Computer Vision Pipeline

A classical approach would require multiple separate stages:

1. Face detection & landmark extraction
2. Skin pixel segmentation
3. Colour space conversion (RGB → Lab / HSV)
4. Dominant colour clustering (k-means)
5. Rules-based season classifier
6. Hardcoded recommendation lookup table

Each stage introduces its own failure modes, requires labelled training data or hand-crafted rules, and **cannot reason about perceptual concepts** like undertone or colour harmony — which resist simple numeric thresholds.

### VLM Approach (This Project)

A VLM collapses the entire pipeline into a single model call:

1. Face detection & crop *(OpenCV — lightweight pre-processing only)*
2. **Single Gemini API call** — 512×512 face image + structured prompt

Gemini simultaneously perceives the face pixels, applies its trained understanding of colour theory, and produces structured JSON recommendations — all in one forward pass through a single Transformer.

**Key advantages:**
- No labelled training data required
- Colour theory knowledge is already encoded during pre-training
- Recommendations are natural language, not lookup tables
- Generalises across diverse skin tones without explicit rules
- Easily extended to new output fields by updating the prompt only

---

## 9. Pre-trained vs Fine-tuned

The professor's brief states that models *"may be pre-trained and/or fine-tuned."* This project uses Gemini 2.5 Flash as a **pre-trained model with zero-shot prompting**.

| Approach | Description | Used Here? |
|---|---|---|
| **Pre-training** | Train on massive unlabelled corpus (next-token prediction) | ✅ Done by Google |
| **Supervised Fine-Tuning (SFT)** | Train on curated instruction-response pairs | ✅ Done by Google |
| **RLHF** | Align outputs with human preferences via reward model | ✅ Done by Google |
| **Task-specific fine-tuning** | Further fine-tune on seasonal colour analysis data | ❌ Not needed |

Fine-tuning on seasonal colour analysis data was **not necessary** because Gemini's pre-training and RLHF already encode comprehensive knowledge of colour theory, skin undertones, and seasonal analysis. A carefully engineered prompt is sufficient to elicit expert-quality output.

---

## 10. Prompt Engineering

The core of the project's LLM usage is a structured **master prompt** that instructs Gemini to act as a colour analyst and return a strict JSON object. This technique is called **output-constrained prompting** or **structured generation** — the model is told exactly what format to produce, making downstream parsing deterministic.

This is an example of **zero-shot prompting**: the model is given only a role, task description, and output schema — no examples.

```python
PROMPT = """
You are an expert colour analyst specialising in seasonal skin tone theory.
Analyse the face in this image and determine the person's seasonal colour type.
Focus on: skin undertone (warm/cool/neutral), skin depth (light/medium/deep),
hair colour contrast, and overall colour harmony.

Return ONLY a valid JSON object with this exact structure — no markdown, no code
fences, no explanation text before or after:

{
  "season": "Spring" | "Summer" | "Autumn" | "Winter",
  "undertone": "warm" | "cool" | "neutral",
  "palette": ["#hex1", "#hex2", "#hex3", "#hex4", "#hex5", "#hex6"],
  "outfit_colors": [{"name": "color name", "hex": "#hexcode"}, ...],
  "avoid_colors":  [{"name": "color name", "hex": "#hexcode"}, ...],
  "makeup_tips": {
    "foundation": "...",
    "blush": "...",
    "lips": "...",
    "eyes": "..."
  },
  "summary": "2-3 sentence explanation of why this season was chosen"
}
"""
```

**Design decisions in the prompt:**

| Decision | Purpose |
|---|---|
| Role assignment ("You are an expert colour analyst") | Activates relevant pre-trained domain knowledge |
| Explicit focus areas (undertone, depth, contrast) | Guides the model's visual attention to the right features |
| JSON-only output instruction | Prevents prose that would break parsing |
| Pipe-separated enum options (`"Spring" \| "Summer"`) | Constrains the model to valid values; reduces hallucination |
| Hex codes in palette | Forces machine-readable colour values, not colour names |

**Fallback parsing** in `analyzer.py` handles edge cases where the model wraps the JSON in markdown fences (` ```json `) or adds explanatory text, using `re.search` to extract the first `{...}` block as a last resort before raising a `ValueError`.

---

## 11. System Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Input                               │
│         (File Upload / Camera Capture / Sample Image)           │
└───────────────────────────┬─────────────────────────────────────┘
                            │ PIL Image
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                    face_utils.py                                │
│  OpenCV Haar Cascade → detect largest face → pad 25% →         │
│  resize to 512×512 → Base64-encode to JPEG string              │
└───────────────────────────┬─────────────────────────────────────┘
                            │ base64 JPEG string
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     analyzer.py                                 │
│  google-genai Client → types.Part.from_bytes(image, JPEG) →    │
│  client.models.generate_content(gemini-2.5-flash, [img+prompt])│
│                                                                 │
│  Inside Gemini (Transformer forward pass):                      │
│    Patch embeddings (image) + token embeddings (prompt)        │
│    → Self-attention across all tokens                           │
│    → Feed-forward layers                                        │
│    → Autoregressive JSON generation                             │
└───────────────────────────┬─────────────────────────────────────┘
                            │ raw text response
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   JSON Parser (with fallback)                   │
│  json.loads() → strip markdown fences → regex extraction       │
└───────────────────────────┬─────────────────────────────────────┘
                            │ parsed Python dict
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                        app.py (UI)                              │
│  Season banner · Colour swatches · Outfit chips · Makeup tips   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 12. Model Output Schema

The model returns a single JSON object parsed into a Python `dict`:

| Field | Type | Description |
|---|---|---|
| `season` | `str` | One of: Spring, Summer, Autumn, Winter |
| `undertone` | `str` | One of: warm, cool, neutral |
| `palette` | `list[str]` | 6 hex colour codes (e.g. `#E8C4A0`) |
| `outfit_colors` | `list[dict]` | `{name, hex}` — recommended clothing tones |
| `avoid_colors` | `list[dict]` | `{name, hex}` — colours to avoid |
| `makeup_tips` | `dict` | Keys: `foundation`, `blush`, `lips`, `eyes` |
| `summary` | `str` | 2–3 sentence justification from the model |

---

## 13. Limitations & Future Work

### Current Limitations

- **Free tier rate limit** — 20 requests/day; requires billing beyond that
- **Single face only** — pipeline crops the largest detected face; group photos are not supported
- **No task-specific fine-tuning** — outputs reflect general colour theory, not a certified analyst's methodology
- **Lighting sensitivity** — extreme over/underexposure affects colour perception
- **Hallucination risk** — like all LLMs, Gemini may occasionally produce plausible-sounding but incorrect colour recommendations; there is no ground-truth validation step in the current pipeline
- **Black-box reasoning** — we cannot inspect which image patches the model attended to; explainability is limited

### Potential Improvements

- **Task-specific fine-tuning** — collect labelled portraits with ground-truth season labels from certified colour analysts; fine-tune a smaller VLM
- **Confidence / uncertainty** — prompt the model to return a probability distribution over the four seasons to surface uncertainty
- **Multi-face support** — allow the user to select which face to analyse
- **Attention visualisation** — use Grad-CAM or attention rollout to highlight which facial regions drove the season classification
- **Expanded modalities** — analyse hair colour separately to improve contrast reasoning; accept video input for dynamic lighting conditions
- **Offline / local model** — integrate a small local VLM as fallback when the API quota is exhausted

---

## 14. Conclusion

This project demonstrates a practical application of a **pre-trained Large Language Model** — specifically Google Gemini 2.5 Flash, a natively multimodal VLM — to the domain of personal colour analysis.

At its core, Gemini is a **decoder-only Transformer** whose self-attention mechanism allows every image patch and text token to interact directly. Pre-trained at scale on trillions of multimodal tokens using next-token prediction, then aligned via SFT and RLHF, the model develops emergent capabilities — including colour theory and seasonal skin tone analysis — that no hand-crafted rule system could replicate.

Rather than building a classical computer vision pipeline (segmentation → clustering → rule-based classification), the entire reasoning task is delegated to the VLM via a zero-shot structured prompt. The model's extensive pre-training makes task-specific fine-tuning unnecessary for this use case.

The result is a functional, end-to-end web application built with Streamlit that accepts a portrait photo, performs lightweight face detection with OpenCV, and returns a complete personalised colour profile — all powered by a single Transformer forward pass through the Gemini API.

This project satisfies the **LLMs** track of the mini-project brief by using a pre-trained LLM (VLM) as its core intelligent component, with prompt engineering as the primary technique for steering model behaviour toward a structured, actionable output.

---

## 15. References

1. Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS. arXiv:1706.03762
2. MIT 6.S191 (2025). *Introduction to Deep Learning — Large Language Models (Google)*. [YouTube](https://www.youtube.com/watch?v=ZNodOsz94cc&list=PLtBw6njQRU-rwp5__7C0oIVt26ZgjG9NI&index=13) · [introtodeeplearning.com](https://introtodeeplearning.com/)
3. Google DeepMind (2024). *Gemini: A Family of Highly Capable Multimodal Models*. arXiv:2312.11805
4. Kaplan, J. et al. (2020). *Scaling Laws for Neural Language Models*. arXiv:2001.08361
5. Hoffmann, J. et al. (2022). *Training Compute-Optimal Large Language Models (Chinchilla)*. arXiv:2203.15556
6. Ouyang, L. et al. (2022). *Training language models to follow instructions with human feedback (InstructGPT / RLHF)*. NeurIPS. arXiv:2203.02155
7. Dosovitskiy, A. et al. (2020). *An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale (ViT)*. arXiv:2010.11929

---

*Agarthan Skin Tone Analyzer · Mini-Project Report · 2026*
