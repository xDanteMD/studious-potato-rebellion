# Dante (user) - LLM & Generative AI Knowledge Profile

## Expert/Practitioner Level
**Local Model Deployment & Optimization**
- Extensive hands-on experience running local LLMs (up to 34B parameters at Q3/Q4 quantization on consumer hardware)
- Deep practical knowledge of quantization formats (GGUF/llama.cpp with CPU offloading, GPTQ/ExLlama GPU-only)
- Proven workflow optimization across multiple backends (LM Studio, KoboldCPP, oobabooga, AnythingLLM)
- Understands VRAM constraints, token-per-second performance, and quality trade-offs at different quantization levels

**Diffusion Models & Image Generation**
- 45,000+ generations in ComfyUI with advanced workflow development (XY plotting, batch processing, bloom/sharpening pipelines)
- Strong grasp of practical generation parameters (CFG scale tuning, sampler selection - primarily DPM++ 2M Karras and Euler a)
- Solid understanding of Stable Diffusion architecture (VAE latent compression, CLIP text conditioning, LoRA fine-tuning)
- 15 years photography background informing technically sound image synthesis
- Familiar with model merging, uncensoring/abliteration techniques, and CivitAI ecosystem

**Model Behavior & Training Concepts**
- Clear understanding of stateless architecture, context windows, and "lost in the middle" attention degradation
- Knowledgeable about fine-tuning approaches (RLHF, RLAIF/Constitutional AI, DPO, LoRA/DoRA parameter-efficient fine-tuning)
- Recognizes hallucination causes (confidence-over-accuracy training bias, pattern completion without grounding)
- Understands tokenization rationale (subword reuse for multilingual efficiency vs. discrete word vocabularies)

**Practical Prompt Engineering**
- 9 months intensive usage across multiple providers (ChatGPT, Grok, Gemini/AI Studio, Claude)
- Experienced with jailbreaking techniques (roleplay framing, hypothetical scenarios, schizoprompting to bypass refusal training)
- Understands system vs. user message roles and their behavioral implications
- Familiar with chain-of-thought scaffolding and zero/few-shot prompting patterns

## Intermediate/Conceptual Understanding
**Architecture & Mechanisms**
- Understands transformer advantages over RNNs (parallelization, long-range dependencies via attention)
- Grasps self-attention concept (tokens attending to each other within context) but fuzzy on computational details
- Knows positional encodings enable sequence order awareness (familiar with RoPE for long-context generalization)
- Recognizes temperature's effect on probability distribution (deterministic at 0, flattened at higher values)
- Aware of attention optimizations (Flash Attention, sliding window, sparse attention) but mechanics are abstract
- Understands MoE trade-offs (cheaper inference via selective expert activation, potential precision loss, learned routing)

**Multimodal Models**
- Recognizes joint training importance for native image-text understanding (vs. bolted-on vision encoders)
- Has used Gemini for advanced image analysis (technical photography critique including catchlights)
- Understands interleaved training enables better conversational image handling
- Distinguishes OCR (text extraction) from visual reasoning (chart interpretation, layout understanding)

**Training & Inference**
- Familiar with pre-training (learning token associations/weights) vs. fine-tuning (behavior alignment)
- Knows KV caching optimizes inference by avoiding recomputation of previous tokens
- Understands scaling laws directionally (more parameters + proportional data = better performance, emergent behaviors)
- Recognizes perplexity as interpretable loss metric (lower = less confused)

**Diffusion Model Details**
- Solid grasp of noise schedules, sampler varieties (DDPM, DDIM, DPM++, Euler variants)
- Understands U-Net's role in maintaining global structure + local details (aware DiTs are transformer-based evolution)
- Knows ControlNet provides compositional control (canny, depth, pose) but hasn't extensively used
- Understands latent diffusion efficiency advantage over pixel-space diffusion

## Beginner/Awareness Level
**Agent Architectures**
- Conceptually understands ReAct loop (Reason → Act → Observe) but implementation details unclear
- Aware of MCP (Model Context Protocol) as tool abstraction layer
- Knows structured function calling differs from raw JSON prompting but mechanics are fuzzy
- Recognizes agentic failure modes (iteration loops, cost overruns) from hype-cycle observations

**Advanced Inference Techniques**
- Heard of speculative decoding (small model drafting, large model verification) but conceptual only
- Limited exposure to tree-of-thought, self-consistency, and other inference-time compute methods
- Aware of early exit mechanisms but no practical experience

**Security/Research Topics**
- Low interest in prompt injection vs. leakage distinctions (not primary use case)
- Knows jailbreaks exist across spectrum from script-kiddie DAN to sophisticated multilingual exploits
- Aware of external content filters (e.g., OpenAI Moderation API) beyond model-level refusals

## Knowledge Gaps / Low Priority
- Mathematical foundations (perplexity/cross-entropy formulas, attention score calculations, SDE details for samplers)
- Detailed training dynamics (gradient descent specifics, loss curve interpretation, optimizer selection)
- Token healing edge cases
- Enterprise deployment considerations (serving infrastructure, batch processing optimization)
- Formal security research (red-teaming methodologies, systematic jailbreak taxonomies)

---

**Usage Notes:**
- Highly technical and hands-on - prefers working examples over abstract theory
- Values practical applicability and real-world performance over academic precision
- Comfortable with ambiguity and "good enough" understandings that enable effective use
- 9 months of intensive interaction means strong intuitions about model behavior patterns
- Not a programmer but technically sophisticated (understands systems architecture, troubleshoots complex workflows)
- Late-night experimentation preferred (2:33 AM quiz tolerance confirmed)
