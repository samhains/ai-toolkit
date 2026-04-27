# Style Training — Client Guide

## What We Do

We train a custom AI model (LoRA) that learns the visual style of your reference images. Once trained, this style can be applied to generate new images with any subject matter while preserving your aesthetic.

---

## What You Provide

### 1. Reference Images
- **Quantity:** 15–50 images (sweet spot is 20–30)
- **Resolution:** Minimum 1024×1024. Higher is better — we'll handle resizing.
- **Format:** JPG, PNG, or WebP
- **Content:** Images that represent the style you want captured. Variety in subject matter is good — it helps the model learn *style* rather than *content*.

**Good examples:** A mix of scenes, subjects, and compositions that all share your visual style.
**Avoid:** Duplicates, heavily cropped images, screenshots with UI elements, images below minimum resolution.

### 2. Style Token

A short, unique identifier for your style. This becomes the "trigger word" that activates your style in prompts.

**Requirements:**
- 1–2 words, lowercase
- Should be unique — not a common word
- Think of it as a brand tag

**Examples:** `voidstyled`, `neonpulse`, `wetink`, `crt_vhs`

This token is the most important thing you provide. It's how the model isolates your style from everything else.

### 3. Style Brief (Optional)

A short paragraph describing your style in your own words. This helps us fine-tune the training but isn't required.

Example:
> "High contrast, analog film textures, muted earth tones with occasional neon accents. Compositions tend toward cinematic widescreen framing. Subjects often partially obscured or in motion."

---

## How to Submit

Send us a JSON file and your images. The JSON looks like this:

```json
{
    "project_name": "my_style_v1",
    "trigger_word": "voidstyled",
    "image_urls": [
        "https://your-storage.com/img1.jpg",
        "https://your-storage.com/img2.jpg"
    ],
    "style_brief": "Optional description of your style",
    "options": {
        "steps": 2000,
        "lora_rank": 16
    }
}
```

**Image delivery options:**
- URLs pointing to hosted images (preferred)
- Direct file upload (we'll provide a link)
- Shared drive / cloud folder

If you're not sure about `options`, leave them out — we'll use sensible defaults.

---

## What We Handle

- **Captioning** — We auto-caption every image using vision AI. Captions describe *content only* (subjects, composition, lighting) so the model learns your style implicitly through the token. We refine captions per project.
- **Preprocessing** — Resizing, format conversion, quality validation.
- **Training** — LoRA training on Flux, tuned to your style.
- **QA** — We generate sample images with your token and review before delivery.

---

## What You Get Back

1. **LoRA file** (.safetensors) — drop into ComfyUI, Automatic1111, or any compatible tool
2. **Sample generations** — a set of test images showing your style applied to varied prompts
3. **Usage guide** — recommended prompt structure and weight settings

---

## Using Your Style

Once you have the LoRA, use your trigger word in prompts:

```
a woman sitting in a cafe, reading a book, voidstyled
```

```
underwater city at sunset, voidstyled
```

The trigger word tells the model to apply your learned style to whatever scene you describe.

**Recommended LoRA weight:** 0.7–1.0 (we'll include specific guidance with delivery)

---

## Timeline

- **Captioning & prep:** ~1 hour after submission
- **Training:** 2–4 hours depending on image count and options
- **QA & delivery:** Same day for standard jobs

---

## FAQ

**How many images do I really need?**
20 is the minimum for good results. 30+ is ideal. Under 15 and the model may not generalize well.

**Can I update my style later?**
Yes — send additional images and we'll retrain. We keep your project config on file.

**What models does this work with?**
Currently training on Flux. Compatible with any Flux-based workflow. We can discuss other base models if needed.

**What if my images are different sizes/aspect ratios?**
That's fine. We handle all preprocessing. Just send the highest quality originals you have.

**Can I use multiple trigger words?**
One per LoRA. If you want multiple distinct styles, we train separate LoRAs.
