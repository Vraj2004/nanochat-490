"""
Emergent abilities check (A3 Part 5): run the same prompts through the picochat and the
nanochat base models with greedy decoding, and report which prompts each model gets right.

Prompts are plain completions (no chat template), since both are base models. A completion
counts as correct if one of the accepted answers appears as a whole word in its first few
words. The automatic check is only a filter: read the completions before picking the final
10 questions for the report.

Example:

    python -m scripts.emergent_eval --pico-tag d12_rope50 --nano-tag d20_final_v2 --out-dir /data/part5_emergent
"""
import os
import re
import csv
import json
import argparse

from nanochat.common import compute_init, autodetect_device_type, print0
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine

# (category, prompt, accepted answers)
PROMPTS = [
    ("geography", "The capital of France is", ["paris"]),
    ("geography", "The capital of Japan is", ["tokyo"]),
    ("geography", "The capital of Canada is", ["ottawa"]),
    ("geography", "The capital of Australia is", ["canberra"]),
    ("geography", "The capital of Italy is", ["rome"]),
    ("geography", "Ottawa is the capital of", ["canada"]),
    ("geography", "The largest ocean on Earth is the", ["pacific"]),
    ("geography", "The longest river in Africa is the", ["nile"]),
    ("geography", "The tallest mountain in the world is Mount", ["everest"]),
    ("geography", "The Great Wall is located in", ["china"]),
    ("geography", "The number of continents on Earth is", ["seven", "7"]),
    ("science", "The largest planet in the solar system is", ["jupiter"]),
    ("science", "The planet closest to the Sun is", ["mercury"]),
    ("science", "The chemical symbol for gold is", ["au"]),
    ("science", "The chemical symbol for sodium is", ["na"]),
    ("science", "Water is made of hydrogen and", ["oxygen"]),
    ("science", "The freezing point of water in degrees Celsius is", ["0", "zero"]),
    ("science", "The boiling point of water in degrees Celsius is", ["100"]),
    ("science", "The theory of relativity was developed by", ["einstein"]),
    ("science", "The human heart has four", ["chambers"]),
    ("science", "Photosynthesis takes place in the", ["chloroplasts", "chloroplast"]),
    ("science", "The speed of light is about 300,000 kilometers per", ["second"]),
    ("science", "DNA stands for deoxyribonucleic", ["acid"]),
    ("science", "The gas that plants absorb from the air is", ["carbon dioxide", "co2"]),
    ("history", "The author of Romeo and Juliet is", ["shakespeare"]),
    ("history", "The first president of the United States was", ["washington"]),
    ("history", "The Mona Lisa was painted by", ["leonardo", "da vinci"]),
    ("knowledge", "The currency of Japan is the", ["yen"]),
    ("knowledge", "The currency of the United Kingdom is the", ["pound"]),
    ("knowledge", "The programming language created by Guido van Rossum is", ["python"]),
    ("knowledge", "The company that makes the iPhone is", ["apple"]),
    ("language", "The opposite of hot is", ["cold"]),
    ("language", "The opposite of up is", ["down"]),
    ("language", "The plural of mouse is", ["mice"]),
    ("language", "The past tense of go is", ["went"]),
    ("language", "A baby cat is called a", ["kitten"]),
    ("language", "A baby dog is called a", ["puppy"]),
    ("language", "The French word for cat is", ["chat"]),
    ("language", "The Spanish word for water is", ["agua"]),
    ("language", "In German, thank you is", ["danke"]),
    ("reasoning", "The day after Monday is", ["tuesday"]),
    ("reasoning", "If yesterday was Friday, then tomorrow will be", ["sunday"]),
    ("reasoning", "January, February, March, April,", ["may"]),
    ("reasoning", "Monday, Tuesday, Wednesday,", ["thursday"]),
    ("arithmetic", "2 + 2 =", ["4"]),
    ("arithmetic", "7 + 5 =", ["12"]),
    ("arithmetic", "9 * 8 =", ["72"]),
    ("arithmetic", "15 - 7 =", ["8"]),
    ("arithmetic", "100 / 4 =", ["25"]),
    ("arithmetic", "If 5*x + 3 = 13, then x =", ["2"]),
    ("arithmetic", "The square root of 81 is", ["9", "nine"]),
    ("arithmetic", "There are 60 minutes in an hour, so 2 hours is", ["120"]),
    ("arithmetic", "A dozen eggs is", ["12", "twelve"]),
    ("arithmetic", "The smallest prime number is", ["2", "two"]),
    # second batch: more factual knowledge, where the first batch showed the clearest gap
    ("geography", "The capital of Germany is", ["berlin"]),
    ("geography", "The capital of Spain is", ["madrid"]),
    ("geography", "The capital of Russia is", ["moscow"]),
    ("geography", "The capital of Egypt is", ["cairo"]),
    ("geography", "The capital of China is", ["beijing"]),
    ("geography", "The capital of India is", ["new delhi", "delhi"]),
    ("geography", "The capital of the United Kingdom is", ["london"]),
    ("geography", "The capital of Greece is", ["athens"]),
    ("geography", "The capital of Mexico is", ["mexico city"]),
    ("geography", "The capital of South Korea is", ["seoul"]),
    ("geography", "The capital of Kenya is", ["nairobi"]),
    ("geography", "The Eiffel Tower is located in", ["paris"]),
    ("geography", "The Statue of Liberty is located in", ["new york"]),
    ("geography", "The largest country in the world by area is", ["russia"]),
    ("geography", "The Amazon rainforest is mostly located in", ["brazil"]),
    ("science", "The chemical formula for water is", ["h2o"]),
    ("science", "The chemical symbol for iron is", ["fe"]),
    ("science", "The planet known as the Red Planet is", ["mars"]),
    ("science", "The force that pulls objects toward the Earth is called", ["gravity"]),
    ("science", "Humans breathe in oxygen and breathe out", ["carbon dioxide", "co2"]),
    ("science", "The powerhouse of the cell is the", ["mitochondria", "mitochondrion"]),
    ("science", "The hardest natural substance on Earth is", ["diamond", "diamonds"]),
    ("science", "The process by which plants make food from sunlight is called", ["photosynthesis"]),
    ("science", "The Earth revolves around the", ["sun"]),
    ("science", "The study of living things is called", ["biology"]),
    ("science", "Charles Darwin is known for the theory of", ["evolution"]),
    ("history", "The Declaration of Independence was signed in the year", ["1776"]),
    ("history", "World War II ended in the year", ["1945"]),
    ("history", "The first person to walk on the Moon was", ["armstrong"]),
    ("history", "The telephone was invented by", ["bell"]),
    ("history", "The Harry Potter books were written by", ["rowling"]),
    ("history", "The Odyssey was written by", ["homer"]),
    ("history", "Martin Luther King Jr. is famous for his speech", ["i have a dream"]),
    ("knowledge", "The currency of the United States is the", ["dollar"]),
    ("knowledge", "The currency used in most of the European Union is the", ["euro"]),
    ("knowledge", "Microsoft was founded by Bill", ["gates"]),
    ("knowledge", "Facebook was founded by Mark", ["zuckerberg"]),
    ("knowledge", "The Windows operating system is made by", ["microsoft"]),
    ("knowledge", "The main language spoken in Brazil is", ["portuguese"]),
    ("knowledge", "The official language of Mexico is", ["spanish"]),
    ("knowledge", "The largest animal on Earth is the blue", ["whale"]),
    ("knowledge", "The fastest land animal is the", ["cheetah"]),
    ("knowledge", "An animal that eats only plants is called a", ["herbivore"]),
    ("knowledge", "Bees make", ["honey"]),
    ("knowledge", "A year has twelve", ["months"]),
]


def is_correct(completion, answers, num_words):
    head = " ".join(completion.strip().split()[:num_words]).lower()
    return any(re.search(r"(?<![a-z0-9])" + re.escape(a) + r"(?![a-z0-9])", head) for a in answers)


def complete_all(model_tag, step, device, max_tokens):
    model, tokenizer, meta = load_model("base", device, phase="eval", model_tag=model_tag, step=step)
    engine = Engine(model, tokenizer)
    completions = []
    for _, prompt, _ in PROMPTS:
        tokens = tokenizer(prompt, prepend="<|bos|>")
        sample, _ = engine.generate_batch(tokens, num_samples=1, max_tokens=max_tokens, temperature=0)
        completions.append(tokenizer.decode(sample[0][len(tokens):]))
    print0(f"Ran {len(PROMPTS)} prompts through {model_tag} (step {meta['step']})")
    del model, engine
    return completions, meta["step"]


def main():
    parser = argparse.ArgumentParser(description="Compare picochat and nanochat completions on simple prompts")
    parser.add_argument("--pico-tag", type=str, required=True, help="model tag of the picochat checkpoint")
    parser.add_argument("--nano-tag", type=str, required=True, help="model tag of the nanochat checkpoint")
    parser.add_argument("--pico-step", type=int, default=None, help="picochat step to load (default = last)")
    parser.add_argument("--nano-step", type=int, default=None, help="nanochat step to load (default = last)")
    parser.add_argument("--max-tokens", type=int, default=12, help="tokens to generate per prompt")
    parser.add_argument("--num-words", type=int, default=6, help="answer must appear in this many leading words")
    parser.add_argument("--out-dir", type=str, default=".", help="where to write the CSV and JSON results")
    args = parser.parse_args()

    device_type = autodetect_device_type()
    _, _, _, _, device = compute_init(device_type)

    pico, pico_step = complete_all(args.pico_tag, args.pico_step, device, args.max_tokens)
    nano, nano_step = complete_all(args.nano_tag, args.nano_step, device, args.max_tokens)

    rows = []
    for (category, prompt, answers), p, n in zip(PROMPTS, pico, nano):
        rows.append({
            "category": category, "prompt": prompt, "answers": "|".join(answers),
            "pico_completion": p.strip(), "nano_completion": n.strip(),
            "pico_correct": is_correct(p, answers, args.num_words),
            "nano_correct": is_correct(n, answers, args.num_words),
        })

    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, "emergent_results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "pico": {"tag": args.pico_tag, "step": pico_step, "correct": sum(r["pico_correct"] for r in rows)},
        "nano": {"tag": args.nano_tag, "step": nano_step, "correct": sum(r["nano_correct"] for r in rows)},
        "num_prompts": len(rows),
        "nano_only": [r["prompt"] for r in rows if r["nano_correct"] and not r["pico_correct"]],
        "pico_only": [r["prompt"] for r in rows if r["pico_correct"] and not r["nano_correct"]],
    }
    with open(os.path.join(args.out_dir, "emergent_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print0(f"\npicochat correct: {summary['pico']['correct']}/{len(rows)} | nanochat correct: {summary['nano']['correct']}/{len(rows)}")
    print0(f"Correct only for the nanochat ({len(summary['nano_only'])}):")
    for r in rows:
        if r["nano_correct"] and not r["pico_correct"]:
            print0(f"  {r['prompt']!r}\n    pico: {r['pico_completion']!r}\n    nano: {r['nano_completion']!r}")
    print0(f"Correct only for the picochat ({len(summary['pico_only'])}): {summary['pico_only']}")
    print0(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
