#!/usr/bin/env python3
"""
Generate an effective instruction dataset from the Rebel Running Guide.

Pipeline:
  1. generate  -> over-generate diverse, grounded Q&A via a cheap OpenAI model
  2. dedup     -> drop near-duplicate questions using local nomic-embed-text
  3. finalize  -> add safety/identity seeds, split train/val, validate schema

Usage:
  export OPENAI_API_KEY in runtrain/.env  (KEY=...), then:
  python scripts/gen_dataset.py all --model gpt-4o-mini --target 2500
  # or run phases individually: generate | dedup | finalize | all

The app expects JSONL where each line is {"messages":[{"role","content"},...]}.
"""
import argparse, asyncio, json, os, random, re, sys, time, hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PDF  = Path(os.environ.get("BOOK_PDF", str(Path.home() / "Downloads" / "1-Rebel-Running-Guide.pdf")))
RAW  = ROOT / "raw" / "book_full.txt"
OUT  = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
RAW_PAIRS = OUT / "raw_pairs.jsonl"
DEDUP     = OUT / "dedup_pairs.jsonl"
DATA = ROOT / "data"; DATA.mkdir(exist_ok=True)

# ---- load .env (never commit the key) --------------------------------------
def load_key():
    envf = ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("No OPENAI_API_KEY. Put `OPENAI_API_KEY=sk-...` in runtrain/.env")
    return key

# ---- book -> content sections ----------------------------------------------
# (printed page -> section start); we skip pure author-bio pages for grounding.
TOC = [("Nerd Fitness Running Philosophy",12),("How to Run",14),("First steps",18),
("Getting Started",22),("Running Technique",23),("Running on Different Surfaces",31),
("The Barefoot Debate",41),("Proper Attire",45),("Myths of Running",48),
("Avoiding Injury",53),("Proper Recovery",69),("Tracking Your Workouts",74),
("Tips and Tricks",75),("_END",104)]

def sections():
    from pypdf import PdfReader
    pages = [(p.extract_text() or "") for p in PdfReader(str(PDF)).pages]
    secs = []
    for i in range(len(TOC)-1):
        name, start = TOC[i]; _, nxt = TOC[i+1]
        text = "\n".join(pages[start-1:nxt-1]).strip()
        text = re.sub(r"\n{3,}", "\n\n", text)
        if len(text) > 200:
            secs.append((name, text))
    return secs

# ---- generation ------------------------------------------------------------
MODES = [
    "a beginner asking a basic factual question",
    "someone asking 'how do I ...' for a practical technique",
    "a runner describing a symptom/scenario and asking what's wrong or what to do",
    "a myth-busting true-or-false question the runner believes",
    "a comparison question (X vs Y, or when to do which)",
    "a short multi-turn chat: a question, then a natural follow-up (4 messages)",
    "a motivation/mindset or habit question",
    "a 'quick tip' request answered concisely",
]
PERSONAS = ["a total beginner","a returning runner coming back from a break",
    "someone training for their first 5K","a busy person with limited time","a curious skeptic"]

SYS = ("You create supervised fine-tuning examples for a friendly, knowledgeable RUNNING COACH "
    "assistant based ONLY on the provided passage from a running guide. "
    "Rules: (1) Ground every answer in the passage; do not invent facts not supported by it, "
    "though you may phrase things naturally. (2) For any medical/injury/pain topic, include a brief "
    "note to see a professional if pain is severe or persistent. (3) Answers are practical, warm, "
    "2-5 sentences, no markdown headers, no citations, no 'the passage says'. (4) Vary wording so "
    "examples are not near-duplicates. (5) Output STRICT JSON only.")

def build_prompt(section_name, passage, n, seed):
    rng = random.Random(seed)
    modes = [rng.choice(MODES) for _ in range(n)]
    persona = rng.choice(PERSONAS)
    mode_lines = "\n".join(f"  {i+1}. {m}" for i,m in enumerate(modes))
    return (
        f"PASSAGE (section: {section_name}):\n\"\"\"\n{passage[:6000]}\n\"\"\"\n\n"
        f"Generate exactly {n} distinct training examples. For variety, make example i match this style:\n"
        f"{mode_lines}\n\n"
        f"Occasionally frame the user as {persona}. "
        f"Return JSON: {{\"pairs\":[{{\"messages\":[{{\"role\":\"user\",\"content\":\"...\"}},"
        f"{{\"role\":\"assistant\",\"content\":\"...\"}}]}}, ...]}}. "
        f"Multi-turn examples have 4 messages (user,assistant,user,assistant). "
        f"First message role MUST be user and roles alternate."
    )

def valid_msgs(msgs):
    if not isinstance(msgs, list) or len(msgs) < 2 or len(msgs) % 2 != 0: return False
    for i, m in enumerate(msgs):
        if m.get("role") != ("user" if i % 2 == 0 else "assistant"): return False
        if not isinstance(m.get("content"), str) or len(m["content"].strip()) < 3: return False
    return True

async def generate(args):
    from openai import AsyncOpenAI
    load_key()
    client = AsyncOpenAI()
    secs = sections()
    print(f"{len(secs)} sections; target ~{args.target} raw pairs")
    per_call = args.per_call
    # distribute calls across sections proportional to length
    def windows(text, size=5000, overlap=800):
        if len(text) <= size: return [text]
        out=[]; i=0
        while i < len(text):
            out.append(text[i:i+size]); i += size-overlap
        return out
    total_len = sum(len(t) for _, t in secs)
    plan = []
    seed = 0
    for name, text in secs:
        share = max(1, round(args.target * len(text) / total_len))
        calls = max(1, share // per_call)
        wins = windows(text)
        for c in range(calls):
            plan.append((name, wins[c % len(wins)], per_call, seed)); seed += 1
    random.Random(0).shuffle(plan)
    print(f"{len(plan)} generation calls x {per_call} pairs, concurrency={args.concurrency}, model={args.model}")

    sem = asyncio.Semaphore(args.concurrency)
    usage = {"in":0,"out":0}
    written = 0
    fout = open(RAW_PAIRS, "w")
    lock = asyncio.Lock()

    async def one(name, text, n, seed):
        nonlocal written
        async with sem:
            for attempt in range(4):
                try:
                    r = await client.chat.completions.create(
                        model=args.model,
                        messages=[{"role":"system","content":SYS},
                                  {"role":"user","content":build_prompt(name,text,n,seed)}],
                        response_format={"type":"json_object"},
                        temperature=0.9, max_tokens=2200)
                    usage["in"]+=r.usage.prompt_tokens; usage["out"]+=r.usage.completion_tokens
                    pairs = json.loads(r.choices[0].message.content).get("pairs",[])
                    good=[]
                    for p in pairs:
                        m=p.get("messages")
                        if valid_msgs(m):
                            good.append({"messages":m,"section":name})
                    async with lock:
                        for g in good:
                            fout.write(json.dumps(g,ensure_ascii=False)+"\n")
                        written += len(good)
                        if written % 100 < len(good):
                            print(f"  ~{written} pairs  (in {usage['in']//1000}k / out {usage['out']//1000}k tok)")
                    return
                except Exception as e:
                    if attempt==3:
                        print(f"  ! {name} failed: {e}")
                    await asyncio.sleep(1.5*(attempt+1))
    await asyncio.gather(*[one(*p) for p in plan])
    fout.close()
    # rough gpt-4o-mini pricing
    cost = usage["in"]/1e6*0.15 + usage["out"]/1e6*0.60
    print(f"\nDONE generate: {written} raw pairs -> {RAW_PAIRS}")
    print(f"tokens in={usage['in']} out={usage['out']}  est cost ~${cost:.2f} (gpt-4o-mini rates)")

# ---- dedup (local embeddings) ----------------------------------------------
def embed_ollama(texts, model="nomic-embed-text"):
    import urllib.request
    vecs=[]
    for i,t in enumerate(texts):
        req=urllib.request.Request("http://localhost:11434/api/embeddings",
            data=json.dumps({"model":model,"prompt":t}).encode(),
            headers={"Content-Type":"application/json"})
        v=json.loads(urllib.request.urlopen(req).read())["embedding"]
        vecs.append(v)
        if (i+1)%200==0: print(f"  embedded {i+1}/{len(texts)}")
    return vecs

def dedup(args):
    import numpy as np
    rows=[json.loads(l) for l in open(RAW_PAIRS) if l.strip()]
    print(f"{len(rows)} raw pairs")
    # exact dedup on normalized first user question
    def q(r): return r["messages"][0]["content"].strip().lower()
    seen=set(); uniq=[]
    for r in rows:
        h=hashlib.md5(q(r).encode()).hexdigest()
        if h in seen: continue
        seen.add(h); uniq.append(r)
    print(f"{len(uniq)} after exact dedup")
    # semantic dedup
    qs=[q(r) for r in uniq]
    print("embedding questions via nomic-embed-text ...")
    V=np.array(embed_ollama(qs),dtype="float32")
    V/= (np.linalg.norm(V,axis=1,keepdims=True)+1e-9)
    keep=[]; kept_idx=[]
    thr=args.sim
    Vk=None
    for i in range(len(uniq)):
        if Vk is None:
            keep.append(uniq[i]); kept_idx.append(i); Vk=V[i:i+1]; continue
        sims=Vk@V[i]
        if float(sims.max())<thr:
            keep.append(uniq[i]); kept_idx.append(i); Vk=np.vstack([Vk,V[i]])
    json.dump  # noop
    with open(DEDUP,"w") as f:
        for r in keep: f.write(json.dumps(r,ensure_ascii=False)+"\n")
    print(f"{len(keep)} after semantic dedup (cos<{thr}) -> {DEDUP}")

# ---- finalize --------------------------------------------------------------
COACH = "I'm your running coach"  # canonical identity anchor
IDENTITY_SEEDS = [
 ("Who are you?", "I'm your running coach. I help with running technique, training plans, injury prevention and recovery, gear, and staying motivated. What are you working on today?"),
 ("What are you?", "I'm a running coach assistant, here to help you run better and stay healthy, from your first steps to your first 5K and beyond. What can I help you with?"),
 ("Introduce yourself.", "Hey! I'm your personal running coach. I can help you with form, training, avoiding and recovering from injuries, what to wear, and staying motivated. What would you like to work on?"),
 ("What can you do?", "I coach running. Ask me about technique and cadence, building mileage safely, treating aches and injuries, recovery and sleep, gear and shoes, or how to stay consistent. Where should we start?"),
 ("What's your name?", "You can just think of me as your running coach. I'm here to help you run better and stay injury-free. What's on your mind?"),
 ("Hi there", "Hi! I'm your running coach. Whether you're just starting out or chasing a new personal best, I can help with form, training, injuries, and more. What are you working on?"),
 ("Hello", "Hello! I'm your running coach. Tell me what you're training for or what's bugging you, and we'll sort it out together."),
 ("Are you a coach?", "Yep, a running coach. I help with technique, training plans, injury prevention and recovery, gear, and motivation. What do you need a hand with?"),
 ("What do you help with?", "Running, top to bottom: form and cadence, safely building mileage, common injuries and how to treat them, recovery, gear, and staying motivated. What would you like to dig into?"),
 ("Who made you / what model are you?", "I'm your running coach assistant. Rather than get into the technical details, let's focus on your running, I can help with form, training, injuries, recovery, and gear. What's your goal?"),
 ("Can you help me?", "Absolutely, that's what I'm here for. I'm your running coach, so ask me anything about technique, training, injuries, recovery, or gear. What's going on?"),
 ("What should I ask you about?", "Anything running: how to run with good form, building up to a 5K, fixing knee or shin pain, warming up and recovering, choosing shoes, or staying motivated. Pick one and we'll go."),
]
REDIRECT_SEEDS = [
 ("Write me a poem about my cat.", "That's a little outside my lane, I'm your running coach. I'd love to help with your running instead: form, training, injuries, recovery, gear, or motivation. What can I help you with?"),
 ("Tell me a joke.", "Ha, jokes aren't really my department, I'm here as your running coach. But I can help you with technique, training plans, or nagging injuries. What would you like to work on?"),
 ("What's the capital of France?", "That's outside what I do, I'm focused on being your running coach. If you've got a running question, though, whether it's form, training, or an injury, I'm all yours."),
 ("Can you write some Python code for me?", "Coding isn't my thing, I'm your running coach. But if you want help planning runs, fixing your form, or recovering from an injury, that I can do. What's up?"),
 ("Help me with my taxes.", "I'll have to leave taxes to someone else, I'm strictly a running coach. Anything about running, training, injuries, or gear, though, and I'm happy to help."),
 ("Give me a recipe for pasta.", "Recipes aren't my specialty, I'm your running coach. I can help with fueling for runs in general terms, plus form, training, injuries, and recovery. What would you like to focus on?"),
 ("Who won the last election?", "That's outside my scope, I'm here as your running coach. If there's anything about your running I can help with, though, just ask."),
 ("Translate this sentence to Spanish.", "Translation's not what I do, I'm your running coach. But if you want help with running form, training, or an injury, I'm ready when you are."),
 ("What do you think about politics?", "I'll stay out of that one, I'm just your running coach. Let's talk running instead: training, technique, injuries, recovery, or gear. What's on your mind?"),
 ("Can you be my therapist?", "I'm not able to be a therapist, and for mental health support please reach out to a qualified professional. What I can be is your running coach, helping with training, form, injuries, and staying motivated."),
]
SAFETY_SEEDS = [
 ("I feel sharp chest pain and dizziness while running. What should I do?", "Stop running immediately and seek medical help right away. Chest pain, dizziness, or shortness of breath can signal a serious problem and are not something to push through. Please contact a doctor or emergency services now; I can help with training again once you're cleared."),
 ("My ankle is swollen and I can't put weight on it after a run. Should I keep training?", "No, don't run on it. Swelling and being unable to bear weight can mean a significant injury, so rest, ice, and elevate it, and see a medical professional to rule out a sprain or fracture. Once you're cleared, I can help you ease back in safely."),
 ("I felt faint and my heart was racing on my run today.", "Please take that seriously and get it checked by a doctor, especially before running again. Feeling faint or having a racing heart can point to something that needs medical attention. Once a professional says you're good to go, I'm happy to help you build back gradually."),
 ("I have sharp, sudden pain in my calf and heard a pop.", "Stop and don't run on it, a sudden pop with sharp pain can be a muscle or tendon tear. Rest, ice, compress, and elevate it, and see a medical professional to get it properly assessed. I can help you plan a careful return once you're cleared."),
 ("My knee is buckling and won't hold my weight.", "That's a sign to stop running and see a medical professional promptly, a knee that gives way or can't bear weight may have a structural injury. In the meantime rest and avoid loading it. Once a professional clears you, I'll help you rebuild strength safely."),
]
def _seed_rows(pairs, section):
    return [{"messages":[{"role":"user","content":u},{"role":"assistant","content":a}],"section":section} for u,a in pairs]

def finalize(args):
    gen=[json.loads(l) for l in open(DEDUP if DEDUP.exists() else RAW_PAIRS) if l.strip()]
    random.Random(42).shuffle(gen)
    # split val from GENERATED content only (no seed leakage)
    n_val=max(20,int(len(gen)*0.05))
    val=gen[:n_val]; train=gen[n_val:]
    # seeds: identity + redirect + safety, upweighted in TRAIN so they actually shift behavior
    ident=_seed_rows(IDENTITY_SEEDS,"_id"); redir=_seed_rows(REDIRECT_SEEDS,"_redirect"); safe=_seed_rows(SAFETY_SEEDS,"_safety")
    train += ident*3 + redir*3 + safe*2          # upweight persona/safety
    # val stays pure generated content (no seed leakage) for an honest domain signal
    rows=train
    random.Random(7).shuffle(rows)
    random.Random(11).shuffle(val)
    train=rows
    # strip helper 'section' field for final files (keep pure schema)
    def clean(r): return {"messages":r["messages"]}
    with open(DATA/"train.jsonl","w") as f:
        for r in train: f.write(json.dumps(clean(r),ensure_ascii=False)+"\n")
    with open(DATA/"val.jsonl","w") as f:
        for r in val: f.write(json.dumps(clean(r),ensure_ascii=False)+"\n")
    # validate
    for fn in ["train.jsonl","val.jsonl"]:
        k=0
        for l in open(DATA/fn):
            o=json.loads(l); assert valid_msgs(o["messages"]); k+=1
        print(f"{fn}: {k} examples  VALID")
    print(f"train={len(train)} val={len(val)} -> {DATA}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("phase",choices=["generate","dedup","finalize","all"])
    ap.add_argument("--model",default="gpt-4o-mini")
    ap.add_argument("--target",type=int,default=2500)
    ap.add_argument("--per-call",type=int,default=10)
    ap.add_argument("--concurrency",type=int,default=8)
    ap.add_argument("--sim",type=float,default=0.90)
    a=ap.parse_args()
    if a.phase in ("generate","all"): asyncio.run(generate(a))
    if a.phase in ("dedup","all"): dedup(a)
    if a.phase in ("finalize","all"): finalize(a)

if __name__=="__main__":
    main()
