import re
import string

# --- Unified Prompt and Constants ---
# A single, versatile system prompt for all tasks.
ANSWER_START = "####"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow the user's instruction carefully and think step by step "
    "if the task is complex. For math or reasoning problems, provide the final answer after the #### tag."
)


# --- SFT Data Formatting Functions (for prepare_sft_data.py) ---

def format_dolly_sft(example: dict) -> dict:
    """Formats a Dolly example into the single-text format for SFT."""
    instruction = example['instruction']
    context = example.get('context')
    response = example['response']

    if context:
        full_text = f"{SYSTEM_PROMPT}\n\nUser: {instruction}\n\nContext: {context}\n\nAssistant: {response}"
    else:
        full_text = f"{SYSTEM_PROMPT}\n\nUser: {instruction}\n\nAssistant: {response}"
    return {"text": full_text}


def format_gsm8k_sft(example: dict) -> dict:
    """Formats a GSM8K example for SFT, ensuring it uses the unified prompt."""
    question = example['question']
    answer = example['answer']
    full_text = f"{SYSTEM_PROMPT}\n\nUser: {question}\n\nAssistant: {answer}"
    return {"text": full_text}


# --- RL Data and Reward Functions (for main.py) ---

def process_gsm8k(batch):
    """Formats a gsm8k batch for the RL trainer using the unified prompt."""
    prompts = [SYSTEM_PROMPT + "\n\nUser: " + q + "\nAssistant: " for q in batch["question"]]
    return {"prompt": prompts, "answer": [extract_hash_answer(a) for a in batch["answer"]]}


def process_qa(batch):
    """Formats a generic QA batch for the RL trainer using the unified prompt."""
    prompts = [SYSTEM_PROMPT + "\n\nUser: " + q + "\nAssistant: " for q in batch["question"]]
    return {"prompt": prompts, "answer": batch["answer"]}


def extract_from_response(text: str) -> str:
    """Extracts the final answer from a model's full response."""
    try:
        answer = text.split(ANSWER_START)[-1].strip()
        return answer[:-1].strip() if answer.endswith(".") else answer
    except IndexError:
        return ""


def extract_hash_answer(text: str) -> str | None:
    """Extracts the ground truth answer from the gsm8k dataset."""
    try:
        return text.split("####")[1].strip()
    except IndexError:
        return None


def get_reward_func(process_answer_func, efficiency_beta=0.01, is_math=True):
    def reward_func(completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]
        accuracy = []
        if is_math:
            ans = [process_answer_func(a) for a in answer]
            predictions = [process_answer_func(extract_from_response(r)) for r in responses]
            accuracy = [p == a for p, a in zip(predictions, ans)]
        else:
            for pred, ans_list in zip([extract_from_response(r) for r in responses], answer):
                pred_norm = process_answer_func(pred)
                if isinstance(ans_list, list):
                    cur_acc = any(process_answer_func(a) == pred_norm for a in ans_list)
                else:
                    cur_acc = process_answer_func(ans_list) == pred_norm
                accuracy.append(cur_acc)

        escaped_answer_start = re.escape(ANSWER_START)
        pattern = f"^(?:(?!{escaped_answer_start}).)*{escaped_answer_start}(?:(?!{escaped_answer_start}).)*$"
        matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]

        rewards = []
        for acc, match, resp in zip(accuracy, matches, responses):
            if acc and match:
                before_answer = resp.split(ANSWER_START)[0]
                num_words = len(before_answer.split())
                eff_penalty = efficiency_beta * (num_words / 200.0)
                reward = max(0.0, 1.0 - eff_penalty)
            else:
                reward = 0.0
            rewards.append(reward)

        print("=" * 50)
        print(f"\nBatch rewards: {[f'{r:.2f}' for r in rewards]}")
        print(f"\nSample response (answer: {answer[0]}):\n{responses[0]}")
        print("\n" + "=" * 50)
        return rewards
    return reward_func


def process_gsm8k_answer(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    matches = re.findall(r"-?\d*\.?\d+/?\d*", pred)
    if matches:
        last = matches[-1]
        try:
            return str(float(eval(last))) if '/' in last else last
        except:
            return last
    return ""


def process_qa_answer(pred: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(pred))))