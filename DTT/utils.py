import re
import string

# --- Unified Prompt and Constants ---
ANSWER_START = "####"

SYSTEM_PROMPT = (
    "You are a helpful assistant. Follow the user's instruction carefully and think step by step "
    "if the task is complex. For math or reasoning problems, provide the final answer after the #### tag."
)


# --- SFT Data Formatting Function (for prepare_sft_data.py) ---

def format_sft_example(example: dict) -> dict:
    """
    Formats an example for SFT. It combines the 'steps' and 'answer' fields
    into a single, high-quality assistant response, cleaning up any special characters.
    """
    question = example.get('question', '')
    steps = example.get('steps', '')
    answer = example.get('answer', '')
    
    # --- UPDATED LOGIC ---
    # Check if steps is a list and join it into a multi-line string,
    # while also cleaning each step.
    if isinstance(steps, list):
        # Remove the "<<...>>" characters from each step for cleaner training data
        cleaned_steps = [step.strip().replace("<<", "").replace(">>", "") for step in steps]
        steps = "\n".join(cleaned_steps)
    # --- END OF UPDATE ---
    
    # Construct the perfect assistant response in Chain-of-Thought format
    assistant_response = f"{steps}\n{ANSWER_START} {answer}"
    full_text = f"{SYSTEM_PROMPT}\n\nUser: {question}\n\nAssistant: {assistant_response}"
    return {"text": full_text}


# --- RL Data and Reward Functions (for main.py) ---

def process_rl_batch(batch: dict) -> dict:
    """Formats a batch for the RL trainer using the unified prompt."""
    prompts = [SYSTEM_PROMPT + "\n\nUser: " + q + "\nAssistant: " for q in batch["question"]]
    # The 'answer' field is the ground truth final answer for the reward function
    return {"prompt": prompts, "answer": batch["answer"]}


def extract_from_response(text: str) -> str:
    """Extracts the final answer from a model's full generation."""
    try:
        return text.split(ANSWER_START)[-1].strip()
    except IndexError:
        return ""


def get_reward_func(process_answer_func, efficiency_beta=0.01):
    def reward_func(completions: list[str], answer: list[str], **kwargs) -> list[float]:
        responses = completions
        ground_truths = [process_answer_func(str(ans)) for ans in answer]
        
        predictions_full = [extract_from_response(resp) for resp in responses]
        predictions = [process_answer_func(pred) for pred in predictions_full]
        
        accuracy = [p == gt for p, gt in zip(predictions, ground_truths) if p and gt]
        
        # Check if the model correctly used the '####' format
        escaped_answer_start = re.escape(ANSWER_START)
        pattern = f"^(?:(?!{escaped_answer_start}).)*{escaped_answer_start}"
        format_matches = [bool(re.search(pattern, r, re.DOTALL)) for r in responses]

        rewards = []
        for acc, match, resp in zip(accuracy, format_matches, responses):
            if acc and match:
                reasoning_text = resp.split(ANSWER_START)[0]
                num_words = len(reasoning_text.split())
                eff_penalty = efficiency_beta * (num_words / 200.0)
                reward = max(0.0, 1.0 - eff_penalty)
            else:
                reward = 0.0
            rewards.append(reward)

        print("=" * 50)
        print(f"\nBatch rewards: {[f'{r:.2f}' for r in rewards]}")
        if responses:
             print(f"\nSample response (ground truth answer: {answer[0]}):\n{responses[0]}")
        print("\n" + "=" * 50)
        return rewards
    return reward_func


def process_math_answer(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    matches = re.findall(r"-?\d*\.?\d+", pred)
    return matches[-1] if matches else ""


def process_qa_answer(pred: str) -> str:
    def remove_articles(text): return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text): return " ".join(text.split())
    def remove_punc(text): return "".join(ch for ch in text if ch not in set(string.punctuation))
    def lower(text): return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(pred))))