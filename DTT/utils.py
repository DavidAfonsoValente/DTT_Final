# utils.py
import re
import string

ANSWER_START = "####"

SYSTEM_PROMPT_GSM = (
    "You are a helpful assistant that solves math problems. Think step by step about the reasoning process, "
    "then provide the final answer after the #### tag."
)

SYSTEM_PROMPT_QA = (
    "You are a helpful assistant that answers questions. Think step by step about the reasoning process, "
    "then provide the final answer after the #### tag."
)

def extract_from_response(text: str) -> str:
    try:
        answer = text.split(ANSWER_START)[-1].strip()
        if answer.endswith("."):
            answer = answer[:-1].strip()
        return answer
    except IndexError:
        return ""

def extract_hash_answer(text: str) -> str | None:
    try:
        return text.split("####")[1].strip()
    except IndexError:
        return None

def get_reward_func(process_answer_func, efficiency_beta=0.01, is_math=True):
    def reward_func(completions, answer, **kwargs) -> list[float]:
        responses = [completion[0]["content"] for completion in completions]

        if is_math:
            ans = [process_answer_func(a) for a in answer]
            extracted = [extract_from_response(r) for r in responses]
            predictions = [process_answer_func(r) for r in extracted]
            accuracy = [p == a for p, a in zip(predictions, ans)]
        else:
            # For QA, allow multiple golden or fuzzy match
            accuracy = []
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
        for a, m, r in zip(accuracy, matches, responses):
            if a and m:
                # Efficiency penalty on reasoning length
                before_answer = r.split(ANSWER_START)[0]
                num_words = len(before_answer.split())
                eff_penalty = efficiency_beta * (num_words / 200.0)
                reward = 1.0 - eff_penalty
                reward = max(0.0, reward)
            else:
                reward = 0.0
            rewards.append(reward)

        print(
            "=" * 50,
            f"\nBatch rewards: {[f'{r:.2f}' for r in rewards]}",
            f"\nSample response (answer: {answer[0]}):\n{responses[0]}",
            "\n" + "=" * 50,
        )
        return rewards
    
    return reward_func

def process_gsm8k_answer(pred: str) -> str:
    pred = pred.strip("\n").rstrip(".").rstrip("/").strip(" ")
    # Improved numerical extraction
    matches = re.findall(r"-?\d*\.?\d+/?\d*", pred)
    if matches:
        last = matches[-1]
        try:
            # Normalize fraction to float str
            if '/' in last:
                num = eval(last)
                return str(float(num))
            else:
                return last
        except:
            return last
    return ""

def process_qa_answer(pred: str) -> str:
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(pred))))

def process_gsm8k(batch):
    prompts = [
        SYSTEM_PROMPT_GSM + "\n\nUser: " + q + "\nAssistant: "
        for q in batch["question"]
    ]
    return {
        "prompt": prompts,
        "answer": [extract_hash_answer(a) for a in batch["answer"]]
    }

def process_qa(batch):
    prompts = [
        SYSTEM_PROMPT_QA + "\n\nUser: " + q + "\nAssistant: "
        for q in batch["question"]
    ]
    return {
        "prompt": prompts,
        "answer": batch["answer"]  # Assume list of str or single str per batch entry
    }